"""Step 8: teach a small model who "she" is from the chat's own quiz (brain/whoquiz.py). LoRA, nothing else.

Used by brain/modal_who4b.py on a GPU, and by a CPU smoke test with a tiny model:
    python brain/who_train.py --smoke --model Qwen/Qwen3.5-0.8B --quiz QUIZ.jsonl
The answer is the only text the model is trained to write (the prompt's tokens carry no loss). Thinking is off.
"""
import argparse
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import whoquiz as Q  # noqa: E402

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05)
# language-model linear layers of Qwen3.5 (full attention, gated delta net, MLP); the vision tower is left alone
TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkvz|in_proj_ba|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj|gate_proj|up_proj|down_proj)$"


def chat_prefix(tok, prompt):
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def encode(tok, item, max_len):
    pre = tok(chat_prefix(tok, Q.prompt_of(item)), add_special_tokens=False)["input_ids"]
    ans = tok(item["answer"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    if len(pre) + len(ans) > max_len:                  # keep the latest message: cut the oldest context first
        pre = pre[-(max_len - len(ans)):]
    return pre + ans, [-100] * len(pre) + ans


def load(model_name, dtype, device):
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype).to(device)
    return tok, model


def train(model_name, items, out_dir, epochs=2, lr=1e-4, batch_tokens=16384, max_len=1536, device="cuda",
          dtype="bfloat16", seed=8, smoke_steps=None, log=print):
    import torch
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    tok, model = load(model_name, getattr(torch, dtype), device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(target_modules=TARGETS, task_type="CAUSAL_LM", **LORA))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA parameters: {n_train / 1e6:.1f} M")
    data = [encode(tok, it, max_len) for it in items]
    rng = random.Random(seed)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
    # batches of similar length, about batch_tokens tokens each
    order = sorted(range(len(data)), key=lambda i: len(data[i][0]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = len(data[i][0])
        if cur and max(cur_max, L) * (len(cur) + 1) > batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    total = epochs * len(batches) if not smoke_steps else smoke_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, total // 20)) *
                                              0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    model.train()
    step, t0 = 0, time.time()
    for ep in range(epochs):
        rng.shuffle(batches)
        for b in batches:
            L = max(len(data[i][0]) for i in b)
            ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
            lab = torch.full((len(b), L), -100, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for r, i in enumerate(b):                  # right padding for training: the padding comes after
                x, y = data[i]
                ids[r, :len(x)] = torch.tensor(x)
                lab[r, :len(y)] = torch.tensor(y)
                att[r, :len(x)] = 1
            out = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=lab.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 10 == 0 or step == 1:
                log(f"epoch {ep + 1} step {step}/{total} loss {out.loss.item():.4f} ({(time.time() - t0) / 60:.1f} min)")
            if smoke_steps and step >= smoke_steps:
                break
        if smoke_steps and step >= smoke_steps:
            break
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    log(f"saved adapter to {out_dir} after {step} steps ({(time.time() - t0) / 60:.1f} min)")
    return model, tok


def answer(model, tok, prompts, device="cuda", batch=1, max_new=24):
    """Greedy answers, one prompt at a time unless batch > 1 (left padding; checked equal on CPU before use)."""
    import torch
    outs = []
    model.eval()
    for k in range(0, len(prompts), batch):
        chunk = [chat_prefix(tok, p) for p in prompts[k:k + batch]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for row in g[:, enc["input_ids"].shape[1]:]:
            outs.append(tok.decode(row, skip_special_tokens=True).strip().split("\n")[0])
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--quiz", required=True)
    ap.add_argument("--out", default="/tmp/who_adapter_smoke")
    a = ap.parse_args()
    items = [json.loads(x) for x in open(a.quiz, encoding="utf-8")]
    if a.smoke:
        import torch
        torch.set_num_threads(4)
        model, tok = train(a.model, items[:24], a.out, epochs=1, batch_tokens=1024, max_len=512, device="cpu",
                           dtype="float32", smoke_steps=2)
        ps = [Q.prompt_of(it) for it in items[24:28]]
        one = answer(model, tok, ps, device="cpu", batch=1)
        four = answer(model, tok, ps, device="cpu", batch=4)
        print("one at a time:", one)
        print("batched (left padding):", four, "same" if one == four else "DIFFERENT")
        print("gold:", [it["answer"] for it in items[24:28]])


if __name__ == "__main__":
    main()
