"""Real-world test 4: lifelong knowledge editing on MQuAKE (real Wikidata facts), nothing handed to the system.

Data: MQuAKE-CF-3k-v2 (Zhong et al., EMNLP 2023; the 2024 fixed version): 3,000 cases built from real
Wikidata facts, 2,764 distinct fact edits over 37 relations with open, multi-word answers ("Fernando
Santos is a citizen of" -> "United Kingdom"), each case with three multi-hop questions whose answer
changes because of the edits. MQuAKE-T holds real-world changes (e.g. New York City's head of
government -> Eric Adams). Calibration uses 563 other MQuAKE-CF cases sharing no entity with the test.

What the system is given: each edit request as the dataset states it (subject, cloze prompt, question,
new object). What it is NOT given at question time: where the subject is, which relation is asked,
which fact applies, or whether any fact applies. It must:
  1. find the subject itself: every word span of the question is encoded and compared with the
     subjects written so far (the memory stays silent when nothing matches),
  2. recognise the relation from the question wording (a classifier fitted on the calibration cases),
  3. read the value for that (subject, relation) from the memory layer, inside the forward pass.
The answer is then generated freely (greedy decoding) and scored by string match against the new
answer and its aliases, as in MQuAKE.

Edits arrive night by night; after each night the memory is re-solved. Compared on the same model,
data and scoring:
  base       no edits
  joint_ls   the exact joint memory (this project): ridge least squares over every edit so far
  batch_ls   the same slots, least squares on tonight's edits only (sequential batch editing, MEMIT-style)
  grace      GRACE-style codebook (Hartvigsen et al. 2023): key = the prompt's hidden state, value added
             when a query is within a radius
  rag        retrieval into the prompt: the same subject/relation lookup, but the matched fact is pasted
             in front of the question as text instead of read from the memory
Multi-hop questions are answered by chaining: the model writes sub-questions (few-shot prompt adapted from
MeLLo, Zhong et al. 2023) and each sub-question is answered by the method under test.

  python reallm/memtest4.py                                   # laptop: Qwen2.5-0.5B, all 2,764 edits
  python reallm/memtest4.py --model EleutherAI/gpt-j-6b       # the model the published MQuAKE numbers use
  python reallm/memtest4.py --data MQuAKE-T.json              # real-world temporal changes
  python reallm/memtest4.py --tiny                            # CPU smoke test
"""
import argparse
import copy
import json
import os
import random
import re
import sys
import time
import unicodedata
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtest as M  # noqa: E402
import memtest2 as M2  # noqa: E402
from memtest3 import pcg  # noqa: E402

ROOT = M.ROOT
URL = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/{}"

# Decomposition examples: MeLLo's published prompt (MQuAKE repository, prompts/MeLLo-prompt.txt), first two
# examples, with its retrieval lines removed because here each sub-question is answered by the method under test.
FEWSHOT = """Question: What is the capital city of the country of citizenship of Ivanka Trump's spouse?
Subquestion: Who is Ivanka Trump's spouse?
Answer: Jared Kushner
Subquestion: What is the country of citizenship of Jared Kushner?
Answer: Canada
Subquestion: What is the capital city of Canada?
Answer: Ottawa
Final answer: Ottawa

Question: Who is the head of state of the country where Rainn Wilson holds a citizenship?
Subquestion: What is the country of citizenship of Rainn Wilson?
Answer: Croatia
Subquestion: What is the name of the current head of state in Croatia?
Answer: Kolinda Grabar-Kitarovic
Final answer: Kolinda Grabar-Kitarovic

"""


def fetch(name):
    path = os.path.join(ROOT, "data", "mquake", name)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL.format(name), path)
    return json.load(open(path, encoding="utf-8"))


def norm(s):
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return " ".join(s.split())


def correct(pred, answers):
    p = " " + norm(pred) + " "
    return any(a and (" " + a + " ") in p for a in (norm(x) for x in answers))


def first_line(s):
    s = s.strip().split("\n")[0].strip()
    return s


def spans(text, max_words=8):
    words = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
    out = set()
    for i in range(len(words)):
        for j in range(i, min(len(words), i + max_words)):
            s = text[words[i][0]:words[j][1]].strip(" \t\n?.,!:;\"()[]")
            s = re.sub(r"(’s|'s)$", "", s).strip(" ,")
            if s and any(ch.isalnum() for ch in s):
                out.add(s)
    return sorted(out)


def parts(model):
    for dn in ("model", "transformer", "gpt_neox"):
        dec = getattr(model, dn, None)
        if dec is not None:
            for ln in ("layers", "h"):
                if hasattr(dec, ln):
                    return dec, getattr(dec, ln)
    raise ValueError("unsupported model layout")


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------

def edits_of(cases):
    """Distinct edits in order of first appearance, with the new answer's aliases."""
    out, seen, by_cloze = [], {}, {}
    for c in cases:
        for r in c["requested_rewrite"]:
            key = (r["subject"], r["prompt"])
            if key in seen:
                continue
            e = dict(subject=r["subject"], cloze=r["prompt"].format(r["subject"]), question=r["question"],
                     target=r["target_new"]["str"], true=r["target_true"]["str"], rel=r["relation_id"],
                     aliases={r["target_new"]["str"]})
            seen[key] = len(out)
            by_cloze.setdefault(e["cloze"], []).append(len(out))
            out.append(e)
        for h in c["new_single_hops"]:
            for k in by_cloze.get(h["cloze"], []):
                e = out[k]
                if norm(e["target"]) == norm(h["answer"]):
                    e["aliases"] |= set(h["answer_alias"]) | {h["answer"]}
    for e in out:
        e["aliases"] = sorted(e["aliases"])
    return out


def hop_facts(cases):
    """Every single-hop fact of the cases, with subject and relation (for calibration and locality)."""
    out = []
    for c in cases:
        for i, h in enumerate(c["single_hops"]):
            s, rel = c["orig"]["triples_labeled"][i][0], c["orig"]["triples"][i][1]
            nh = c["new_single_hops"][i]
            out.append(dict(subject=s, rel=rel, cloze=h["cloze"], question=h["question"], answer=h["answer"],
                            aliases=[h["answer"]] + h["answer_alias"], edited=norm(nh["answer"]) != norm(h["answer"])))
    return out


# ----------------------------------------------------------------------------------------------
# model plumbing
# ----------------------------------------------------------------------------------------------

class Runner4:
    def __init__(self, model, tok, dev, hook_layers):
        self.model, self.tok, self.dev = model, tok, dev
        self.base, self.blocks = parts(model)
        self.head = model.get_output_embeddings()
        self.d = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd
        self.state = {"mode": None}
        self.inject = "last"
        V = self.head.weight.shape[0]
        stop = [i for i in range(V) if "\n" in tok.decode([i])]
        if tok.eos_token_id is not None:
            stop.append(tok.eos_token_id)
        self.stop = torch.zeros(V, dtype=torch.bool, device=dev)
        self.stop[torch.tensor(sorted(set(stop)), dtype=torch.long, device=dev)] = True
        for l in sorted(set(hook_layers)):
            self.blocks[l].register_forward_hook(self._hook(l))

    def _hook(self, l):
        def hook(module, args, output):
            st = self.state
            mode = st["mode"]
            if mode is None or st.get("layer") != l:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if mode == "names":
                m = st["mask"][..., None].float()
                st["out"].append((h.float() * m).sum(1) / m.sum(1))
                raise M.Stop
            if mode == "last":
                st["out"].append(h[:, -1].float())
                raise M.Stop
            add = torch.zeros_like(h)
            if mode == "delta0":
                if st.get("inject") == "all":
                    add = st["delta"].to(h.dtype)[:, None, :] * st["smask"][..., None].to(h.dtype)
                else:
                    add[:, 0] = st["delta"].to(h.dtype)
            elif mode == "add_last":
                add[:, -1] = st["add"].to(h.dtype)
            elif mode == "grace":
                sims = F.normalize(h[:, -1].float(), dim=-1) @ st["keys"].T
                best, j = sims.max(1)
                on = (best >= st["theta"]).float()
                add[:, -1] = (st["vals"][j] * on[:, None]).to(h.dtype)
                st["fired"].append(on)
            else:
                return None
            h = h + add
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return hook

    def encode(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids, am = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
        return ids, am, (am.cumsum(-1) - 1).clamp(min=0)

    @torch.no_grad()
    def states(self, texts, layer, mode, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos = self.encode(texts[i:i + bs])
            self.state = {"mode": mode, "layer": layer, "mask": am, "out": outs}
            try:
                self.base(input_ids=ids, attention_mask=am, position_ids=pos)
            except M.Stop:
                pass
        self.state = {"mode": None}
        return torch.cat(outs) if outs else torch.zeros(0, self.d, device=self.dev)

    @torch.no_grad()
    def generate(self, texts, max_new, layer=None, adds=None, grace=None, bs=None):
        bs = bs or self.gen_bs
        outs, fired = [], []
        for i in range(0, len(texts), bs):
            ids, am, pos = self.encode(texts[i:i + bs])
            if adds is not None:
                self.state = {"mode": "add_last", "layer": layer, "add": adds[i:i + bs]}
            elif grace is not None:
                self.state = dict(grace, mode="grace", layer=layer, fired=fired)
            else:
                self.state = {"mode": None}
            o = self.base(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=True)
            if not (adds is not None and self.inject == "all"):
                self.state = {"mode": None}
            cache, p = o.past_key_values, pos[:, -1:]
            nxt = self.head(o.last_hidden_state[:, -1]).argmax(-1)
            gen = [nxt]
            done = self.stop[nxt].clone()
            for _ in range(max_new - 1):
                if bool(done.all()):
                    break
                am = torch.cat([am, torch.ones_like(am[:, :1])], 1)
                p = p + 1
                o = self.base(input_ids=nxt[:, None], attention_mask=am, position_ids=p, past_key_values=cache, use_cache=True)
                cache = o.past_key_values
                nxt = self.head(o.last_hidden_state[:, -1]).argmax(-1)
                gen.append(nxt)
                done |= self.stop[nxt]
            self.state = {"mode": None}
            for row in torch.stack(gen, 1).tolist():
                outs.append(self.tok.decode(row, skip_special_tokens=True))
            del cache, o
        self.fired = torch.cat(fired) if fired else None
        return outs

    def optimize_deltas(self, groups, targets, layer, steps, lr, wd, chunk=32, inject=None):
        inject = inject or self.inject
        """One vector per edit, added after block `layer` at the prompt's last position, that makes the frozen
        model produce the whole target (teacher forcing over all its tokens), shared by the edit's prompts."""
        P = len(groups[0])
        V = self.head.weight.shape[0]
        pad = self.tok.pad_token_id
        out = []
        for i in range(0, len(groups), chunk):
            G, T = groups[i:i + chunk], targets[i:i + chunk]
            nf = len(G)
            texts = [p for g in G for p in g]
            # the answer followed by a line break, so the stored value also says where the answer ends
            tg = [self.tok.encode(" " + t + "\n", add_special_tokens=False) for t in T for _ in range(P)]
            ids, am, pos = self.encode(texts)
            S = max(len(t) for t in tg)
            step = torch.full((len(texts), S), pad, device=self.dev)
            lab = torch.full((len(texts), S), -100, device=self.dev)
            smask = torch.zeros(len(texts), S, dtype=am.dtype, device=self.dev)
            for r, t in enumerate(tg):
                seq = [int(ids[r, -1])] + t[:-1]
                step[r, :len(seq)] = torch.tensor(seq, device=self.dev)
                lab[r, :len(t)] = torch.tensor(t, device=self.dev)
                smask[r, :len(seq)] = 1
            spos = pos[:, -1:] + torch.arange(S, device=self.dev)[None]
            full = torch.cat([am[:, :-1], smask], 1)
            with torch.no_grad():
                self.state = {"mode": None}
                cache0 = self.base(input_ids=ids[:, :-1], attention_mask=am[:, :-1], position_ids=pos[:, :-1],
                                   use_cache=True).past_key_values
            delta = torch.zeros(nf, self.d, device=self.dev, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            for _ in range(steps):
                cache = copy.deepcopy(cache0)
                self.state = {"mode": "delta0", "layer": layer, "delta": delta.repeat_interleave(P, 0), "inject": inject, "smask": smask}
                hs = self.base(input_ids=step, attention_mask=full, position_ids=spos, past_key_values=cache,
                               use_cache=True).last_hidden_state
                logits = self.head(hs).float()
                ce = F.cross_entropy(logits.reshape(-1, V), lab.reshape(-1), ignore_index=-100, reduction="none")
                ce = ce.view(len(texts), S).sum(1) / (lab != -100).sum(1)
                loss = (ce.view(nf, P).mean(1) + wd * (delta ** 2).sum(1)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                del cache
            self.state = {"mode": None}
            out.append(delta.detach())
        return torch.cat(out)


# ----------------------------------------------------------------------------------------------
# addressing: subjects found by span matching, relations by a classifier, (subject, relation) -> slots
# ----------------------------------------------------------------------------------------------

class Index:
    def __init__(self, run, a, Le, L, ent_proj, rel_proj, theta, dev, seed):
        self.run, self.Le, self.L, self.dev = run, Le, L, dev
        self.mu_e, self.P_e = ent_proj[0].to(dev), ent_proj[1].to(dev)
        self.mu_r, self.P_r, self.C = rel_proj[0].to(dev), rel_proj[1].to(dev), rel_proj[2].to(dev)
        self.theta = theta
        self.E = torch.zeros(0, a.qe, device=dev)
        self.subjects = []
        self.cache = {}
        g = torch.Generator().manual_seed(seed)
        nrel = self.C.shape[0]
        self.R = torch.stack([torch.linalg.qr(torch.randn(a.qe, a.qe, generator=g))[0] for _ in range(nrel)]).to(dev)
        k1, k2 = torch.randn(a.nsub, a.qe // 2, generator=g), torch.randn(a.nsub, a.qe // 2, generator=g)
        self.K1, self.K2 = F.normalize(k1, dim=1).to(dev), F.normalize(k2, dim=1).to(dev)
        self.nsub, self.K, self.ks, self.tau = a.nsub, a.topk, min(a.topk, a.nsub), a.tau
        self.N = a.nsub * a.nsub

    def keys(self, strings):
        new = sorted({s for s in strings if s not in self.cache})
        for i in range(0, len(new), 4096):
            chunk = new[i:i + 4096]
            q = F.normalize((self.run.states(chunk, self.Le, "names", bs=256) - self.mu_e) @ self.P_e, dim=1)
            for s, k in zip(chunk, q):
                self.cache[s] = k
        return torch.stack([self.cache[s] for s in strings]) if strings else torch.zeros(0, self.E.shape[1], device=self.dev)

    def add_subject(self, s):
        """Returns (index, merged_with_other_string)."""
        k = self.keys([s])
        if len(self.E):
            best, j = (k @ self.E.T).max(1)
            if float(best) >= self.theta:
                return int(j), self.subjects[int(j)] != s
        self.E = torch.cat([self.E, k])
        self.subjects.append(s)
        return len(self.subjects) - 1, False

    def detect(self, texts):
        """For each text: (subject index or -1, matched span)."""
        sp = [spans(t) for t in texts]
        flat = [s for ss in sp for s in ss]
        if not flat or len(self.E) == 0:
            return [(-1, None)] * len(texts)
        self.keys(flat)
        res = []
        for ss in sp:
            if not ss:
                res.append((-1, None))
                continue
            sims = torch.stack([self.cache[s] for s in ss]) @ self.E.T
            best, j = sims.max(1)
            ok = best >= self.theta
            if not bool(ok.any()):
                res.append((-1, None))
                continue
            cand = [(float(best[k]), len(ss[k]), k) for k in range(len(ss)) if bool(ok[k])]
            _, _, k = max(cand)
            res.append((int(j[k]), ss[k]))
        return res

    def rel_class(self, masked):
        z = (self.run.states(masked, self.L, "last") - self.mu_r) @ self.P_r
        return torch.cdist(z, self.C).argmin(1).tolist() if len(masked) else []

    def address(self, subj, rel):
        e = self.E[torch.tensor(subj, device=self.dev)]
        q = F.normalize(torch.einsum("nij,nj->ni", self.R[torch.tensor(rel, device=self.dev)], e), dim=1)
        h = q.shape[1] // 2
        v1, i1 = (q[:, :h] @ self.K1.T).topk(self.ks, dim=1)
        v2, i2 = (q[:, h:] @ self.K2.T).topk(self.ks, dim=1)
        c = (v1[:, :, None] + v2[:, None, :]).reshape(len(q), -1)
        sc, t = c.topk(self.K, dim=1)
        idx = i1.gather(1, t // self.ks) * self.nsub + i2.gather(1, t % self.ks)
        return idx, torch.softmax(sc / self.tau, dim=1)


def mask_subject(text, span):
    return text.replace(span, "X", 1) if span and span in text else text


# ----------------------------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------------------------

def load_model(args, extra_texts):
    if not args.tiny:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    else:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        words = set()
        for t in extra_texts:
            words.update(re.findall(r"\w+|[^\w\s]", t))
        vocab = {"[PAD]": 0, "[UNK]": 1, "\n": 2}
        for w in sorted(words):
            vocab.setdefault(w, len(vocab))
        tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                                             num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                             tie_word_embeddings=True))
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="MQuAKE-CF-3k-v2.json")
    ap.add_argument("--night", type=int, default=500, help="edits per night")
    ap.add_argument("--nsub", type=int, default=256)
    ap.add_argument("--qe", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--delta-steps", type=int, default=30)
    ap.add_argument("--delta-lr", type=float, default=0.2)
    ap.add_argument("--delta-wd", type=float, default=1e-3)
    ap.add_argument("--retain-n", type=int, default=300, help="first-night edits re-checked every night")
    ap.add_argument("--eval-edits", type=int, default=0, help="edits scored at the end (0 = all)")
    ap.add_argument("--mh-cases", type=int, default=0, help="multi-hop cases scored (0 = all)")
    ap.add_argument("--mh-q", type=int, default=3, help="questions per case (MQuAKE counts a case if any is right)")
    ap.add_argument("--methods", default="base,joint_ls,batch_ls,grace,rag")
    ap.add_argument("--mh-methods", default="base,joint_ls,batch_ls,grace,rag")
    ap.add_argument("--gen-bs", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--quick", action="store_true", help="real model, 300 cases: checks the pipeline end to end in minutes")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.quick:
        args.out = args.out or os.path.join(ROOT, "results", "reallm4_quick")
    if args.tiny:
        args.out = args.out or os.path.join(ROOT, "results", "reallm4_tiny")
    tag = os.path.splitext(args.data)[0].replace("MQuAKE-", "").lower()
    args.out = args.out or os.path.join(ROOT, "results", "reallm4" if tag == "cf-3k-v2" else f"reallm4_{tag}")
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    dev = args.device

    # ---- data: test cases, and calibration cases sharing no entity with any test file
    cases = fetch(args.data)
    cf = fetch("MQuAKE-CF.json")
    test_ents = set()
    for name in ("MQuAKE-CF-3k-v2.json", "MQuAKE-T.json"):
        for c in fetch(name):
            for t in c["orig"]["triples_labeled"] + c["orig"]["new_triples_labeled"]:
                test_ents.update((t[0], t[2]))
    calib_cases = [c for c in cf if not ({t[0] for t in c["orig"]["triples_labeled"] + c["orig"]["new_triples_labeled"]}
                                          | {t[2] for t in c["orig"]["triples_labeled"] + c["orig"]["new_triples_labeled"]}) & test_ents]
    if args.quick:
        cases = cases[:300]
        args.night, args.mh_cases, args.mh_q = 100, 100, 1
    if args.tiny:
        cases, calib_cases = cases[:40], calib_cases[:40]
        args.device = dev = "cpu"
        args.night, args.nsub, args.qe, args.topk, args.retain_n = 20, 32, 16, 8, 10
        args.delta_steps, args.mh_cases, args.mh_q, args.iters = 20, 6, 1, 200
    edits = edits_of(cases)
    calib_edits = edits_of(calib_cases)
    calib_hops = hop_facts(calib_cases)
    log(f"{args.data}: {len(cases)} cases, {len(edits)} distinct edits; calibration: {len(calib_cases)} cases, "
        f"{len(calib_edits)} edits, {len(calib_hops)} single-hop facts")
    texts_for_vocab = []
    if args.tiny:
        for c in cases + calib_cases:
            texts_for_vocab += c["questions"] + [h["question"] + " " + h["cloze"] + " " + " ".join(h["answer_alias"]) + " " + h["answer"]
                                                 for h in c["single_hops"] + c["new_single_hops"]]
            texts_for_vocab += [r["prompt"] + " " + r["subject"] + " " + r["target_new"]["str"] for r in c["requested_rewrite"]]
        texts_for_vocab += [FEWSHOT] + [p + " " + a + " " + s for p, a, s in M2.KNOWN] + ["Q: A: Fact: X"]
    model, tok = load_model(args, texts_for_vocab)
    run0 = Runner4(model, tok, dev, [])
    nl = len(run0.blocks)
    Lc = sorted({max(1, int(round(nl * f)) - 1) for f in (0.375, 0.5, 0.625)})
    Lec = sorted({max(0, int(round(nl * f)) - 1) for f in (1 / 6, 1 / 4, 1 / 3)})
    run = Runner4(model, tok, dev, Lc + Lec)
    run.gen_bs = args.gen_bs
    R = {"config": dict(vars(args), model="tiny-random" if args.tiny else args.model), "n_layers": nl, "d": run.d,
         "edits": len(edits), "cases": len(cases), "calibration_cases": len(calib_cases), "nights": []}
    qa = lambda q: f"Q: {q}\nA:"

    # ---- person key: names encoded alone, whitened on calibration entity names; threshold from them
    cal_names = sorted({h["subject"] for h in calib_hops} | {e["subject"] for e in calib_edits} | {e["target"] for e in calib_edits})
    best = None
    for l in Lec:
        H = run.states(cal_names, l, "names", bs=256).double().cpu()
        Hb = run.states(cal_names[::-1], l, "names", bs=7).double().cpu().flip(0)
        mu = H.mean(0)
        ev, U = torch.linalg.eigh(((H - mu).T @ (H - mu)) / len(H))
        Pe = (U[:, -args.qe:] / ev[-args.qe:].clamp(min=1e-9).sqrt()).float()
        q = F.normalize((H.float() - mu.float()) @ Pe, dim=1)
        qb = F.normalize((Hb.float() - mu.float()) @ Pe, dim=1)
        same = (q * qb).sum(1)
        s = q @ q.T
        s.fill_diagonal_(-2)
        nn = s.max(1).values
        gap = float(same.min() - nn.max())
        if best is None or gap > best[0]:
            best = (gap, l, mu.float(), Pe, same, nn)
    _, Le, mu_e, P_e, same, nn = best
    lo, hi = float(same.min()), float(nn.max())
    theta = (lo + hi) / 2 if lo > hi else float(torch.quantile(nn, 0.999))
    R["person_key"] = {"block": Le, "theta": round(theta, 5), "same_name_min_cos": round(lo, 5), "different_names_max_cos": round(hi, 5),
                       "calibration_names": len(cal_names), "different_names_passing": round(float((nn >= theta).float().mean()), 5)}
    log("person key", json.dumps(R["person_key"]))

    # ---- memory layer: pick the block whose injected targets transfer best from cloze to question form
    cal = calib_edits[: (8 if args.tiny else 48)]
    choice = {}
    for L in Lc:
        for inj in ("last", "all"):
            run.inject = inj
            T = run.optimize_deltas([[e["cloze"]] for e in cal], [e["target"] for e in cal], L, args.delta_steps, args.delta_lr, args.delta_wd)
            g1 = run.generate([e["cloze"] for e in cal], 10, L, adds=T)
            g2 = run.generate([qa(e["question"]) for e in cal], 10, L, adds=T)
            choice[(L, inj)] = {"cloze": np.mean([correct(first_line(g), e["aliases"]) for g, e in zip(g1, cal)]),
                                "question_transfer": np.mean([correct(first_line(g), e["aliases"]) for g, e in zip(g2, cal)])}
    L, inj = max(choice, key=lambda k: (choice[k]["cloze"] + choice[k]["question_transfer"], k[1] == "last", -k[0]))
    run.inject = inj
    R["memory_block"] = {"block": L, "inject": inj,
                         "candidates": {f"{k[0]}/{k[1]}": {kk: round(float(vv), 3) for kk, vv in v.items()} for k, v in choice.items()}}
    log("memory block", json.dumps(R["memory_block"]))

    # ---- relation classifier on calibration single-hop facts (cloze and question, subject masked)
    rel_ids = sorted({h["rel"] for h in calib_hops})
    rtexts, rlab = [], []
    for h in calib_hops:
        for t in (h["cloze"], h["question"]):
            if h["subject"] in t:
                rtexts.append(mask_subject(t, h["subject"]))
                rlab.append(rel_ids.index(h["rel"]))
    Hr = run.states(rtexts, L, "last")
    lab = torch.tensor(rlab)
    present = sorted(set(rlab))
    remap = {c: i for i, c in enumerate(present)}
    lab = torch.tensor([remap[c] for c in rlab])
    mu_r, P_r, _ = M.fit_projection(Hr, lab, max(1, len(present) - 1), args.seed, "lda")
    Z = (Hr.float().cpu() - mu_r) @ P_r
    Cent = torch.stack([Z[lab == i].mean(0) for i in range(len(present))])
    idx = Index(run, args, Le, L, (mu_e, P_e), (mu_r, P_r, Cent), theta, dev, args.seed + 1)
    # held-out check of the classifier: calibration edits' own cloze and question
    chk = [(mask_subject(t, e["subject"]), e["rel"]) for e in calib_edits for t in (e["cloze"], e["question"]) if e["rel"] in rel_ids]
    pred = idx.rel_class([t for t, _ in chk])
    R["relation_classifier"] = {"relations": len(present), "training_texts": len(rtexts),
                                "calibration_edit_accuracy": round(float(np.mean([present[p] == rel_ids.index(r) for p, (_, r) in zip(pred, chk)])), 4)}
    log("relation classifier", json.dumps(R["relation_classifier"]))

    # ---- GRACE radius: 5% of different calibration facts' questions may fire on a key
    kc = F.normalize(run.states([qa(e["question"]) for e in calib_edits], L, "last"), dim=1)
    kk = F.normalize(run.states([e["cloze"] for e in calib_edits], L, "last"), dim=1)
    sim = kc @ torch.cat([kk, kc]).T
    n = len(calib_edits)
    for i in range(n):
        sim[i, i] = -2
        sim[i, n + i] = -2
    grace_theta = float(torch.quantile(sim.max(1).values, 0.95))
    R["grace_theta"] = round(grace_theta, 4)

    # ---- known facts the base model answers correctly (damage check)
    known = [(p, a) for p, a, _ in M2.KNOWN]
    kg = run.generate([p for p, _ in known], 6)
    known = [(p, a) for (p, a), g in zip(known, kg) if correct(first_line(g), [a])] if not args.tiny else known[:10]
    R["known_facts_base_correct"] = len(known)

    # ---- the stream
    methods = args.methods.split(",")
    V = {m: torch.zeros(idx.N, run.d) for m in ("joint_ls", "batch_ls") if m in methods}
    fact_of = {}             # (subject index, relation class) -> fact id
    facts = []               # per fact: dict(edit, subj, rel, idx, w)
    T_all = torch.zeros(0, run.d, device=dev)
    active = []              # fact ids currently stored (overwritten ones drop out)
    gkeys, gvals = [], []
    stats = {"subject_merges": 0, "write_collisions": 0}
    early = None

    def lookup(texts):
        """Fact id per text, or -1: subject found by span matching, relation classified with the subject masked."""
        det = idx.detect(texts)
        masked = [mask_subject(t, sp) for t, (j, sp) in zip(texts, det)]
        rel = idx.rel_class(masked)
        return [fact_of.get((j, c), -1) if j >= 0 else -1 for (j, _), c in zip(det, rel)]

    def reads(fids, Vm):
        add = torch.zeros(len(fids), run.d, device=dev)
        hit = [k for k, f in enumerate(fids) if f >= 0]
        if hit:
            ix = torch.stack([facts[fids[k]]["idx"] for k in hit])
            ww = torch.stack([facts[fids[k]]["w"] for k in hit])
            add[hit] = F.embedding_bag(ix, Vm, per_sample_weights=ww, mode="sum")
        return add

    def answer(method, prompts, det_texts, max_new=10, Vd=None):
        """Generate with a method; det_texts are the question/cloze texts the lookup sees."""
        if method == "base":
            return run.generate(prompts, max_new)
        if method in ("joint_ls", "batch_ls"):
            return run.generate(prompts, max_new, L, adds=reads(lookup(det_texts), Vd[method]))
        if method == "grace":
            if not gkeys:
                return run.generate(prompts, max_new)
            return run.generate(prompts, max_new, L, grace={"keys": torch.cat(gkeys), "vals": torch.cat(gvals), "theta": grace_theta})
        if method == "rag":
            fids = lookup(det_texts)
            pre = [f"Fact: {facts[f]['edit']['cloze']} {facts[f]['edit']['target']}.\n" if f >= 0 else "" for f in fids]
            return run.generate([a + p for a, p in zip(pre, prompts)], max_new)
        raise ValueError(method)

    def edit_scores(es, Vd, ms, forms=("cloze", "question")):
        out = {}
        for m in ms:
            r = {}
            for form in forms:
                prompts = [e["cloze"] if form == "cloze" else qa(e["question"]) for e in es]
                det = [e["cloze"] if form == "cloze" else e["question"] for e in es]
                g = answer(m, prompts, det, Vd=Vd)
                r[form] = round(float(np.mean([correct(first_line(x), e["aliases"]) for x, e in zip(g, es)])), 4)
            out[m] = r
        return out

    for s in range(0, len(edits), args.night):
        tn = time.time()
        new = edits[s: s + args.night]
        new_ids = []
        idx.keys([e["subject"] for e in new])
        new_rel = idx.rel_class([mask_subject(e["cloze"], e["subject"]) for e in new])
        for e, c in zip(new, new_rel):
            j, merged = idx.add_subject(e["subject"])
            stats["subject_merges"] += int(merged)
            if (j, c) in fact_of:
                stats["write_collisions"] += 1
                old = fact_of[(j, c)]
                if old in active:
                    active.remove(old)
            f = len(facts)
            facts.append({"edit": e, "subj": j, "rel": c})
            fact_of[(j, c)] = f
            active.append(f)
            new_ids.append(f)
        ix, ww = idx.address([facts[f]["subj"] for f in new_ids], [facts[f]["rel"] for f in new_ids])
        for k, f in enumerate(new_ids):
            facts[f]["idx"], facts[f]["w"] = ix[k], ww[k]
        Tn = run.optimize_deltas([[e["cloze"], qa(e["question"])] for e in new], [e["target"] for e in new], L,
                                 args.delta_steps, args.delta_lr, args.delta_wd)
        T_all = torch.cat([T_all, Tn])
        t_targets = time.time() - tn
        if "grace" in methods:
            gkeys.append(F.normalize(run.states([e["cloze"] for e in new] + [qa(e["question"]) for e in new], L, "last"), dim=1))
            gvals.append(torch.cat([Tn, Tn]))
        act = torch.tensor(active, device=dev)
        rows_all = M.Rows(torch.stack([facts[f]["idx"] for f in active]), torch.stack([facts[f]["w"] for f in active]), idx.N)
        live = set(active)
        live_new = [f for f in new_ids if f in live]
        rows_new = M.Rows(torch.stack([facts[f]["idx"] for f in live_new]), torch.stack([facts[f]["w"] for f in live_new]), idx.N) if live_new else None
        row = {"night": s // args.night + 1, "edits": s + len(new), "facts_stored": len(active), **stats,
               "distinct_slots_touched": int(torch.unique(rows_all.idx).numel()), "target_s": round(t_targets, 1), "methods": {}}
        Vd = {}
        for m in V:
            tm = time.time()
            Vm = V[m].to(dev)
            if m == "joint_ls":
                Vm, it = pcg(rows_all, T_all[act], Vm, args.lam, args.iters, 1e-4)
            elif live_new:
                Vm, it = pcg(rows_new, T_all[torch.tensor(live_new, device=dev)], Vm, args.anchor, args.iters, 1e-4, prior=Vm)
            else:
                it = 0
            rel_err = M.certificate(rows_all, Vm, T_all[act])
            row["methods"][m] = {"iters": it, "solve_s": round(time.time() - tm, 1),
                                 "cert_frac_over_0.2": round(float((rel_err > 0.2).float().mean()), 4)}
            Vd[m] = Vm
        if early is None:
            early = new[: args.retain_n]
            R["first_night_before_editing"] = edit_scores(early, Vd, ["base"])
            T_or = Tn[: len(early)]
            g1 = run.generate([e["cloze"] for e in early], 10, L, adds=T_or)
            g2 = run.generate([qa(e["question"]) for e in early], 10, L, adds=T_or)
            R["first_night_target_injected_directly"] = {
                "cloze": round(float(np.mean([correct(first_line(g), e["aliases"]) for g, e in zip(g1, early)])), 4),
                "question": round(float(np.mean([correct(first_line(g), e["aliases"]) for g, e in zip(g2, early)])), 4)}
            log("first night: before editing", json.dumps(R["first_night_before_editing"]), "| target injected directly",
                json.dumps(R["first_night_target_injected_directly"]))
        retain = edit_scores(early, Vd, [m for m in methods if m in ("joint_ls", "batch_ls", "grace")], forms=("cloze",))
        for m, r in retain.items():
            row["methods"].setdefault(m, {})["first_night_cloze"] = r["cloze"]
        for m in V:
            V[m] = Vd[m].cpu()
        if dev == "cuda":
            row["max_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            torch.cuda.empty_cache()
        row["night_s"] = round(time.time() - tn, 1)
        R["nights"].append(row)
        log(json.dumps(row))
        json.dump(R, open(os.path.join(args.out, "result.json"), "w"), indent=1)

    Vd = {m: V[m].to(dev) for m in V}
    out_json = os.path.join(args.out, "result.json")
    # ---- final: every edit, both forms
    es = edits if not args.eval_edits else rng.sample(edits, min(args.eval_edits, len(edits)))
    R["final_edits"] = {"n": len(es), **edit_scores(es, Vd, methods)}
    # how often the lookup finds the right fact for the question form (no subject given)
    fids = lookup([e["question"] for e in es])
    R["final_edits"]["lookup_right_fact_question_form"] = round(float(np.mean([fid >= 0 and facts[fid]["edit"] is e for fid, e in zip(fids, es)])), 4)
    log("final edits", json.dumps(R["final_edits"]))
    json.dump(R, open(out_json, "w"), indent=1)

    # ---- locality: unedited facts in the test cases (incl. unedited relations of edited subjects), known facts
    hops = [h for h in hop_facts(cases) if not h["edited"]]
    seen, loc = set(), []
    for h in hops:
        if h["question"] not in seen:
            seen.add(h["question"])
            loc.append(h)
    if args.tiny:
        loc = loc[:20]
    elif len(loc) > 2000:
        loc = rng.sample(loc, 2000)
    edited_subjects = {e["subject"] for e in edits}
    base_loc = answer("base", [qa(h["question"]) for h in loc], [h["question"] for h in loc])
    R["locality"] = {"unedited_facts": len(loc), "of_which_about_edited_subjects": sum(h["subject"] in edited_subjects for h in loc)}
    for m in methods:
        if m == "base":
            continue
        g = answer(m, [qa(h["question"]) for h in loc], [h["question"] for h in loc], Vd=Vd)
        same_out = [first_line(a) == first_line(b) for a, b in zip(g, base_loc)]
        same_sub = [x for x, h in zip(same_out, loc) if h["subject"] in edited_subjects]
        gk = answer(m, [p for p, _ in known], [p for p, _ in known], max_new=6, Vd=Vd)
        R["locality"][m] = {"unchanged": round(float(np.mean(same_out)), 4),
                            "unchanged_same_subject_other_relation": round(float(np.mean(same_sub)), 4) if same_sub else None,
                            "known_facts_correct": round(float(np.mean([correct(first_line(x), [a]) for x, (_, a) in zip(gk, known)])), 4)}
    R["locality"]["base_accuracy_on_unedited"] = round(float(np.mean([correct(first_line(x), h["aliases"]) for x, h in zip(base_loc, loc)])), 4)
    log("locality", json.dumps(R["locality"]))
    json.dump(R, open(out_json, "w"), indent=1)

    # ---- multi-hop: direct question, and chained sub-questions answered by each method
    mh = cases if not args.mh_cases else cases[: args.mh_cases]
    Q = [(ci, q) for ci, c in enumerate(mh) for q in c["questions"][: args.mh_q]]
    R["multihop"] = {"cases": len(mh), "questions_per_case": args.mh_q}
    qs_of = {}
    for k, (ci, _) in enumerate(Q):
        qs_of.setdefault(ci, []).append(k)
    for m in [x for x in args.mh_methods.split(",") if x in methods]:
        g = answer(m, [qa(q) for _, q in Q], [q for _, q in Q], max_new=12, Vd=Vd)
        direct = [correct(first_line(x), [mh[ci]["new_answer"]] + mh[ci]["new_answer_alias"]) for (ci, _), x in zip(Q, g)]
        chains = chain(run, m, [q for _, q in Q], answer, Vd)
        ok = [correct(a, [mh[ci]["new_answer"]] + mh[ci]["new_answer_alias"]) for (ci, _), a in zip(Q, chains)]
        old = [correct(a, [mh[ci]["answer"]] + mh[ci]["answer_alias"]) for (ci, _), a in zip(Q, chains)]
        by_case = lambda v: float(np.mean([any(v[k] for k in qs_of[c]) for c in range(len(mh))]))
        R["multihop"][m] = {"chain_case_accuracy": round(by_case(ok), 4), "chain_question_accuracy": round(float(np.mean(ok)), 4),
                            "chain_gave_old_answer": round(by_case(old), 4), "direct_case_accuracy": round(by_case(direct), 4)}
        if m == "base":
            R["multihop"]["base"]["chain_case_accuracy_on_ORIGINAL_answers"] = R["multihop"]["base"].pop("chain_gave_old_answer")
        log("multi-hop", m, json.dumps(R["multihop"][m]))
        json.dump(R, open(out_json, "w"), indent=1)
    R["total_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_json, "w"), indent=1)
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write(report(R))
    log("done", out_json)


def chain(run, method, questions, answer, Vd, max_steps=5):
    """Model-written sub-questions (few-shot), each answered by the method; returns final answers."""
    state = [FEWSHOT + f"Question: {q}\n" for q in questions]
    last = [""] * len(questions)
    final = [None] * len(questions)
    for _ in range(max_steps):
        act = [i for i in range(len(questions)) if final[i] is None]
        if not act:
            break
        lines = [first_line(x) for x in run.generate([state[i] for i in act], 32)]
        sub = []
        for i, line in zip(act, lines):
            if line.startswith("Final answer:"):
                final[i] = line[len("Final answer:"):].strip()
            elif line.startswith("Subquestion:") and len(line) > len("Subquestion:") + 3:
                state[i] += line + "\n"
                sub.append((i, line[len("Subquestion:"):].strip()))
            else:
                final[i] = last[i] or line
        if sub:
            got = answer(method, [f"Q: {q}\nA:" for _, q in sub], [q for _, q in sub], max_new=12, Vd=Vd)
            for (i, _), a in zip(sub, got):
                a = first_line(a).rstrip(".")
                last[i] = a
                state[i] += f"Answer: {a}\n"
    return [f if f is not None else l for f, l in zip(final, last)]


def report(R):
    ms = list(R["final_edits"].keys() - {"n", "lookup_right_fact_question_form"})
    order = [m for m in ("base", "joint_ls", "batch_ls", "grace", "rag") if m in ms]
    L = [f"# Real-world test 4: MQuAKE lifelong editing — {R['config']['model']} ({R['config']['data']})", "",
         f"{R['edits']:,} distinct edits from {R['cases']:,} cases, written {R['config']['night']} per night; "
         f"memory after block {R['memory_block']['block']} of {R['n_layers']} (value added at: {R['memory_block']['inject']} position(s)); calibration on {R['calibration_cases']} disjoint cases.", "",
         "At question time nothing is given: the subject is found by matching every word span, the relation by a classifier.", "",
         "## Every edit after the last night (new answer generated; string match against answer + aliases)", "",
         "| Method | Cloze (written form) | Question form |", "|---|---|---|"]
    for m in order:
        L.append(f"| {m} | {R['final_edits'][m]['cloze']:.3f} | {R['final_edits'][m]['question']:.3f} |")
    L += ["", f"Lookup found the right fact from the question alone: {R['final_edits']['lookup_right_fact_question_form']:.3f}", "",
          "## First night's edits after each night (cloze)", "", "| Edits | " + " | ".join(m for m in ("joint_ls", "batch_ls", "grace") if m in order) + " |",
          "|---|" + "---|" * len([m for m in ("joint_ls", "batch_ls", "grace") if m in order])]
    for n in R["nights"]:
        L.append(f"| {n['edits']:,} | " + " | ".join(f"{n['methods'].get(m, {}).get('first_night_cloze', float('nan')):.3f}"
                                                     for m in ("joint_ls", "batch_ls", "grace") if m in order) + " |")
    L += ["", "## Locality (unedited facts; identical output to the unedited model)", "", "```", json.dumps(R["locality"], indent=1), "```", "",
          "## Multi-hop (MQuAKE: a case counts if any of its questions is answered with the new answer)", "", "```",
          json.dumps(R["multihop"], indent=1), "```", "",
          "Published on GPT-J, MQuAKE-CF, 3,000 edits (Zhong et al. 2023): MeLLo 14.2%, MEMIT 5.4% multi-hop accuracy.", "",
          "Setup: " + json.dumps({k: R[k] for k in ("person_key", "memory_block", "relation_classifier", "grace_theta", "known_facts_base_correct")}), "",
          "First night, before editing / target injected directly: " + json.dumps(R["first_night_before_editing"]) + " / "
          + json.dumps(R["first_night_target_injected_directly"]), "", f"Total time: {R.get('total_s', 0) / 60:.1f} min"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
