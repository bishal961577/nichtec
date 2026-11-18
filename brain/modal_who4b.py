"""Step 8 (reports/Two ends check 2026.md): teach Qwen3.5-4B, by itself, who "she" is, and serve its answers.

The quiz is made inside the job from public chat (brain/whoquiz.py): a person named outright is hidden behind a
pronoun and the model must find them in the conversation. No person labels. Sources, pinned:
  - REALTALK chats 1-8 (github.com/danny911kr/REALTALK at 48b3ab88, Apache-2.0)
  - EmpatheticDialogues train split (Rashkin et al. 2019, CC BY-NC 4.0; tarball sha256 56f234d7...)
  - first names: the Kantrowitz-Ross names corpus (1991, free with credit; nltk_data names.zip, sha256 0eec7e95...)
Held out for checking: REALTALK chats 9-10 and EmpatheticDialogues' valid split. The test split is step 8's exam
and is never read here.

On your computer, from the repository root (after `git pull`):
    modal run brain/modal_who4b.py              # trains (about 15-25 minutes on one H100), then checks the
                                                # held-out quiz with and without the training; prints both scores
    modal deploy brain/modal_who4b.py           # serves the answers; prints a URL like
                                                # https://<your-workspace>--nichtec-who4b-serve.modal.run
    Paste the URL in the chat. It uses the same password as the 27B (Modal secret nichtec-vllm-key).
Stop it when step 8 is done (it also stops by itself after 5 idle minutes):
    modal app stop nichtec-who4b
Cost: an H100 is roughly $4 per hour and the L40S that serves roughly $2; expect about $2-4 in all.
Do not write the URL into the repository: it contains your workspace name.
"""
import json
import os
import pathlib

import modal

BASE = "Qwen/Qwen3.5-4B"
REALTALK = ("https://github.com/danny911kr/REALTALK.git", "48b3ab884a34baf659243d3188df478dd401a70a")
ED = ("https://dl.fbaipublicfiles.com/parlai/empatheticdialogues/empatheticdialogues.tar.gz",
      "56f234d77b7dd1f005fd365bb17769cfe346c3c84295b69bc069c8ccb83be03d")
NAMES = ("https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/names.zip",
         "0eec7e958b34982662b8f05824ae64642dea097b08057ade65c252191c5fe7ca")
ADAPTER = "/who/adapter_v1"
HERE = pathlib.Path(__file__).parent

image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git")
         .pip_install("torch==2.9.1", "transformers==5.20.2", "peft==0.21.1", "accelerate==1.11.0",
                      "fastapi[standard]==0.121.0", "flash-linear-attention==0.6.0")
         # flash-linear-attention: fast kernels for Qwen3.5's gated delta-net layers; if they fail to import,
         # transformers falls back to its plain PyTorch version (slower, more memory), so the job still runs
         .add_local_file(HERE / "whoquiz.py", "/root/whoquiz.py")
         .add_local_file(HERE / "who_train.py", "/root/who_train.py"))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
who_vol = modal.Volume.from_name("nichtec-who-model", create_if_missing=True)
app = modal.App("nichtec-who4b")


def fetch(url, sha, path):
    import hashlib
    import urllib.request
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    h = hashlib.sha256(open(path, "rb").read()).hexdigest()
    assert h == sha, f"{url}: sha256 {h} is not the pinned {sha}"
    return path


def sources(root="/tmp/src"):
    import subprocess
    import tarfile
    os.makedirs(root, exist_ok=True)
    rt = os.path.join(root, "REALTALK")
    if not os.path.exists(rt):
        subprocess.run(["git", "clone", "-q", REALTALK[0], rt], check=True)
    subprocess.run(["git", "-C", rt, "checkout", "-q", REALTALK[1]], check=True)
    tar = fetch(ED[0], ED[1], os.path.join(root, "ed.tar.gz"))
    with tarfile.open(tar) as t:
        t.extractall(root, members=[m for m in t.getmembers() if m.name.endswith(("train.csv", "valid.csv"))])
    names = fetch(NAMES[0], NAMES[1], os.path.join(root, "names.zip"))
    return os.path.join(rt, "data"), os.path.join(root, "empatheticdialogues"), names


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache, "/who": who_vol},
              timeout=3 * 60 * 60)
def train_and_check():
    import who_train as T
    import whoquiz as Q
    rt, ed, names = sources()
    Q.load_names(names)
    train = Q.quiz(Q.realtalk_convs(rt, range(1, 9)) + Q.ed_convs(os.path.join(ed, "train.csv")))
    held = Q.quiz(Q.realtalk_convs(rt, range(9, 11)) + Q.ed_convs(os.path.join(ed, "valid.csv")), seed=9)
    print(f"quiz: {len(train)} to learn from, {len(held)} held out")
    model, tok = T.train(BASE, train, ADAPTER, epochs=2, batch_tokens=8192)
    prompts = [Q.prompt_of(it) for it in held]
    tuned = T.answer(model, tok, prompts, batch=8)
    with model.disable_adapter():
        base = T.answer(model, tok, prompts, batch=8)
    gold = [it["answer"] for it in held]
    score = lambda pred: sum(Q.answer_match(p, g) for p, g in zip(pred, gold))
    res = {"train_items": len(train), "held_items": len(held), "base_right": score(base), "tuned_right": score(tuned),
           "held_unclear": sum(g == "unclear" for g in gold)}
    json.dump({"result": res, "held": [{"answer": g, "base": b, "tuned": t} for g, b, t in zip(gold, base, tuned)]},
              open("/who/check_v1.json", "w"), indent=0)
    who_vol.commit()
    print(json.dumps(res))
    return res


@app.function(image=image, gpu="L40S", volumes={"/root/.cache/huggingface": hf_cache, "/who": who_vol},
              secrets=[modal.Secret.from_name("nichtec-vllm-key")], scaledown_window=5 * 60, timeout=60 * 60,
              max_containers=1)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    import threading

    import torch
    from fastapi import FastAPI, Header, HTTPException
    from peft import PeftModel

    import who_train as T
    tok, base = T.load(BASE, torch.bfloat16, "cuda")
    model = PeftModel.from_pretrained(base, ADAPTER)
    lock = threading.Lock()
    web = FastAPI()

    @web.get("/health")
    def health():
        return {"ok": True, "base": BASE, "adapter": ADAPTER}

    @web.post("/who")
    def who(body: dict, authorization: str = Header(default="")):
        if authorization != f"Bearer {os.environ['VLLM_API_KEY']}":
            raise HTTPException(status_code=401, detail="unauthorised")
        prompts = body.get("prompts", [])[:64]
        with lock:                                     # one generation at a time on the GPU
            if body.get("adapter", True):
                return {"answers": T.answer(model, tok, prompts, batch=8)}
            with model.disable_adapter():
                return {"answers": T.answer(model, tok, prompts, batch=8)}
    return web


@app.local_entrypoint()
def main():
    print(train_and_check.remote())
