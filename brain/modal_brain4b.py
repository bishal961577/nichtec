"""Growing brain test (reports/Growing brain 2026.md) on Qwen3.5-4B: one H100 per world, all three arms on it.

The usual way: deploy once, and the Claude session starts and collects the runs itself over HTTPS (its sandbox
cannot use the Modal command line, which speaks gRPC). On your computer, from the repository root:
    git pull
    modal deploy brain/modal_brain4b.py              # prints a URL like
                                                     # https://<your-workspace>--nichtec-brain4b-api.modal.run
Paste the URL in the chat. It uses the same password as the 27B (Modal secret nichtec-vllm-key). Do not write the
URL into the repository: it contains your workspace name. The endpoint costs nothing while idle; remove it when the
test is over with `modal app stop nichtec-brain4b`.
    POST /start  {"stage": "check" | "dev" | "exam", "commit": "<40-character hash>"}  -> {seed: call id}
                 one H100 job per world, running the experiment's code exactly as committed in the public repository
                 at that hash (fetched from GitHub, its sha256 recorded in the results), so fixes need a push, not
                 a redeploy, and the password cannot run anything that is not in the repository
    GET  /result/<call id>                           -> {"status": "running" | "done" | "failed", ...}
    POST /cancel/<call id>
Or run a stage directly from your computer instead:
    modal run brain/modal_brain4b.py --stage check   # one world, two days: the plumbing on a GPU (~10 min, ~$1)
    modal run brain/modal_brain4b.py --stage dev     # the two dev worlds (prompts and parsing may change after
                                                     # this; every change is logged in the report)
    modal run brain/modal_brain4b.py --stage exam    # the eight exam worlds, once, only after the FREEZE commit
Each world's results are written to results/growing_brain/<stage>/world_<seed>.json on your computer, and the summary
with the pre-registered verdict is printed and saved as summary.json.
Cost: an H100 is roughly $4 per hour and a world takes roughly 30-50 minutes, so about $3 per world: check ~$1,
dev ~$6, exam ~$25. Worlds run side by side, so the exam takes about an hour of waiting.
"""
import json
import os
import pathlib
import sys

import modal

BASE = "Qwen/Qwen3.5-4B"
HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.9.1", "transformers==5.20.2", "peft==0.21.1", "accelerate==1.11.0",
                      "flash-linear-attention==0.6.0", "fastapi[standard]==0.121.0")
         # flash-linear-attention: fast kernels for Qwen3.5's gated delta-net layers; without them transformers
         # falls back to plain PyTorch (slower), so the job still runs
         .add_local_file(HERE / "growing_brain.py", "/root/growing_brain.py")
         .add_local_file(HERE / "who_train.py", "/root/who_train.py")
         .add_local_file(HERE / "whoquiz.py", "/root/whoquiz.py"))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
app = modal.App("nichtec-brain4b")
REPO_RAW = "https://raw.githubusercontent.com/bishal961577/nichtec/{commit}/brain/{name}"
CODE = ("growing_brain.py", "who_train.py", "whoquiz.py")


def load_code(commit):
    """The experiment's code from the repository at an exact commit, or (no commit) the files baked in at deploy."""
    import hashlib
    import re
    import urllib.request
    d = "/root"
    if commit:
        assert re.fullmatch(r"[0-9a-f]{40}", commit), "commit must be a full 40-character hash"
        d = f"/tmp/code_{commit}"
        os.makedirs(d, exist_ok=True)
        for name in CODE:
            path = os.path.join(d, name)
            if not os.path.exists(path):
                urllib.request.urlretrieve(REPO_RAW.format(commit=commit, name=name), path)
    sys.path.insert(0, d)
    for name in CODE:
        sys.modules.pop(name[:-3], None)
    import growing_brain as B
    assert os.path.dirname(os.path.abspath(B.__file__)) == d, f"loaded {B.__file__}, not from {d}"
    sums = {name: hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest() for name in CODE}
    return B, sums


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache}, timeout=4 * 60 * 60)
def run_world(stage: str, seed: int, commit: str = "") -> str:
    import time
    B, sums = load_code(commit)
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    t0 = time.time()
    head = B.HFHead(BASE, log=log)
    w = B.make_world(seed)
    qs = ("Does it have scales?", "Is it active at night?", "Does the gate open for it?", "Does it lay eggs?",
          "Is it larger than a horse?", "Does it sing?")
    gap = head.batching_check([B.read_prompt(n, w["C"][n]["desc"], q) for n in w["kept"][:6] for q in qs])
    log(f"batched vs one at a time: {gap}")
    # right padding cannot leak into the text, so any gap is bf16 rounding; a real fault would flip confident answers
    assert gap["confident_flips"] == 0 and gap["max_gap"] < 1.0, f"batching changes the answers: {gap}"
    R = B.run_world(head, seed, log=log, days=2 if stage == "check" else None)
    R.update(model=BASE, stage=stage, commit=commit or "baked-in at deploy", code_sha256=sums, batching_gap=gap,
             minutes=round((time.time() - t0) / 60, 1), log=lines)
    return json.dumps(R)


@app.function(image=image, secrets=[modal.Secret.from_name("nichtec-vllm-key")], timeout=120)
@modal.asgi_app()
def api():
    """Start a stage's worlds and collect their results over HTTPS (Bearer = the nichtec-vllm-key password)."""
    from fastapi import FastAPI, Header, HTTPException
    web = FastAPI()
    not_ready = tuple({TimeoutError, getattr(modal.exception, "TimeoutError", TimeoutError)})

    def auth(header):
        if header != f"Bearer {os.environ['VLLM_API_KEY']}":
            raise HTTPException(status_code=401, detail="wrong password")

    @web.get("/health")
    def health():
        return {"ok": True}

    @web.post("/start")
    def start(body: dict, authorization: str = Header(None)):
        auth(authorization)
        import growing_brain as B
        stage, commit = body.get("stage"), body.get("commit", "")
        if stage not in B.STAGES:
            raise HTTPException(status_code=400, detail=f"stage must be one of {sorted(B.STAGES)}")
        seeds = body.get("seeds") or B.STAGES[stage]
        return {str(s): run_world.spawn(stage, int(s), commit).object_id for s in seeds}

    @web.get("/result/{call_id}")
    def result(call_id: str, authorization: str = Header(None)):
        auth(authorization)
        fc = modal.FunctionCall.from_id(call_id)
        try:
            return {"status": "done", "result": json.loads(fc.get(timeout=0))}
        except not_ready:
            return {"status": "running"}
        except Exception as e:                                   # the job itself failed: say how
            return {"status": "failed", "error": f"{type(e).__name__}: {e}"[:4000]}

    @web.post("/cancel/{call_id}")
    def cancel(call_id: str, authorization: str = Header(None)):
        auth(authorization)
        modal.FunctionCall.from_id(call_id).cancel()
        return {"cancelled": call_id}
    return web


@app.local_entrypoint()
def main(stage: str = "check"):
    import growing_brain as B
    seeds = B.STAGES[stage]
    out = HERE.parent / "results" / "growing_brain" / stage
    os.makedirs(out, exist_ok=True)
    print(f"{stage}: worlds {seeds}, written to {out}")
    res = []
    for js in run_world.starmap([(stage, s) for s in seeds], order_outputs=False):
        R = json.loads(js)
        (out / f"world_{R['seed']}.json").write_text(json.dumps(R))
        print(f"world {R['seed']} done in {R['minutes']} min")
        res.append(R)
    if stage == "check":
        for a in B.ARMS:
            print(a, [(x["day"], x["final"], x["rule"]) for x in R["arms"][a]])
        print("kept:", R["keeps"], "reading:", R["reading"])
        return
    S = B.summarise(sorted(res, key=lambda r: r["seed"]))
    B.report(S)
    (out / "summary.json").write_text(json.dumps({k: v for k, v in S.items() if k != "per_world"}, indent=1))
