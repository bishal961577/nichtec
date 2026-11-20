"""Growing brain test (reports/Growing brain 2026.md) on Qwen3.5-4B: one H100 per world, all three arms on it.

On your computer, from the repository root (after `git pull`):
    modal run brain/modal_brain4b.py --stage check   # one world, two days: the plumbing on a GPU (~10 min, ~$1)
    modal run brain/modal_brain4b.py --stage dev     # the two dev worlds (prompts and parsing may change after
                                                     # this; every change is logged in the report)
    modal run brain/modal_brain4b.py --stage exam    # the eight exam worlds, once, only after the FREEZE commit
Each world's results are written to results/growing_brain/<stage>/world_<seed>.json on your computer, and the summary
with the pre-registered verdict is printed and saved as summary.json. Commit and push that folder (or paste the
printed summary in the chat).
Cost: an H100 is roughly $4 per hour and a world takes roughly 30-50 minutes, so about $3 per world: check ~$1,
dev ~$6, exam ~$25. Worlds run side by side, so the exam takes about an hour of waiting.
Nothing here needs a password: the jobs run inside your Modal account and nothing is served.
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
                      "flash-linear-attention==0.6.0")
         # flash-linear-attention: fast kernels for Qwen3.5's gated delta-net layers; without them transformers
         # falls back to plain PyTorch (slower), so the job still runs
         .add_local_file(HERE / "growing_brain.py", "/root/growing_brain.py")
         .add_local_file(HERE / "who_train.py", "/root/who_train.py")
         .add_local_file(HERE / "whoquiz.py", "/root/whoquiz.py"))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
app = modal.App("nichtec-brain4b")


@app.function(image=image, gpu="H100", volumes={"/root/.cache/huggingface": hf_cache}, timeout=4 * 60 * 60)
def run_world(stage: str, seed: int) -> str:
    import time
    import growing_brain as B
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    t0 = time.time()
    head = B.HFHead(BASE, log=log)
    w = B.make_world(seed)
    n0, n1, n2 = w["kept"][:3]
    gap = head.batching_check([B.read_prompt(n, w["C"][n]["desc"], q) for n in (n0, n1, n2)
                               for q in ("Does it have scales?", "Is it active at night?", "Does the gate open for it?")])
    log(f"batched vs one at a time, largest margin gap: {gap:.4f}")
    assert gap < 0.05, "left-padded batches change the answers: stop and look"
    R = B.run_world(head, seed, log=log, days=2 if stage == "check" else None)
    R.update(model=BASE, stage=stage, batching_gap=gap, minutes=round((time.time() - t0) / 60, 1), log=lines)
    return json.dumps(R)


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
