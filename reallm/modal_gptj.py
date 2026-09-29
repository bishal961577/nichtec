"""Test 4 on GPT-J-6B, the model the published MQuAKE results use, on one rented H100 (Modal) within about $5.

Once, on your computer:
  pip install modal
  modal setup                                      # logs in through the browser

Run (from the repository root; --detach keeps it running if the laptop sleeps or the terminal closes):
  modal run --detach reallm/modal_gptj.py
Copy the results again later (e.g. after a disconnect):
  modal run reallm/modal_gptj.py --fetch-only

What runs:
  1. a CPU container downloads GPT-J in half precision (12 GB) and the MQuAKE files into a Modal volume, so the
     GPU is not paid to wait for a download (skipped when already there);
  2. an H100 runs test 4 on MQuAKE-CF-3k-v2 (all 2,764 edits, all 3,000 multi-hop cases, 3 questions each),
     then on MQuAKE-T (real-world changes) if enough of the budget is left;
  3. the results are copied to results/reallm4_gptj/ and results/reallm4_gptj_t/.

Cost control (H100 $0.001097/s = $3.95/h, plus CPU and memory):
  - the GPU container is killed at TIMEOUT_MIN minutes whatever happens: at most about $4.2;
  - the test stops starting new multi-hop chunks near BUDGET_MIN minutes, and MQuAKE-T only starts when at least
    MIN_T_MIN minutes of the budget are left;
  - results are saved and the volume committed after every stage, so a stopped run keeps what it finished;
  - an automatic restart of the GPU container (e.g. preemption) would pay twice: a job that already started once
    refuses to run again;
  - the test aborts on its own after the first few minutes if the stored values do not work on this model.
"""
import os
import pathlib
import time

import modal

TIMEOUT_MIN = 58
BUDGET_MIN = 46
MIN_T_MIN = 8
MODEL_DIR = "/vol/models/gpt-j-6b-fp16"
DATA = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/{}"
RUNS = {"MQuAKE-CF-3k-v2.json": "reallm4_gptj", "MQuAKE-T.json": "reallm4_gptj_t"}
HERE = pathlib.Path(__file__).resolve().parent

app = modal.App("memtest-gptj")
vol = modal.Volume.from_name("memtest-gptj", create_if_missing=True)
# the exact versions the code was tested with; this torch build uses CUDA 13.0, which Modal's driver (580) supports
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.14.0", "transformers==5.17.0", "tokenizers==0.23.2", "huggingface_hub==1.33.0",
                      "safetensors==0.8.0", "numpy==2.4.6")
         .add_local_dir(HERE, remote_path="/root/reallm", ignore=["**/__pycache__/**", "*.pyc"]))


@app.function(image=image, volumes={"/vol": vol}, cpu=2, memory=4096, timeout=40 * 60)
def download():
    import urllib.request

    from huggingface_hub import snapshot_download
    if not os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        print("downloading EleutherAI/gpt-j-6b, revision float16 (12 GB) ...", flush=True)
        snapshot_download("EleutherAI/gpt-j-6b", revision="float16", local_dir=MODEL_DIR)
    data = "/vol/data/mquake"
    os.makedirs(data, exist_ok=True)
    for name in ("MQuAKE-CF-3k-v2.json", "MQuAKE-CF.json", "MQuAKE-T.json"):
        if not os.path.exists(os.path.join(data, name)):
            urllib.request.urlretrieve(DATA.format(name), os.path.join(data, name))
    vol.commit()
    print("model and data ready:", sorted(os.listdir(MODEL_DIR)), sorted(os.listdir(data)), flush=True)


@app.function(image=image, volumes={"/vol": vol}, gpu="H100", cpu=2, memory=32768, timeout=TIMEOUT_MIN * 60)
def run(job: str, budget_min: float):
    import sys

    t0 = time.time()
    vol.reload()
    marker = f"/vol/jobs/{job}"
    if os.path.exists(marker):
        print(f"job {job} already started once (automatic restart?): not running it again, to avoid paying twice")
        return None
    os.makedirs("/vol/jobs", exist_ok=True)
    open(marker, "w").write(time.ctime())
    vol.commit()
    sys.path.insert(0, "/root/reallm")
    import memtest4 as M4
    M4.ROOT = "/vol"               # data in /vol/data/mquake
    M4.SAVE_HOOK = vol.commit      # every save reaches the volume at once
    common = ["--model", MODEL_DIR, "--dtype", "bf16", "--gen-bs", "128", "--delta-chunk", "64", "--mh-q", "3",
              "--mh-methods", "joint_ls,base,rag"]
    for data, out in RUNS.items():
        left = budget_min - (time.time() - t0) / 60
        if data != "MQuAKE-CF-3k-v2.json" and left < MIN_T_MIN:
            print(f"skipping {data}: {left:.1f} min of the budget left", flush=True)
            break
        print(f"==== {data}: {left:.1f} min of the budget left", flush=True)
        try:
            R = M4.main(common + ["--data", data, "--out", f"/vol/results/{out}", "--budget-min", f"{left:.2f}"])
        except Exception:
            import traceback
            traceback.print_exc()
            print(f"{data} stopped by the error above; everything up to its last finished stage is saved", flush=True)
            break
        if R.get("aborted"):
            break
    vol.commit()
    return round(time.time() - t0, 1)


@app.function(image=image, volumes={"/vol": vol}, timeout=10 * 60)
def collect():
    vol.reload()
    files = {}
    for out in RUNS.values():
        d = f"/vol/results/{out}"
        if os.path.isdir(d):
            for name in os.listdir(d):
                files[f"{out}/{name}"] = open(os.path.join(d, name), "rb").read()
    return files


def fetch_results():
    files = collect.remote()
    root = HERE.parent / "results"
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        print("wrote", p)
    if not files:
        print("no results in the volume yet")


@app.local_entrypoint()
def main(budget: float = BUDGET_MIN, fetch_only: bool = False):
    if fetch_only:
        fetch_results()
        return
    if budget > TIMEOUT_MIN - 8:
        raise SystemExit(f"--budget must leave at least 8 minutes before the hard stop at {TIMEOUT_MIN} minutes")
    job = time.strftime("%Y%m%d-%H%M%S")
    print("1/3 model and data into the volume (CPU container, a few cents) ...")
    download.remote()
    print(f"2/3 H100 run {job}: budget {budget:.0f} min, hard stop at {TIMEOUT_MIN} min (at most about $4.2) ...")
    try:
        s = run.remote(job, budget)
        if s:
            print(f"H100 time {s / 60:.1f} min, about ${s * 0.001097 + s / 3600 * 0.35:.2f} with CPU and memory")
    finally:
        print("3/3 copying results ...")
        fetch_results()
