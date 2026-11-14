"""Serve Qwen3.5-4B on one Modal GPU for test 1's phone-sized arms (amendment 3). No H100 needed: a 4B model fits on
a 48 GB L40S with room for the 131,072-token context.

On your computer, once:
    pip install modal
    modal setup                                   # logs in through the browser
Start it (from the repository root):
    modal deploy brain/modal_qwen4b.py
    -> prints a URL like  https://<your-workspace>--nichtec-qwen4b-serve.modal.run   Paste that URL in the chat.
    The first request takes a few minutes: the GPU starts and downloads the model (~9 GB, kept in a Modal volume so
    later starts are faster).
Stop it when the test is done (it also stops by itself after 5 idle minutes):
    modal app stop nichtec-qwen4b

Cost: Modal bills the GPU per second while a container runs (an L40S is roughly $2 per hour; check your dashboard).
If your plan does not allow an L40S, set GPU below to "A100-40GB", "A10G" or "L4" (24 GB is enough for 4B, with a
shorter context: set MAX_LEN to 81920 on a 24 GB card).
The URL has no password: anyone who knows it could use your GPU while it is up, so stop the app after the test.
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-4B"
GPU = "L40S"
MAX_LEN = 131072                 # the 5,000-note notebook is ~60K Qwen tokens, plus up to 16K of thinking
VLLM_VERSION = "0.33.0"

image = (modal.Image.debian_slim(python_version="3.12")
         .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
app = modal.App("nichtec-qwen4b")


@app.function(image=image, gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache},
              scaledown_window=5 * 60, timeout=6 * 60 * 60, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def serve():
    # OpenAI-compatible API at <URL>/v1. --reasoning-parser qwen3 returns the thinking separately from the answer;
    # --language-model-only skips the vision tower (text-only test, more memory for the context).
    subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", "8000",
                      "--max-model-len", str(MAX_LEN), "--reasoning-parser", "qwen3", "--language-model-only"])
