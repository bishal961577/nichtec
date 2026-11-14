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
Access: the endpoint requires a Modal proxy auth token (requires_proxy_auth=True). A request without one is refused by
Modal's own proxy, before any GPU starts. (Until this change the URL had no password, and it was written into run logs
in a public repository; anyone who found it could have kept the GPU busy at the owner's cost.)
    1. Modal dashboard -> Settings -> Proxy Auth Tokens -> create one. It has a key (wk-...) and a secret (ws-...).
    2. Give them to the client as the headers Modal-Key and Modal-Secret: either as a network secret of the cloud
       environment (the session asks with a secure form; never paste them in a chat), or as the environment variables
       MODAL_KEY and MODAL_SECRET, which brain/order_gap.py's Endpoint sends.
    3. modal deploy brain/modal_qwen4b.py   (replaces the open endpoint).
"""
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-4B"
GPU = "L40S"
MAX_LEN = 131072                 # the 5,000-note notebook is ~60K Qwen tokens, plus up to 16K of thinking
VLLM_VERSION = "0.33.0"
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"   # torch 2.14.0 (vLLM 0.33.0's pin) is built for CUDA 13.0.3

# First deploy crashed at vLLM's warm-up: its default FlashInfer sampler compiles GPU code on first use and the slim
# image had no CUDA compiler (nvcc). Two fixes, each enough alone: NVIDIA's devel image (has nvcc), and
# VLLM_USE_FLASHINFER_SAMPLER=0 (vLLM's plain PyTorch top-k/top-p sampler; same sampling, nothing to compile).
image = (modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
         .entrypoint([])
         .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("nichtec-vllm-cache", create_if_missing=True)   # torch.compile results: faster restarts
app = modal.App("nichtec-qwen4b")


@app.function(image=image, gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              scaledown_window=5 * 60, timeout=6 * 60 * 60, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60, requires_proxy_auth=True)
def serve():
    # OpenAI-compatible API at <URL>/v1. --reasoning-parser qwen3 returns the thinking separately from the answer;
    # --language-model-only skips the vision tower (text-only test, more memory for the context).
    subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", "8000",
                      "--max-model-len", str(MAX_LEN), "--reasoning-parser", "qwen3", "--language-model-only"])
