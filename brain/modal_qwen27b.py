"""Serve Qwen3.5-27B-FP8 on one Modal GPU for the two ends check, step 4: the bigger describer
(reports/Two ends check 2026.md). The same model family as the phone's 4B, about 7 times larger.

On your computer, once (skip if you did it for the 4B):
    pip install modal
    modal setup                                   # logs in through the browser
Make the endpoint's password, once:
    python3 -c "import secrets; print(secrets.token_urlsafe(32))"        # prints a random password; keep it handy
    modal secret create nichtec-vllm-key VLLM_API_KEY=<that password>
Start it (from the repository root):
    modal deploy brain/modal_qwen27b.py
    -> prints a URL like  https://<your-workspace>--nichtec-qwen27b-serve.modal.run   Paste that URL in the chat.
    The session then asks for the password through a secure form (never paste it in the chat).
    The first request takes 10-15 minutes: the GPU starts and downloads the model (about 29 GB, kept in the same Modal
    volume as the 4B, so later starts are faster).
Stop it when the run is done (it also stops by itself after 5 idle minutes):
    modal app stop nichtec-qwen27b

Cost: Modal bills the GPU per second while a container runs (an H100 is roughly $4 per hour; check your dashboard).
Step 4 is about 1,500 short calls: expect 20-40 minutes of GPU in all, start-up included, so about $1.50-3.
Why an H100: Qwen's FP8 weights are block-quantised, which H100s run natively. A cheaper L40S (48 GB) also has room,
but vLLM would fall back to slower FP8 kernels there, untested for this model. To try it, set GPU = "L40S".

Access: every request needs the password, sent as "Authorization: Bearer <password>"; vLLM refuses the rest. This is
one header, so the cloud session can hold it as a single network secret. (The 4B script uses Modal's proxy auth
instead, which needs two headers.) Do not write the URL into the repository: it contains your workspace name.
"""
import os
import subprocess

import modal

MODEL = "Qwen/Qwen3.5-27B-FP8"
GPU = "H100"                     # 80 GB: FP8 weights about 29 GB, the rest for the cache
MAX_LEN = 16384                  # a describer prompt is about 1,500 tokens and its reply at most 700
VLLM_VERSION = "0.33.0"          # as for the 4B
CUDA_IMAGE = "nvidia/cuda:13.0.3-devel-ubuntu22.04"

image = (modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
         .entrypoint([])
         .uv_pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0"}))
hf_cache = modal.Volume.from_name("nichtec-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("nichtec-vllm-cache", create_if_missing=True)
app = modal.App("nichtec-qwen27b")


@app.function(image=image, gpu=GPU, volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
              secrets=[modal.Secret.from_name("nichtec-vllm-key")],
              scaledown_window=5 * 60, timeout=6 * 60 * 60, max_containers=1)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=30 * 60)
def serve():
    # OpenAI-compatible API at <URL>/v1, as for the 4B: --reasoning-parser qwen3 keeps any thinking apart from the
    # answer (step 4 runs with thinking off); --language-model-only skips the vision tower.
    subprocess.Popen(["vllm", "serve", MODEL, "--served-model-name", MODEL, "--host", "0.0.0.0", "--port", "8000",
                      "--max-model-len", str(MAX_LEN), "--reasoning-parser", "qwen3", "--language-model-only",
                      "--gpu-memory-utilization", "0.92", "--api-key", os.environ["VLLM_API_KEY"]])
