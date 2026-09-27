# On-device feasibility: phone hardware and measured LLM inference/fine-tuning (for a personal AI that updates its own weights on the phone)

Research date: 2026-09-27. Verification legend used on every bullet:
- **[P]** = number read directly from the primary document text (Hugging Face paper mirror of the arXiv text, or Qualcomm's own AI Hub model card on Hugging Face).
- **[S]** = number taken from a search-engine extract of the linked page. The page itself could not be opened because WebFetch to arxiv.org, apple.com, qualcomm.com, usenix.org, aclanthology.org, alphaxiv.org, huggingface.co (web) and others was blocked by the egress proxy. Treat [S] as secondary. The URL is the page the extract came from.
- **[D]** = my own arithmetic from cited numbers. These appear only under "Inferences", and the assumptions are stated.

---

## Q1. Phone hardware 2025–2026 (RAM, storage, NPU TOPS/precision, memory bandwidth, battery Wh, sustained power)

### Takeaway
2025–26 flagships have 12–16 GB LPDDR5X with about 77–85 GB/s peak bandwidth and 15.5–19.8 Wh batteries. Vendors advertise NPU peaks of about 80–100 "TOPS" at low precision (INT2/INT4/FP8), usually without official TOPS/W or precision footnotes. The binding limits for weight updates are memory bandwidth, RAM headroom and a sustained thermal envelope of roughly 3–7 W. Peak TOPS is not the binding limit. Mid-range phones (Pixel 10a, iPhone 17e) have 8 GB of RAM.

### Cited Findings
**Apple (A19 Pro / A19)**
- iPhone 17 Pro and Pro Max have 12 GB RAM. [S] — [MacRumors A19 vs A19 Pro](https://www.macrumors.com/2025/09/09/iphone-17-a19-chip/); [Wikipedia iPhone 17 Pro](https://en.wikipedia.org/wiki/IPhone_17_Pro)
- A19 Pro memory is 12 GB LPDDR5X at 9600 MT/s, with bandwidth "up to 76.8 GB/s". [S] — [Argmax iPhone 17 benchmarks](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks); [Apple A19 (Wikipedia)](https://en.wikipedia.org/wiki/Apple_A19)
- A19 Pro has a 6-core GPU with "Neural Accelerators" in every GPU core, which Apple says gives up to about 4× the peak GPU AI compute of A18 Pro. It also has a 16-core Neural Engine. The vapor chamber gives "up to 40 percent better sustained performance" than the previous generation. [S] — [Apple Newsroom, iPhone 17 Pro](https://www.apple.com/newsroom/2025/09/apple-unveils-iphone-17-pro-and-iphone-17-pro-max/)
- **Neural Engine TOPS for A19 Pro is not reliably sourced.** One aggregated snippet says "160 TOPS" ([MacRumors roundup / Wikipedia, via search](https://www.macrumors.com/roundup/iphone-17-pro/)). A low-quality blog says "about 45 TOPS" ([Inquisitive Universe](https://inquisitiveuniverse.com/2025/10/09/apple-a19-and-apple-a19-pro-full-specs-review/)). I found no Apple statement of a TOPS number. **Conflicting; do not use.**
- A microbenchmark of the A19 (5-core GPU, iPhone 17) GPU Neural Accelerators is consistent with 128 FP16 MACs per cycle per core at about 1460 MHz. The author says the "A19 iPhone GPU matches the M3 Max for FP16 matrix multiplication". [S] — [tzakharko, Investigating the GPU Neural Accelerators on A19/M5](https://tzakharko.github.io/apple-neural-accelerators-benchmark/)
- iPhone 17 Pro battery: 15.534 Wh (3988 mAh, physical-SIM model) and 16.558 Wh (4252 mAh, eSIM model). [S] — [Wikipedia iPhone 17 Pro](https://en.wikipedia.org/wiki/IPhone_17_Pro)
- iPhone 17 Pro Max battery: 18.748 Wh (4823 mAh at 3.887 V, physical SIM) and 19.772 Wh (5088 mAh at 3.886 V, eSIM). This comes from a leak of Chinese regulatory filings. [S] — [Macworld](https://www.macworld.com/article/2901629/entire-iphone-17-ranges-battery-specs-revealed-in-massive-leak.html); [MacObserver](https://www.macobserver.com/news/what-is-iphone-17-pro-max-battery-capacity/)
- Mid-range Apple: iPhone 17e has an A19 chip and 8 GB RAM. Snippet also lists a 4005 mAh battery and 256/512 GB storage. [S] — [MacRumors, iPhone 17e has 8GB of RAM](https://www.macrumors.com/2026/03/05/iphone-17e-has-8gb-of-ram/)

**Google (Tensor G5)**
- Pixel 10 Pro: Tensor G5, 16 GB RAM, 4870 mAh battery. Storage options are 128/256/512 GB/1 TB. The 128 GB model uses UFS 3.1 and the others use UFS 4.0. [S] — [GSMArena Pixel 10 Pro](https://www.gsmarena.com/google_pixel_10_pro_5g-13987.php)
- Tensor G5 has a 4th-generation TPU that Google says is "up to 60% more powerful". Gemini Nano on G5 is "2.6x faster and twice as efficient" as on G4. The Gemini Nano token window grew from 12K (G4) to 32K (G5). Google gives no absolute TOPS figure. [S] — [Google blog Tensor G5](https://blog.google/products-and-platforms/devices/pixel/tensor-g5-pixel-10/); [Android Authority](https://www.androidauthority.com/google-tensor-g5-3583168/)
- Mid-range Google: Pixel 10a has Tensor G4, 8 GB RAM, 128/256 GB storage and a 5100 mAh battery. [S] — [Google Store Pixel 10a specs](https://store.google.com/product/pixel_10a_specs?hl=en-US); [GSMArena Pixel 10a](https://www.gsmarena.com/google_pixel_10a_5g-14474.php)

**Qualcomm (Snapdragon 8 Elite Gen 5)**
- Qualcomm says the Hexagon NPU is 37% faster than the previous generation, supports INT2 and FP8, and runs local LLMs at "up to 220 tokens per second" (previous generation about 70 tok/s). The model and precision behind the 220 tok/s claim are not stated in the snippets. [S] — [Tribune India](https://www.tribuneindia.com/news/business/qualcomm-just-dropped-the-mic-with-the-snapdragon-8-elite-gen-5/); [startupnews, 220 tok/s](https://startupnews.fyi/2025/09/25/this-new-snapdragon-chipset-supports-220-tokens-per-second-heres-why-thats-a-big-deal/)
- NPU TOPS is unofficial and inconsistent. One site says "100 TOPS with native INT2" and "80 TOPS" for the fused AI block. The memory controller is LPDDR5X-5300 MHz, 4×16-bit (64-bit), with a maximum of 84.8 GB/s and up to 24 GB. This is a low-quality aggregator. [S] — [multicoreperformance.com](https://multicoreperformance.com/snapdragon-8-elite-gen-5-dedicated-ai-spec-sheet-80-tops-architecture-deep-thermal-truths/); official brief (not openable): [Qualcomm product brief PDF](https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/Snapdragon-8-Elite-Gen-5-product-brief.pdf)
- Galaxy S26 Ultra: Snapdragon 8 Elite Gen 5 for Galaxy, 12 or 16 GB RAM, 5000 mAh battery, up to 1 TB storage, 60 W wired charging. [S] — [GSMArena S26 Ultra](https://www.gsmarena.com/samsung_galaxy_s26_ultra_5g-14320.php)

**MediaTek (Dimensity 9500)**
- NPU 990 is described as the "first smartphone SoC to feature 100 TOPS" and uses a dual-NPU "Generative AI Engine 2.0". MediaTek claims 100% faster 3B-LLM output, 128K-token context and 56% lower power at peak performance. Hardware BitNet 1.58-bit support saves up to 33% power. MediaTek explicitly markets "**on-device LLM LoRA training**". [S] — [MediaTek press release](https://www.mediatek.com/press-room/mediatek-dimensity-9500-unleashes-best-in-class-performance-ai-experiences-and-power-efficiency-for-the-next-generation-of-mobile-devices); [Counterpoint](https://counterpointresearch.com/en/insights/mediatek-dimensity-9500-powering-powering-next-gen-smartphones)

**Sustained power / thermal**
- Qualcomm reportedly targets sustained performance for Snapdragon 8 Gen 3 at a "3W skin temperature budget". Flagship sustained TDP is "roughly 5–7 watts". Peak power has reached about 30 W (Exynos 2600 stress). The burst-to-sustained ratio is about 3:1 (2024) to 5:1 (2025). These are secondary/aggregator claims of moderate quality. [S] — [wccftech](https://wccftech.com/new-results-show-smartphone-thermal-capabilities-have-hit-a-wall-in-cooling-chipsets/); [product.ai physics guide](https://product.ai/truth-graph/smartphones/performance/)
- During mobile LLM inference (models up to 3B), device power reaches "as high as 8–10 W", with 3–10 GFLOPs and 1–3 GB of DRAM traffic per token. [S] — [MNN-AECS, arXiv 2506.19884](https://arxiv.org/pdf/2506.19884)

### Inferences
- [D] Battery Wh for phones listed only in mAh, using about 3.87–3.89 V nominal (the voltage in the iPhone filings): Pixel 10 Pro 4870 mAh ≈ 18.9 Wh; Galaxy S26 Ultra 5000 mAh ≈ 19.4 Wh; Pixel 10a 5100 mAh ≈ 19.8 Wh; iPhone 17e 4005 mAh ≈ 15.5 Wh. Working range: **about 15.5–19.8 Wh** per phone.
- [D] Bandwidth figures follow from the bus arithmetic: 9600 MT/s × 64 bit / 8 = 76.8 GB/s (A19 Pro). 10.6 Gbps × 64 bit / 8 = 84.8 GB/s (SD 8 Elite Gen 5). Plan with **about 75–85 GB/s peak** on flagships.
- Vendor "TOPS" are low-precision peaks (INT2/INT4/FP8) for inference. No vendor publishes backward-pass (training) throughput. MediaTek's LoRA-training claim is the only vendor statement found that on-device training is a supported NPU use case.
- The practical sustained budget for a background training job is plausibly about 3–7 W. Short bursts can reach 8–10 W before throttling (see the Q2 sustained-load results).

### Gaps
- No official Apple TOPS for the A19 Pro Neural Engine, and no official Qualcomm TOPS for 8 Elite Gen 5, were found. Google publishes no Tensor G5 TPU TOPS.
- No official memory bandwidth for Tensor G5 or Dimensity 9500 was found. My searches also did not surface the Dimensity 9500 LPDDR5X speed.
- Mid-range SoC NPU TOPS (Snapdragon 7 Gen 4, Dimensity 8450) and RAM of typical 2026 mid-range Android phones were not found in reliable sources.
- No manufacturer publishes the sustained SoC power budget in W for 2025–26 phones. The 3 W / 5–7 W numbers are secondary. Storage tiers for iPhone 17 Pro/Pro Max were not captured from a source.

---

## Q2. Measured on-device LLM inference (tokens/s, J/token, prefill vs decode) for 1B–8B models

### Takeaway
On 2025–26 flagship NPUs, 4-bit 1B/3B/8B models **decode at about 68 / 30 / 15 tok/s** and **prefill at about 4,000 / 1,800 / 1,200 tok/s** (Qualcomm AI Hub, Snapdragon 8 Elite Gen 5). Decode is memory-bandwidth bound. Measured energy is roughly **0.26–0.67 J per generated token for 7B-class models** (2024 hardware). GPU-based inference loses 40–60% of throughput within about a minute of sustained load. The Apple Neural Engine throttles least.

### Cited Findings
**Qualcomm AI Hub (vendor-measured, GENIE runtime, w4a16, 4096 context, Snapdragon 8 Elite Gen 5 for Galaxy)** [P]
- Llama-3.2-1B-Instruct: **68.18 tok/s** decode. TTFT is 0.0323 s (≤128-token prompt) to 1.033 s (4096-token prompt). The previous-generation 8 Elite gives 66.19 tok/s. — [qualcomm/Llama-v3.2-1B-Instruct model card](https://huggingface.co/qualcomm/Llama-v3.2-1B-Instruct)
- Llama-3.2-3B-Instruct: **30.10 tok/s** decode, TTFT 0.0722–2.311 s. The 8 Elite gives 28.03 tok/s. With w4 weights and w4 activations, the Gen 5 gives 16.24 tok/s. The GENIEX_QAIRT runtime gives 17.80 tok/s on w4a16. — [qualcomm/Llama-v3.2-3B-Instruct model card](https://huggingface.co/qualcomm/Llama-v3.2-3B-Instruct)
- Llama-3.1-8B-Instruct: **14.99 tok/s** decode, TTFT 0.1088–3.481 s. The 8 Elite gives 14.64 tok/s, and the GENIEX_QAIRT runtime on Gen 5 gives 10.85 tok/s. — [qualcomm/Llama-v3.1-8B-Instruct model card](https://huggingface.co/qualcomm/Llama-v3.1-8B-Instruct)
- A user on the Qualcomm forum reports 5.1 tok/s for Llama 3.1 8B on Snapdragon 8 Elite against Qualcomm's "published 13.1 tok/s". Real-world numbers can be far below vendor figures. [S] — [Qualcomm support forum](https://mysupport.qualcomm.com/supportforums/s/question/0D5dK00000GkGOtSAN/llama-31-8b-on-snapdragon-8-elite-getting-51-toks-vs-published-131-toks-what-configuration-is-needed-to-match-the-benchmark)

**Apple on-device foundation model (~3B)**
- 2024 model on iPhone 15 Pro: "time-to-first-token latency of about **0.6 millisecond per prompt token**, and a generation rate of **30 tokens per second**". [S] — [Apple ML Research, Introducing Apple's On-Device and Server Foundation Models](https://machinelearning.apple.com/research/introducing-apple-foundation-models)
- 2025 model: about 3B parameters with **2-bit quantization-aware training**. KV-cache sharing covers 37.5% of layers, which cuts KV memory by 37.5% and TTFT by about 37.5%. [P] — [Apple Intelligence Foundation Language Models Tech Report 2025 (HF mirror of arXiv 2507.13575)](https://huggingface.co/papers/2507.13575)
- 2025 model compression: 2 bits per weight via QAT, with the embedding table at 4 bits and the KV cache at 8 bits. LoRA adapters recover the quality lost to compression. [S] — [Apple ML Research, 2025 updates](https://machinelearning.apple.com/research/apple-foundation-models-2025-updates)
- Argmax says Apple deploys the ~3B Foundation Model on the **Neural Engine** "for top energy-efficiency". It measured only a 1.01–1.15× speedup of the iPhone 17 lineup over iPhone 16 Pro on its inference tasks. [S] — [Argmax, iPhone 17 benchmarks](https://www.argmaxinc.com/blog/iphone-17-on-device-inference-benchmarks)

**Google Gemini Nano (Pixel 10 Pro XL, Tensor G5)**
- Android Authority measured Gemini Nano 3 at about **9.6 tok/s** average and Nano 4 Fast at about **19.14 tok/s** average (AICore Developer Preview). [S] — [Android Authority, Gemini Nano 4 tested](https://www.androidauthority.com/gemini-nano-4-benchmarks-3655763/)

**Research systems (2024–2025 hardware)**
- PowerInfer-2 on OnePlus 12 (Snapdragon 8 Gen 3), TurboSparse-Mistral-7B: **0.257 J/token**, against QNN at 0.373 J/token and llama.cpp at **0.672 J/token**, over 100 prompts from lmsys-chat-1m. TurboSparse-Mixtral-47B reaches 11.68 tok/s with 19 GB of memory and 2.13 tok/s with 7 GB. [S] — [PowerInfer-2, arXiv 2406.06282](https://arxiv.org/pdf/2406.06282) (HF mirror exists: [huggingface.co/papers/2406.06282](https://huggingface.co/papers/2406.06282))
- mllm-NPU: Qwen1.5-1.8B prefill of **1,106 tok/s** on a Xiaomi 14 (1024-token prompt). Averages are 22.4× faster prefill and 30.7× lower energy than baselines. On a Redmi K60 Pro, 1024-token prefill energy is 35.6–59.5× lower than llama.cpp-CPU. [S] — [mllm-NPU, arXiv 2407.05858](https://huggingface.co/papers/2407.05858)
- Snapdragon 8 Gen 3 reaches 10–20 tok/s decode for Qwen1.5-4B under MLC-LLM. PhoneLM-1.5B prefill reaches 654 tok/s on the 8 Gen 3 NPU. [S] — [Understanding LLMs in Your Pockets, arXiv 2410.03613](https://arxiv.org/html/2410.03613v1)
- A cross-framework measurement study found that "NPUs excel at compute-bound prefilling, while CPUs outperform all other backends in memory-bound decoding". Framework-induced gaps reach 10× on NPUs. [S] — [Is Your NPU Ready for LLMs?, arXiv 2607.05475](https://arxiv.org/html/2607.05475v1)
- "The Battery Price of edge AI" (18 model configurations, 2 modern smartphones plus a server) finds on-device inference is on average **about 3× less energy-efficient than batched server inference**. Energy per token versus quantization bit-width is non-monotonic. 88–90% of on-device per-token environmental impact comes from device embodied carbon. [S] — [arXiv 2609.11940](https://arxiv.org/html/2609.11940); [replication package](https://zenodo.org/records/19230528)

**Sustained-load / thermal**
- iPhone 16 Pro with Qwen2.5-1.5B 4-bit settles at **23.7 tok/s** after a **41.5% drop** under sustained load. The Galaxy S24 Ultra is likewise constrained by thermal throttling, which the authors say "preclud[es] always-on deployment". [S] — [LLM Inference at the Edge… Under Sustained Load, arXiv 2603.23640](https://arxiv.org/html/2603.23640v2)
- iPhone 17 Pro with Gemma 4 E2B 4-bit over a 10-minute sustained run: CoreML/ANE 22 tok/s (67% of burst retained), MLX/GPU 18 tok/s (38% retained), LiteRT-LM/GPU 27 tok/s (48% retained). The GPU runtimes lost more than 50% within about 60 s. The author notes iOS exposes no per-subsystem watts. This is an independent blogger with a GitHub harness. [S] — [john-rocky, "GPU wins the sprint, the Neural Engine wins the marathon"](https://dev.to/john-rocky/iphone-on-device-llm-the-gpu-wins-the-sprint-the-neural-engine-wins-the-marathon-42lo); [apple-silicon-llm-bench](https://github.com/john-rocky/apple-silicon-llm-bench)

### Inferences
- [D] **Prefill rates** implied by Qualcomm TTFT at 4096 tokens: 1B ≈ 4096/1.033 ≈ **3,970 tok/s**; 3B ≈ 4096/2.311 ≈ **1,770 tok/s**; 8B ≈ 4096/3.481 ≈ **1,180 tok/s**. Apple's 0.6 ms/token implies about 1,670 tok/s for its 3B on an iPhone 15 Pro. These imply effective NPU throughput of about 10–19 TFLOP-equivalent/s during prefill: 2 × params × tok/s, e.g. 2 × 8e9 × 1180 ≈ 1.9e13.
- [D] **Decode is bandwidth-bound.** A 4-bit 8B model is about 4–4.5 GB of weights. At 15 tok/s that is about 60–68 GB read per second, roughly 70–80% of the 84.8 GB/s peak. So decode tok/s ≈ (bandwidth × efficiency) / model bytes.
- Training resembles prefill, since it processes whole sequences in parallel, not token-by-token decode. So the prefill numbers, not decode, are the right upper bound for on-device training compute, if the backward pass could run on the NPU (see Q3: in practice it mostly does not).
- The sustained-throttling results (40–60% drops within about a minute on GPU) mean a multi-minute training job should be budgeted at sustained, not burst, rates.

### Gaps
- No measured J/token for 2025–26 flagships (A19 Pro, 8 Elite Gen 5, Tensor G5, D9500) was retrieved. The "Battery Price" paper has them but its tables could not be opened.
- The exact tokens/s table in Meta's ExecuTorch quantized Llama 3.2 blog could not be retrieved. Only the relative gains were captured: 2.5× decode, 4.2× prefill, 56% smaller, 41% less memory on OnePlus 12. [S] — [Meta AI blog](https://ai.meta.com/blog/meta-llama-quantized-lightweight-models/)
- No Gemini Nano prefill rates and no Apple 2025-model tokens/s on A19 Pro from Apple itself were found.

---

## Q3. Measured on-device fine-tuning (MobileFineTuner, PocketLLM, FwdLLM, MeZO, MeBP, LoRA on phones, Apple adapters, federated precedent)

### Takeaway
Backprop LoRA fine-tuning of 0.5–4B models **fits in under 1 GB on an iPhone 15 Pro Max** (Apple MeBP, 2025), and down to about 136 MB for 0.5B with MeSP (2026). The measured wall-clock cost is still high. Public measurements show **about 1–1.75 h to LoRA-tune a 1B model on about 18k tokens** on an S25 or iPhone 16 (QVAC), which is only a few training tokens per second. Zeroth-order (MeZO-type) methods cut memory to inference level but need **10–100× more steps**. Their gradient estimates correlate only about 0.001 with true gradients. Production precedent for on-device training exists, but only for small models: Gboard federated learning runs while charging, idle and on unmetered Wi-Fi. Apple's own adapter training runs **off-device**.

### Cited Findings
**MeZO (Malladi et al., NeurIPS 2023): primary** [P] — [HF mirror of arXiv 2305.17333](https://huggingface.co/papers/2305.17333)
- MeZO "fine-tun[es] LMs with the same memory footprint as inference". Backprop with Adam needs "up to 12× the memory required for inference". For OPT-13B, full fine-tuning and PEFT need **12× and 6×** more memory than inference respectively.
- On one A100 80 GB: MeZO trains 30B, Adam fine-tuning only 2.7B, prefix fine-tuning 6.7B.
- Convergence cost: the OPT experiments ran "MeZO for **20K steps** and fine-tuning for 5 epochs, or **625 steps**". The RoBERTa-large experiments ran MeZO for **100K steps** against 1,000 fine-tuning steps. Per step, MeZO was **7.74× faster** and needed 8× fewer GPUs on 30B, for about half the total GPU-hours.
- Quality: on OPT-13B, MeZO was within 1% of fine-tuning on 7 of 11 tasks. RoBERTa-large got within 5% at k=512. MeZO "only works when using prompts".
- Checkpoint storage: a 66B MeZO trajectory is the seed plus 20,000 × 2 bytes, **under 0.1 MB**. LoRA (19M params) needs 38 MB.
- Theory: without assumptions, the ZO slowdown scales with parameter count d. Under a low effective-rank Hessian assumption, it scales with effective rank r instead.

**Apple MeBP (Song & Tang, EMNLP 2025 Industry): backprop on iPhone**
- Implemented in Swift on an **iPhone 15 Pro Max (8 GB RAM)**. LLMs from **0.5B to 4B** (Gemma 3, Qwen 2.5) are fine-tuned with **less than 1 GB of memory**. LoRA weights are "dozens of megabytes". [S] — [Apple ML Research page](https://machinelearning.apple.com/research/memory-efficient-backpropagation); [arXiv 2510.03425](https://arxiv.org/html/2510.03425); code: [apple/ml-mebp](https://github.com/apple/ml-mebp)
- ZO methods "require **10× to 100× more steps** than backpropagation". MeBP takes **43% to 94% more compute time per gradient step** than MeZO but "converges faster and better… in terms of both the number of optimization steps and total compute wall-clock time". [S] — [arXiv 2510.03425](https://arxiv.org/html/2510.03425)
- The per-step seconds table (Table 2) exists but its values could not be retrieved.

**Follow-ups to MeBP (2026)**
- MeSP (ACL 2026 Industry) cuts memory by 49% on average against MeBP for Qwen2.5 0.5–3B with identical gradients. For Qwen2.5-0.5B, peak memory falls **from 361 MB to 136 MB**, with 28% compute overhead. Setup: 4-bit (group size 64) base, bf16 LoRA rank 8, batch 1. It also measures **MeZO gradient estimates at cosine similarity ≈0.001 with true gradients**. [S] — [arXiv 2602.13069](https://arxiv.org/html/2602.13069v1); [ACL Anthology](https://aclanthology.org/2026.acl-industry.62/)
- LCSB (layer-cyclic selective backprop) gives up to **1.40× speedup** over MeBP with under 2% quality loss. In MeBP, "weight decompression alone accounts for 32–42% of backward time". A 4-bit 3B model that diverged under full backprop converged under LCSB. [S] — [arXiv 2602.13073](https://arxiv.org/abs/2602.13073)

**MobileFineTuner (Dec 2025): open-source C++ phone training stack**
- Supports full fine-tuning and LoRA with memory-efficient attention, activation checkpointing, gradient accumulation, ZeRO-style sharding and energy-aware scheduling. Evaluated on real phones with GPT-2, Gemma 3 and Qwen2.5. [S] — [arXiv 2512.08211](https://arxiv.org/abs/2512.08211); [GitHub](https://github.com/king21-noass/MobileFineTuner)
- Reported peak RSS for LoRA on WikiText-2, seq 128: **Qwen2.5-0.5B 2,887 MB** and Gemma3-270M 4,014 MB. At seq 256, Qwen2.5-0.5B reaches 3,437 MB. Another configuration reports **6,111 MB peak for Qwen2.5-0.5B, which crashed the Pixel 8 (8 GB)**. The difference is probably whether the optimizations were on; the snippet does not say. GPT-2 small (124M) full fine-tuning ran on a Pixel 8 Pro (seq 128, batch 8). [S] — [arXiv 2512.08211 HTML](https://arxiv.org/html/2512.08211)

**PocketLLM (2024): zeroth-order on a phone**
- Uses derivative-free (MeZO) fine-tuning on an **OPPO Reno 6**: RoBERTa-large in about **4 GB** and OPT-1.3B in about **6.5 GB**. No time per step was captured. [S] — [arXiv 2407.01031](https://arxiv.org/abs/2407.01031); [ACL Anthology PrivateNLP 2024](https://aclanthology.org/2024.privatenlp-1.10/)

**FwdLLM (USENIX ATC 2024): forward-gradient federated fine-tuning**
- Abandons backprop in favour of "perturbed inference" that can run on mobile NPUs. It combines this with PEFT and enables federated fine-tuning of **LLaMA-7B with 1.5 GB peak memory** on mobile devices. It claims up to three orders of magnitude faster convergence and a **14.6× memory reduction** against prior FedLLM methods. Evaluation used a Pixel 7 Pro with a GPTQ INT8/INT4 base and FP32 LoRA. [S] — [arXiv 2308.13894](https://arxiv.org/abs/2308.13894); [USENIX ATC'24](https://www.usenix.org/conference/atc24/presentation/xu-mengwei); [GitHub](https://github.com/UbiquitousLearning/FwdLLM)

**QVAC Fabric (Tether, 2025–26): LoRA on phone GPUs (Vulkan/Metal, llama.cpp-based)**
- BitNet b1.58 LoRA: a 125M model tunes in about **10 min on a Samsung S25**. A 1B BitNet on about 300 documents (**about 18k tokens**) takes **1 h 18 min on S25 (Adreno)** and **1 h 45 min on iPhone 16**. BitNet-1B (TQ1_0) uses up to 77.8% less VRAM than Gemma-3-1B F16. [S] — [Tether news](https://tether.io/news/tethers-qvac-launches-worlds-first-cross-platform-bitnet-lora-framework-to-enable-billion-parameter-ai-training-and-inference-on-consumer-gpus-and-smartphones/); [HF blog](https://huggingface.co/blog/qvac/fabric-llm-finetune-bitnet)
- A third-party paper cites QVAC as reporting "**1 hour 40 minutes per epoch** for a 1.7B model on an Adreno 830". The same paper *projects* (does not measure) about 1.3 h per 1000-example epoch for phi-1.5 LoRA (batch 1, 256 tokens, rank 4, 1.44 GiB) on a phone with about 80 GB/s DRAM. [S] — [AQLoRA, arXiv 2608.23816](https://arxiv.org/pdf/2608.23816)

**Other memory-reduction results relevant to phones**
- Llama-3.2-3B LoRA at 4K tokens: the FP32 baseline needs **40.41 GB**. INT4 base plus checkpointing, disk offload, Top-1000 softmax approximation and logits masking bring it to **1.59 GB** (over 25× less). At 16K tokens the optimized pipeline needs 6.95 GB. [S] — [arXiv 2606.19528](https://arxiv.org/html/2606.19528v1)
- "Parameter efficiency is not memory efficiency": IA3 (about 0.003% trainable parameters) still peaked at 32 GB in the authors' setup, because activation tensors scale with sequence length. Their LARS method cuts memory against LoRA by 33.54% on GPU and 51.95% on CPU. [S] — [arXiv 2604.22783](https://arxiv.org/abs/2604.22783)

**Apple adapters (what is public)**
- The Foundation Models framework supports custom LoRA adapters of **rank 32, about 160 MB each**. Training runs **off-device** on a Mac with Apple silicon and 32 GB+ memory, or on a Linux GPU box, via a Python/PyTorch toolkit. [S] — [Apple developer: Foundation Models adapter training](https://developer.apple.com/apple-intelligence/foundation-models-adapter); [blakecrosley.com](https://blakecrosley.com/blog/foundation-models-custom-adapters)
- No public Apple numbers for *on-device* adapter training of the Foundation Model were found. MeBP (above) is Apple research, not a shipped feature.

**Federated learning precedent (Gboard)**
- Gboard trains only when the phone is **charging, on an unmetered network and idle**. Early work required at least 2 GB of device memory, and a federated round closes after 100–500 client updates. [S] — [Hard et al. 2018, Federated Learning for Mobile Keyboard Prediction](https://ar5iv.arxiv.org/html/1811.03604); [Google Research blog 2017](https://research.google/blog/federated-learning-collaborative-machine-learning-without-centralized-training-data/); [Gboard Help](https://support.google.com/gboard/answer/12373137?hl=en)
- Google "trained and deployed **more than twenty** Gboard language models" with FL and differential privacy (DP-FTRL). Two of them also used secure aggregation. [S] — [Xu et al. 2023, arXiv 2305.18465](https://arxiv.org/abs/2305.18465); [Google Research blog 2024](https://blog.research.google/2024/02/advances-in-private-training-for.html)

### Inferences
- [D] **Measured training throughput is low.** QVAC's 1B BitNet run processed about 18k tokens in 78 min, about **3.8 training tokens/s**. That assumes a single pass; the epoch count was not captured, so with 3 epochs it would be about 11.5 tok/s. That is two to three orders of magnitude below the NPU prefill rates in Q2 (about 1,000–4,000 tok/s). The gap reflects GPU/CPU-only backward passes and 4-bit dequantization overhead (32–42% of backward time per LCSB), not a hard hardware limit.
- On memory, backprop LoRA with a 4-bit base, checkpointing and weight streaming now fits 0.5–4B models in well under the 8–16 GB of current phones. **Memory is no longer the blocker for 1–4B LoRA; time and energy are.** Zeroth-order methods remain relevant only where RAM is extremely tight (under 1 GB). Their 10–100× step penalty outweighs their roughly 1.4–1.9× faster steps (MeBP's 43–94% extra per-step time implies MeZO steps are about 1.4–1.9× faster).
- [D] Combining MeBP's numbers: net ZO wall-clock penalty ≈ (10 to 100 steps) ÷ (1.43 to 1.94 per-step advantage) ≈ **5× to 70× slower** than backprop for equal progress.
- Production precedent (Gboard) establishes the operating envelope: charging, idle, unmetered network, night time. It covers small RNN/transformer keyboard LMs, not billion-parameter LLMs.

### Gaps
- MeBP Table 2 (seconds per step and MB for each model) and MobileFineTuner's time and energy tables could not be read because arXiv is blocked. PocketLLM step times were also not found.
- No measured joules per training step or per training token on any 2025–26 phone was found.
- No evidence that any vendor NPU SDK (QNN, Core ML/ANE, LiteRT, NeuroPilot) exposes general LLM backward passes to third parties. MediaTek's LoRA-training marketing claim has no public benchmark.
- No study was found that measures fact acquisition (e.g. accuracy on newly taught facts per training step) on phones.

---

## Q4. QLoRA memory numbers and applicability to phones (plus other memory-efficient training methods)

### Takeaway
QLoRA's recipe is a frozen NF4 4-bit base, bf16 LoRA adapters, gradient checkpointing and paged optimizers. It puts a 7B model's training footprint at about **5 GB for the base plus about 0.1 GB of LoRA and activation state per sequence**. That is phone-sized (12–16 GB flagships), but the method as published depends on CUDA kernels and NVIDIA unified-memory paging. Phone ports (MeBP/MeSP, QVAC, MobileFineTuner) re-implement the same idea: a quantized frozen base plus small adapters plus recomputation.

### Cited Findings
- QLoRA "reduces the average memory requirements of finetuning a 65B parameter model from **>780GB** of GPU memory to **<48GB**" without degrading runtime or predictive performance against 16-bit full fine-tuning. [P] — [QLoRA, HF mirror of arXiv 2305.14314](https://huggingface.co/papers/2305.14314)
- For a 7B LLaMA at batch size 1 with LoRA at 0.2% of weights: LoRA parameters take **26 MB**; LoRA input gradients take **567 MB**, falling to about **18 MB per sequence** with gradient checkpointing; the **4-bit base consumes 5,048 MB**. The authors conclude that "most of the memory footprint for LLM finetuning comes from activation gradients and not from the learned LoRA parameters". [P] — [QLoRA](https://huggingface.co/papers/2305.14314)
- Double Quantization saves about 0.37 bits per parameter (about 3 GB on a 65B model). Paged Optimizers use NVIDIA unified memory to absorb checkpointing memory spikes. The deployed Guanaco 7B "requires just 5 GB of memory". [P] — [QLoRA](https://huggingface.co/papers/2305.14314)
- For phone equivalents, see Q3: MeBP (0.5–4B in under 1 GB, iPhone 15 Pro Max), MeSP (Qwen2.5-0.5B at 136 MB, 4-bit base, rank 8), the 2606.19528 pipeline (3B at 4K context in 1.59 GB), and MeZO/PocketLLM/FwdLLM (inference-level memory).

### Inferences
- [D] A 4-bit base costs about 0.5–0.6 bytes per parameter including scales: 1B ≈ 0.6 GB, 3B ≈ 1.7–2 GB, 8B ≈ 4.5–5 GB. Add about 0.1–0.6 GB of LoRA, gradient and optimizer state (per the QLoRA 7B breakdown) plus activations for the chosen sequence length. **An 8B QLoRA-style job at short sequence lengths plausibly fits in 12–16 GB flagships. A 3B job fits in 8 GB mid-range phones**, but the OS and other apps share this RAM. This inference comes from the QLoRA numbers, not from a phone measurement of 8B training.
- QLoRA's contribution for phones is that the adapter is the only thing written back. The persistent "weight update" artifact per training session is tens to hundreds of MB (26 MB QLoRA-7B example; 160 MB Apple rank-32 adapter), not the whole model.

### Gaps
- No published measurement of QLoRA/NF4 training of a 7–8B model on a phone was found. Phone results top out at about 4B (MeBP) for measured backprop.
- Paged-optimizer equivalents on mobile (flash offload of optimizer state) have one data point (2606.19528's "disk offloading") but no energy numbers.

---

## Q5. Energy per FLOP / per op on phone NPUs/GPUs, and daily energy budget for background training while charging

### Takeaway
No vendor publishes TOPS/W for 2025–26 phone NPUs, and no measured pJ/FLOP for training on phones was found. The best available anchors are indirect. (a) DRAM access costs about **20 pJ/bit** (Horowitz 45 nm, 640 pJ per 32-bit read; also cited for A17 Pro LPDDR5), versus sub-pJ to a few pJ for arithmetic. (b) Measured **0.26–0.67 J per generated token** for 7B-class decode on a 2024 flagship. (c) Device power of **about 3–10 W** under AI load. Background training is conventionally gated to charging, idle and unmetered conditions (Gboard; iOS BGProcessingTask with requiresExternalPower).

### Cited Findings
- Horowitz (ISSCC 2014, 45 nm): reading 32 bits from DRAM costs about **640 pJ**. A 32-bit integer multiply costs about 3.1 pJ. A 32-bit float multiply costs about 18.5× as much energy as an 8-bit integer multiply. [S] — [Horowitz, "Computing's energy problem" (ResearchGate)](https://www.researchgate.net/publication/271463146_11_Computing's_energy_problem_and_what_we_can_do_about_it); [energy table](https://www.researchgate.net/figure/Energy-consumption-of-multiply-accumulations-Horowitz-2014_tbl1_301848151)
- The A17 Pro's LPDDR5 memory-access energy is cited as **20 pJ/bit**. This is secondary, from a PIM architecture paper. [S] — [PIM-AI (arXiv 2411.17309) summary](https://pith.science/paper/2411.17309)
- LPDDR5X deep power-down with retention draws 2–4 mW per GB. [S] — [Lexar Enterprise LPDDR5X guide](https://lexarenterprise.com/lpddr5x-power-consumption-guide/)
- Measured J/token (inference, OnePlus 12): PowerInfer-2 0.257, QNN 0.373, llama.cpp 0.672 J/token (TurboSparse-Mistral-7B). [S] — [PowerInfer-2](https://arxiv.org/pdf/2406.06282)
- Mobile LLMs up to 3B involve "3–10 GFLOPS/token computation and up to 1–3 GB/token DRAM memory visits, raising device power to as high as 8–10 W". [S] — [MNN-AECS, arXiv 2506.19884](https://arxiv.org/pdf/2506.19884)
- Hexagon NPU marketing claims "about 45% performance improvement per watt" (older generation), with no absolute TOPS/W. The same article notes TOPS/W "does not account for the actual workload, the precision". [S] — [Medium explainer](https://medium.com/@craigadebanji46/npu-performance-on-smartphones-explained-38b6435e37e3); [Creative Strategies NPU wattage white paper](https://creativestrategies.com/research/white-paper-the-npu-wattage-advantage/)
- iOS `BGProcessingTask` can require external power (`requiresExternalPower`) and grants "several minutes" of run time for heavy work such as on-device Core ML training, only while the device is idle. Developer reports say the CPU-usage limit is lifted when external power is required. [S] — [Andy Ibanez, Modern Background Tasks](https://www.andyibanez.com/posts/modern-background-tasks-ios13/); [Apple Developer Forums thread](https://developer.apple.com/forums/thread/690666)
- Gboard training runs only while charging, idle and on Wi-Fi/unmetered networks. [S] — [Gboard Help](https://support.google.com/gboard/answer/12373137?hl=en); [Hard et al. 2018](https://ar5iv.arxiv.org/html/1811.03604)

### Inferences
All [D], with assumptions stated:
- **DRAM-bound energy floor for decode:** bytes read per token × 8 × 20 pJ/bit. For a 4-bit 1B model (~0.7 GB) that is about 0.11 J/token; for a 4-bit 8B model (~4.5 GB), about 0.72 J/token. This matches the measured 0.26–0.67 J/token for 7B-class models (PowerInfer-2 reads fewer bytes via sparsity). It supports treating weight movement as the dominant energy term.
- **Effective energy per FLOP at inference:** a dense 7B decode is about 14 GFLOP/token. At 0.26–0.67 J/token that is **about 20–50 pJ per FLOP effective** at batch 1 (memory-dominated). Batch/sequence-parallel work like prefill or training reuses each weight fetch across many tokens. So effective pJ/FLOP should be much lower, plausibly single-digit pJ or less, but **no measurement was found**.
- **Training FLOPs:** use the standard approximations of about 6N FLOPs/token for full backprop (2N forward + 4N backward) and about 4N for LoRA with a frozen base, plus about 2N if activation checkpointing recomputes the forward pass. These are standard approximations, not sourced here. For a 3B model: roughly 12–19 GFLOP per training token.
- **Worked scenario (per learned fact).** Assume one "fact" is taught with about 20 paraphrased sequences × 64 tokens × 3 epochs ≈ 4,000 training tokens. This is a modelling assumption, not a sourced number.
  - At the *measured* about 4 training tok/s (QVAC 1B, phone GPU): about 17 min per fact. At 5 W sustained that is about 5 kJ ≈ **1.4 Wh per fact**, about 7–9% of a 16–20 Wh battery.
  - At an *optimistic* 50 training tok/s (a 3B model at about 1 TFLOP/s effective sustained; not measured): about 80 s per fact, about 0.4 kJ ≈ **0.11 Wh per fact**.
  - With MeZO-style ZO: multiply by about 5–70× (Q3).
- **Worked scenario (per night).** Assume an 8 h charging window at a 3–5 W sustained training envelope. That is 24–40 Wh drawn from the wall, 1.2–2.6× the battery capacity. It costs about **US$0.004–0.006 per night** at an assumed US$0.15/kWh. It yields about 115k tokens (at 4 tok/s) to 1.4M tokens (at 50 tok/s), or roughly **30–360 facts per night** under the 4,000-tokens-per-fact assumption. Electricity cost is negligible. The binding constraints are thermal envelope, charging heat, time window and training throughput.

### Gaps
- No vendor TOPS/W or pJ/op for A19 Pro ANE, Hexagon (8 Elite Gen 5), Tensor G5 TPU or NPU 990 was found.
- No measured energy per training step or token on any phone was found (MobileFineTuner and EdgeFlowerTune report energy but their tables could not be opened: [EdgeFlowerTune arXiv 2605.08636](https://arxiv.org/abs/2605.08636)).
- No source quantifies how much sustained compute power a phone tolerates *while charging*, where battery charging heat competes with the SoC for the same skin-temperature budget. The Android WorkManager / JobScheduler idle-and-charging constraints and the actual iOS BGProcessingTask time limits were not retrieved from primary documentation.

---

## Q6. Flash write endurance (TBW) and the cost of rewriting weights / a 1 GB memory file daily

### Takeaway
Phone UFS storage uses TLC NAND rated at about 1,000–3,000 P/E cycles. Kioxia's formula TBW = capacity × P/E ÷ WAF gives tens to hundreds of TB for 128 GB–1 TB phones. **Writing 1 GB per day uses well under 1% of endurance per year**, and rewriting a full 4-bit 8B model (~4.5 GB) daily is still only a few percent per year. Adapter-only or seed-only (MeZO) updates are negligible.

### Cited Findings
- TBW = Capacity × P/E cycles ÷ Write Amplification Factor. Kioxia's worked example: a **128 GB UFS device, 3,000 cycles, WAF 4 → 96 TB written**. Managed flash (eMMC/UFS) is specified in P/E cycles rather than TBW. [S] — [Kioxia, Understanding TBW vs P/E cycles](https://americas.kioxia.com/content/dam/kioxia/en-us/business/memory/mlc-nand/asset/KIOXIA-TBW-vs-PE-Cycles-Tech-Brief.pdf); [eBOM summary of Kioxia guidance](https://www.ebom.com/kioxia-the-recommended-approach-for-endurance-calculations-with-e-mmc-and-ufs/); [Kioxia WAF/endurance brief](https://americas.kioxia.com/content/dam/kioxia/en-us/business/memory/mlc-nand/asset/KIOXIA_TBW_WAF_NAND_Endurance_Technical_Brief.pdf)
- Raw consumer TLC endurance is typically cited at **1,000–3,000 P/E cycles**. [S] — [Cyber Raiden, Micron 3D TLC overview](https://cyberraiden.com/2026/09/07/micron-3d-tlc-nand-architecture-generations-endurance-product-ecosystem-and-system-behavior/) (secondary blog)
- Samsung UFS 4.0: up to 4,200 MB/s sequential read and **2,800 MB/s sequential write**. Read efficiency is 6.0 MB/s per mA, 46% better than UFS 3.1. Capacities go up to 1 TB. [S] — [Samsung Semiconductor newsroom](https://news.samsungsemiconductor.com/global/samsung-develops-first-ufs-4-0-storage-solution-compliant-with-new-industry-standard/); [TechPowerUp](https://www.techpowerup.com/294504/samsung-announces-ufs-4-0-to-deliver-up-to-4-200-mb-s-read-2-800-mb-s-write-speeds-for-memory-cards)
- The Pixel 10 Pro 128 GB variant uses UFS 3.1; larger variants use UFS 4.0. [S] — [GSMArena](https://www.gsmarena.com/google_pixel_10_pro_5g-13987.php)
- Update artifact sizes: MeZO trajectory under 0.1 MB [P] ([MeZO](https://huggingface.co/papers/2305.17333)); QLoRA 7B LoRA params 26 MB [P] ([QLoRA](https://huggingface.co/papers/2305.14314)); Apple rank-32 adapter about 160 MB [S] ([Apple developer](https://developer.apple.com/apple-intelligence/foundation-models-adapter)).

### Inferences
All [D]:
- TBW for a **256 GB** phone: 256 GB × 1,000–3,000 ÷ 4 = **64–192 TB**. For **128 GB**: 32–96 TB. For **1 TB**: 250–750 TB.
- **1 GB/day** = 0.365 TB/yr, which is **0.2–0.6% per year** of a 256 GB device's TBW (0.4–1.1% for 128 GB).
- **Full 4-bit 8B model rewrite daily** (~4.5 GB/day) = 1.64 TB/yr, which is **0.9–2.6%/yr** at 256 GB (1.7–5.1%/yr at 128 GB). A 4-bit 3B rewrite (~1.8 GB/day) is about 1%/yr or less at 256 GB.
- **LoRA adapter rewrite** (26–160 MB/day) is at most 0.06 TB/yr, which is negligible.
- Write time: 1 GB at 2.8 GB/s peak takes about 0.4 s, so write energy is probably on the order of joules or less per GB. **This is not measured**; UFS write power was not found.
- Conclusion for the model: storage endurance and write energy are not binding constraints for daily weight or memory-file updates on 128 GB+ phones. RAM, thermal envelope and training throughput dominate.

### Gaps
- No phone OEM publishes P/E rating, WAF or TBW for its UFS parts. The 1,000–3,000 cycle range comes from a secondary blog, not a UFS datasheet.
- Actual WAF for small random writes, such as a frequently updated memory file, can far exceed 4. No measurement for phone workloads was found.
- UFS 4.0 write power (mA or W during sequential write) was not found, so the joules to write 1 GB is estimated, not sourced.
