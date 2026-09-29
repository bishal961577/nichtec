# Knowledge Editing for LLMs (2024 – 29 Sep 2026): Lifelong/Sequential Editing at Scale, Multi-hop Editing, Benchmark Reliability, and a Novelty Check

Verification legend used throughout:
- **[PRIMARY]**: number read directly from the paper text/table (via the Hugging Face papers mirror of arXiv) or the official GitHub/HF dataset card.
- **[SNIPPET]**: taken from a web-search engine's extract of the primary PDF/HTML. The primary page itself could not be opened. Treat as unconfirmed.
- Access limits in this environment: the egress proxy blocked arxiv.org, ar5iv, alphaxiv, openreview.net, proceedings.iclr.cc/iclr.cc, ijcai.org, ojs.aaai.org, aclanthology.org, semanticscholar, paperswithcode, zenodo and several review sites. Only the HF papers mirror (with partial coverage), github.com / raw.githubusercontent.com, HF datasets and search snippets were reachable. The HF mirror had no copy of MEMOIR (2506.07899), RLEdit (2502.05759), MeG (2512.14395), CHECK (2508.00914), KEDKG (2412.13782), HYPE (2505.18343), MCircKE (2604.05876), GMeLLo (2408.15903), PokeMQA, DeepEdit, RAE or GLAME. The web-search budget ran out before every snippet-level claim could be re-checked.

## Lifelong / sequential editing at scale (methods, numbers, settings, weaknesses, largest scale)

### Takeaway
By Sep 2026, two figures compete for "largest reported lifelong scale", and they use very different protocols.
- **Largest raw count:** UltraEdit reports **2,000,000 edits** (100 edits per turn, on its own UltraEditBench, scored with token-level exact match). Its figures there are Eff ≈ 81–83% and Spe ≈ 76–78%. It also reports 500K edits on WikiBigEdit.
- **Strict one-edit-per-step:** the strongest result is AlphaEdit + Norm-Anchor Scaling (NAS, arXiv 2602.02543). It covers the full CounterFact stream (20,877 edits) and the full ZsRE stream (19,086 edits) with efficacy of about 92–99%. However, CounterFact specificity is only about 60%, against 84–90% for the unedited model.
- **Parameter-preserving memories** report retention at 1K–15K edits: WISE and MEMOIR.
- **Under realistic generative evaluation the picture collapses.** WILD ("Mirage") finds about 10% success after 1,000 sequential edits, and EtCon finds AlphaEdit and MEMIT at or near 0% in "real-world" lifelong evaluation.

"Good retention at scale" therefore holds only under teacher-forced / token-EM metrics.

### Cited Findings

**Comparison table: lifelong / sequential editors.**
Column meanings:
- Rel/Eff = reliability / efficacy.
- Gen = generalisation (paraphrases).
- Loc/Spe = locality / specificity.
- CF = CounterFact.
- "1/step" = one edit committed per step. "b100" = batches of 100 edits applied sequentially.

| Method (venue) | Type | Model | #Edits & protocol | Benchmark | Rel/Eff | Gen | Loc/Spe | Status / source |
|---|---|---|---|---|---|---|---|---|
| GRACE (NeurIPS 2023) | frozen LM + discrete key-value codebook at one layer (deferral radius) | LLaMA-2-7B | T=1,000, 1/step | ZsRE | .93 | .08 | 1.00 | [PRIMARY, from WISE Table 2] [WISE](https://huggingface.co/papers/2405.14768) |
| GRACE | same | Mistral-7B | T=1,000 | ZsRE | 1.00 | .02 | 1.00 | [PRIMARY, WISE Table 2] [WISE](https://huggingface.co/papers/2405.14768) |
| WISE (NeurIPS 2024) | side-memory FFN copy + activation router + random-mask sharding + Ties-merge | LLaMA-2-7B | T=1,000, 1/step | ZsRE | .77 | .72 | 1.00 | [PRIMARY] [WISE](https://huggingface.co/papers/2405.14768) |
| WISE | same | Mistral-7B | T=1,000 | ZsRE | .70 | .67 | 1.00 | [PRIMARY] [WISE](https://huggingface.co/papers/2405.14768) |
| WISE (re-run by UltraEdit) | same | LLaMA-3-8B-Instruct | 20K, 1/step | ZsRE | 40.94 | 40.27 | 37.40 | [PRIMARY, UltraEdit Table 2] [UltraEdit](https://huggingface.co/papers/2505.14679) |
| MEMIT (sequential, per WISE) | locate-then-edit | LLaMA-2-7B | T=1,000 | ZsRE | .04 | .04 | .02 | [PRIMARY] [WISE](https://huggingface.co/papers/2405.14768) |
| AlphaEdit (ICLR 2025) | L&E + null-space projection | LLaMA3-8B | sequential, b100. Total 2,000 is my recollection and was NOT verified in the text I read | CF | 98.90 | 94.22 | 67.88 | [PRIMARY] [AlphaEdit](https://huggingface.co/papers/2410.02355) |
| AlphaEdit | same | LLaMA3-8B | same | ZsRE | 94.47 | 91.13 | 32.55 | [PRIMARY] (pre-edit ZsRE Spe 31.89) |
| AlphaEdit | same | GPT-J | same | CF / ZsRE | 99.75 / 99.79 | 96.38 / 96.00 | 75.48 / 28.29 | [PRIMARY] [AlphaEdit](https://huggingface.co/papers/2410.02355) |
| AlphaEdit (re-run by NAS) | same | Llama3-8B | 20,877 CF / 19,086 ZsRE, 1/step | CF / ZsRE | 66.27 / 22.30 | 58.75 / 20.34 | 49.98 / 1.57 | [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543) |
| AlphaEdit (re-run by UltraEdit) | same | Mistral-7B-v0.3 | 20K, b100 | ZsRE | 0.00 | 0.00 | 0.00 | [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679) |
| RLEdit (ICML 2025) | RL-trained hypernetwork | LLaMA-3-8B-Instruct | 20K, b100 (UltraEdit re-run) | ZsRE | 91.34 | 89.68 | 41.94 | [PRIMARY via UltraEdit]. Own paper: Llama-3-8B, Gemma-2-9B, Mistral-7B; "59.24% improvement … 2.11% of the time" [SNIPPET] [PMLR](https://proceedings.mlr.press/v267/li25an.html) |
| RLEdit (re-run by NAS) | same | Llama3-8B | 20,877 / 19,086, 1/step | CF / ZsRE | 65.38 / 81.59 | 49.18 / 78.77 | 47.86 / 27.03 | [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543) |
| ENCORE (Gupta et al. 2025, "Lifelong Knowledge Editing requires Better Regularization") | L&E + Most-Probable Early Stopping + Frobenius-norm constraint | (Llama-3-8B, GPT2-XL, Llama-2 family; per-model numbers not read) | 10,000 sequential edits | CF / ZsRE | not read | – | – | [PRIMARY abstract] "scaling … to 10,000 edits while reducing editing time by 42-61%" [ENCORE](https://huggingface.co/papers/2502.01636) |
| UltraEdit (TMLR-reviewed; OpenReview GoJLp3BIRV) | training-free closed-form ridge per turn + lifelong normalisation | LLaMA-3-8B-Instruct | 20K, b100 | ZsRE | 90.07 | 87.36 | 49.51 | [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679) |
| UltraEdit* | same | LLaMA-3-8B-Instruct | **100K**, b100 | ZsRE | 87.80 | 85.48 | 46.74 | [PRIMARY] |
| UltraEdit* | same | LLaMA-3-8B-Instruct | **500K**, b100 | WikiBigEdit | 68.99 | 63.59 | 52.28 (Personas 55.04) | [PRIMARY] |
| UltraEdit* | same | GPT-J / Mistral-7B / LLaMA-3-8B-Inst / Qwen2.5-7B-Inst | **2M**, b100 | UltraEditBench | 81.65 / 81.70 / 83.45 / 80.70 | 76.80 / 77.25 / 79.11 / 75.78 | 76.44 / 77.09 / 78.05 / 76.01 | [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679) |
| UltraEdit (re-run by NAS) | same | Llama3-8B | 20,877 / 19,086, 1/step | CF / ZsRE | 60.06 / 77.61 | 52.90 / 75.15 | 43.88 / 50.26 | [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543) |
| MEMOIR (NeurIPS 2025) | frozen LM + residual memory, sample-dependent sparse masks (TopHash), activation-pattern retrieval | LLaMA-3(-8B) | 1,000 sequential | ZsRE | avg 0.93 (+14 pts over AlphaEdit) | – | – | [SNIPPET] [arXiv](https://arxiv.org/abs/2506.07899), [NeurIPS](https://neurips.cc/virtual/2025/poster/115624) |
| MEMOIR | same | LLaMA-3 | **15,000** sequential | ZsRE | avg 0.88 (MEMIT/AlphaEdit ~0 after a few thousand) | – | – | [SNIPPET] |
| MeG (ICLR 2026) | one dynamic neuron whose weights a diffusion model generates per query, + "familiarity network" | Phi-2 / GPT-J | 1,024 → 10,000 (massive/batch; trained on the edit set) | ZsRE, CF | ZsRE 10K Score 82.80 (Phi-2) vs MALMEN 45.64; Locality 83.99% (GPT-J) | – | – | [SNIPPET] [arXiv](https://arxiv.org/abs/2512.14395). The training pipeline (text encoder → familiarity net → neuron weights → diffusion model) comes from the README [PRIMARY] [GitHub](https://github.com/RodeWayne/MeG-for-Knowledge-Editing) |
| RLSEdit (arXiv 2601.15686, Jan 2026) | online **cumulative ridge LS** via recursive least squares (Woodbury) + anchor term | Llama-3-8B | **10K**, b100, 3 seeds | CF | 89.94 | 72.84 | 60.56 | [PRIMARY] [RLSEdit](https://huggingface.co/papers/2601.15686) |
| RLSEdit | same | Qwen2.5-7B | 10K, b100 | CF | 94.45 | 68.55 | 73.37 (AlphaEdit 94.10/70.29/75.29) | [PRIMARY] |
| NAS = AlphaEdit + Norm-Anchor Scaling (arXiv 2602.02543) | L&E + value-vector norm anchoring | Llama3-8B | **20,877 (full CF) / 19,086 (full ZsRE), 1/step** | CF / ZsRE | 97.95 / 93.17 | 82.71 / 88.30 | 60.44 / 32.15 (pre-edit 89.79 / 32.09) | [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543) |
| NAS | same | GPT-J | same | CF / ZsRE | 98.73 / 92.08 | 90.68 / 85.04 | 60.75 / 22.24 | [PRIMARY] |
| NAS | same | Qwen2.5-7B | same | CF / ZsRE | 90.44 / 96.84 | 55.05 / 88.10 | 73.21 / 42.52 | [PRIMARY] |
| EtCon (arXiv 2512.04753) | Targeted Proximal SFT on FFNs (CoT-augmented labels) + GRPO consolidation | Qwen-2.5-7B-Instruct / Llama-3-8B-Instruct | sequential. Main-table count not read; the appendix goes "up to 3,000 sequential edits" | ZsRE, CF, QAEdit; **GPT-4.1 judge**, autoregressive | Qwen ZsRE 69.4, QAEdit 75.1; Llama ZsRE 73.5 | Qwen ZsRE 60.8, QAEdit 63.0 | 24.2–33.6 | [PRIMARY] [EtCon](https://huggingface.co/papers/2512.04753) |

**NAS / "Norm Anchors Make Model Edits Last" (arXiv 2602.02543)**
- The HF mirror's v1 title is "Toward Ultra-Long-Horizon Sequential Model Editing" (Liu, Zhu, Miao, Fujisawa). Web search lists the arXiv PDF under the title "Norm Anchors Make Model Edits Last", so it is the same paper retitled. [HF](https://huggingface.co/papers/2602.02543); [arXiv PDF listing](https://arxiv.org/pdf/2602.02543)
- **Diagnosis:** L&E updates cause exponential growth of the edited MLP weight norm (log R_n is linear in n). This correlates with collapse, with a reported Spearman ρ.
- **Method:** NAS rescales each target value vector to the anchor a = E‖v_new‖. The anchor is estimated from N = 1,000 pilot edits on the clean model. The authors prove the weight norm stays bounded. [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543)
- **Collapse points (CP@60):**
  - MEMIT and PRUNE: 400 → 1.5k edits.
  - RECT: 400 → 1.4k.
  - AlphaEdit: 4k → >21k.
  - Average editing success rises from 51.9 to 89.3 (+37.4 pp, +72.2% relative).
  - [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543)
- **General capability:** prior L&E baselines' GLUE-style scores "drop to near-zero within the first few thousand edits (typically ≤4k)". NAS keeps non-trivial GLUE through 20,877 edits, although MRPC declines. [PRIMARY]
- **Internal inconsistency:** the ZsRE stream size is given as 19,086 in the text but 19,082 in the Fig. 1 caption. [PRIMARY]
- **Stated limitation:** a fixed anchor may be suboptimal for non-stationary streams or edits of varying difficulty. [PRIMARY]

**UltraEdit (arXiv 2505.14679)** [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679)
- **Update rule:** per module, Δθ = (HᵀH + I)⁻¹HᵀV. H holds hidden states at the answer-label position. V holds normalised, scaled gradients. Running mean and variance "lifelong normalisation" acts as online whitening. The ridge solve covers **only the current turn's batch**.
- **Speed claims:** more than 7× faster and 4× less VRAM than the previous SOTA. It is the "only method" that can edit a 7B model on a 24 GB GPU.
- **Benchmark:** UltraEditBench builds more than 2M pairs from Wikidata5M triples, with questions generated by GPT-4o-mini.
- **Protocol mismatch:** baselines were run only to 20K edits (ZsRE / FEVER / UltraEditBench) and 17K (WikiBigEdit), while UltraEdit* was run to 100K / 500K / 2M.
- **Multi-hop:** WikiBigEdit "Reasoning" (multi-hop) stays at 35.64 for UltraEdit on LLaMA-3-8B-Instruct at 17K edits.

**WikiBigEdit (ICML 2025; arXiv 2503.05683)** [PRIMARY] [WikiBigEdit](https://huggingface.co/papers/2503.05683)
- The benchmark has more than 500K QA pairs (about 7M tokens) from real Wikidata diffs over 8 intervals (Feb–Jul 2024). It is auto-extending. It tests rephrase, persona, locality and multi-hop.
- Conclusion: "limited transfer of knowledge editing techniques to practically scalable lifelong updates … standard approaches such as retrieval augmentation or continual finetuning with model merging can perform better."
- It notes the usual benchmark sizes: CounterFact 20K, ZsRE 1K, SelfCheckGPT 600.

**RLSEdit (arXiv 2601.15686)** [PRIMARY] [RLSEdit](https://huggingface.co/papers/2601.15686)
- Eq. 15: W*_t = argmin_W Σ_{i≤t} ‖K_i W − V_i‖² + λ²‖W − W0‖² + μ²‖K0 W − V0‖².
- Letting μ→∞ recovers the null-space (hard-preservation) regime.
- It gives deviation bounds (Theorem 4.1) and asymptotic consistency to a ridge population minimiser (Prop. 4.2).
- It claims preservation of GLUE and held-out reasoning/code benchmarks, and better retention of early edits.

**EtCon (arXiv 2512.04753)** [PRIMARY] [EtCon](https://huggingface.co/papers/2512.04753)
- Under "real-world lifelong editing evaluation", MEMIT "collapses entirely" on Qwen-2.5-7B-Instruct.
- AlphaEdit gets 15.9% Reliability on ZsRE but 0.0% on all metrics for CounterFact and QAEdit (Qwen-2.5). On Llama-3 it reaches 61.0% on CF, but locality falls to 16.1%.
- FT-M gets 5.6% and WISE 4.5% ZsRE Reliability on Qwen-2.5.
- Consolidation "cannot repair damage incurred during the editing stage".

**2026 null-space follow-ups and reproductions**
- **AlphaEdit reproducibility study** (arXiv 2606.26783, Jul 2026) [SNIPPET] [arXiv](https://arxiv.org/abs/2606.26783):
  - It reproduces the original metrics but finds a discrepancy in the reported fluency and consistency.
  - The advantage does not generalise uniformly to Qwen2.5, Gemma-2 and Phi-3.
  - Performance degrades at longer horizons, so the null-space protection is "bounded rather than unconditional".
  - Large-scale editing harms BoolQ, HellaSwag and XSTest (safety).
- **BetaEdit** (arXiv 2605.09285, IJCAI-26): approximate null spaces leak knowledge and cause long-horizon failure; it re-adds a history-aware penalty term. [SNIPPET] [arXiv](https://arxiv.org/abs/2605.09285); [GitHub](https://github.com/lbq8942/BetaEdit)
- **LOKI** (arXiv 2606.19679, Jun 2026): HSIC-based layer selection plus null-space-projected gradient updates without access to previous knowledge; "up to 14% improvement in average accuracy". [SNIPPET] [arXiv](https://arxiv.org/abs/2606.19679)
- **Titles found only (not read):**
  - HiEdit (arXiv 2604.11214, hierarchical RL lifelong editing).
  - DOW-KE (arXiv 2608.16932).
  - REPAIR (arXiv 2510.01879).
  - NMKE (arXiv 2510.22139).
  - LyapLock (arXiv 2505.15702).
  - QueueEDIT (arXiv 2506.17864).

**MEMOIR repo:** it demos 100 sequential edits on ZsRE and SelfCheckGPT with Llama-3-8B, is built on EasyEdit, and uses a dataset-specific `irr_threshold`. [PRIMARY] [GitHub](https://github.com/qym7/MEMOIR)

### Inferences
- **Largest scale with "good" retention:**
  - **By raw count:** UltraEdit's 2M edits (100 per turn, own benchmark, token EM).
  - **For strict one-edit-per-step on standard benchmarks:** AlphaEdit + NAS at about 20K edits.
  - **For frozen-backbone memories:** MEMOIR at about 15K [SNIPPET].
  - None of these counts has been validated with generative (WILD-style) evaluation at that scale. Where independent re-runs exist (NAS re-running UltraEdit, RLEdit and AlphaEdit; EtCon re-running AlphaEdit and WISE), the headline numbers fall sharply.
- **Protocol drives results as much as method.** AlphaEdit ranges from 98.9 Eff (Llama3, b100 × ~2K) to 66.3 Eff (1/step × 20K), and to 0.0 on Mistral at 20K. Any cross-paper table must fix model, batch size, total edits and metric.
- **Specificity/locality is the weakest dimension everywhere.** CF specificity of about 60 after 20K edits (NAS) is roughly 30 pp below the unedited model. ZsRE "Spe" is often near the pre-edit value only because the base model is weak on those questions (pre-edit Spe of about 32).
- **The dominant diagnosis of sequential collapse in 2025–26 is norm growth and over-optimisation of the edited weights** (ENCORE, NAS, BetaEdit, EtCon all say so). Methods that keep the backbone frozen (GRACE, WISE, MEMOIR, MeG) avoid norm growth but struggle to generalise to paraphrases (GRACE Gen .08) or rely on routers/thresholds.

### Gaps
- **Unverified or not read:**
  - MEMOIR, MeG and RLEdit primary tables could not be opened, so their numbers are [SNIPPET].
  - The ENCORE per-model table and EtCon Table 2 in full (including the number of edits) were not read.
  - NAS appendix C.4 (GRACE / MELO / WISE / MEMOIR / ENCORE / LyapLock at the 20K scale) was not read.
  - AlphaEdit's total edit count (2,000 per my recollection) is not verified from the text read.
- **No independent re-evaluation** of UltraEdit's 2M-edit claim was found, and no WILD/LLM-judge evaluation at more than 20K edits for any method.
- **HiEdit, LOKI, BetaEdit and DOW-KE** (2026) scale numbers were not obtained.

## Multi-hop editing on MQuAKE (methods, numbers, retrieval vs weight edits, best at 3,000 edits)

### Takeaway
- **Weight-editing methods remain very poor at multi-hop under MQuAKE's "k instances at once" protocol.** With GPT-J, MEMIT gets 5.4% multi-hop accuracy at k = 3,000 on MQuAKE-CF. Newer weight editors (AcE, MCircKE, HYPE, CaKE) report relative gains but mostly in small-batch settings.
- **The best 3,000-edit numbers come from retrieval / in-context / KG methods:**

  | Method | Model | Multi-hop accuracy at k = 3,000 | Status |
  |---|---|---|---|
  | GMeLLo | GPT-J | 49.0% | [SNIPPET] |
  | GMeLLo | Vicuna-7B | 41.9% | [SNIPPET] |
  | CHECK | GPT-J | 42.27% on "MQuAKE-CF-3k"; edit batch unverified | [SNIPPET] |
  | MeLLo | GPT-3.5 | 30.9% | [PRIMARY] |

- **G-Walk reports 57.37% at 1,000 edits** on the corrected data [SNIPPET].
- Dataset-version drift (CF-3k vs CF-3k-v2 vs Remastered) makes these numbers non-comparable across papers.

### Cited Findings

**MQuAKE protocol and baseline numbers** (EMNLP 2023; arXiv 2305.14795 v3) [PRIMARY] [MQuAKE](https://huggingface.co/papers/2305.14795)
- MQuAKE-CF has more than 9K questions of 2, 3 or 4 hops. The 3K subset was chosen "to ensure there is no knowledge conflict". A v3 footnote adds: "We have fixed this subset in 2024/9 to resolve an issue of knowledge conflict", which gives MQuAKE-CF-3k-v2.
- MQuAKE-T has 1,868 instances, each with one real Wikidata update (2021-04 → 2023-04).
- **Protocol:** edit batches of k ∈ {1, 100, 1000, 3000} instances on CF and {1, 100, 500, 1868} on T, with all edits in a group injected at once. Each CF instance carries up to 4 edited facts.
- **Metric:** multi-hop accuracy counts a hit if *any* of the 3 paraphrased questions is answered correctly.
- **Single-instance results, GPT-J:** MEMIT edit-wise 96.2%, multi-hop 7.0% (11.8% with CoT). ROME multi-hop 7.4% (19.9% with CoT). The unedited base model scores 40.5% multi-hop.

**MQuAKE v3 Table 5: multi-hop accuracy by edit batch k** [PRIMARY]. These numbers may reflect the post-fix subset; I did not compare them with the EMNLP camera-ready.

| Base model | Method | CF k=1 | CF k=100 | CF k=1000 | CF k=3000 | T k=1 | T k=100 | T k=500 | T k=1868 |
|---|---|---|---|---|---|---|---|---|---|
| GPT-J | MEMIT | 11.8 | 10.5 | 7.9 | 5.4 | 4.8 | 1.0 | 0.2 | 0.0 |
| GPT-J | MEND | 11.7 | 9.6 | 7.6 | 6.1 | 38.2 | 17.4 | 12.7 | 4.6 |
| GPT-J | MeLLo | 38.9 | 20.3 | 15.6 | 14.2 | 85.9 | 45.7 | 33.8 | 30.7 |
| Vicuna-7B | MeLLo | 38.7 | 13.1 | 10.3 | 10.0 | 84.4 | 56.3 | 52.6 | 51.3 |
| GPT-3.5 | MeLLo | 52.1 | 35.3 | 31.8 | 30.9 | 91.1 | 87.4 | 86.2 | 85.5 |

GPT-3.5 means gpt-3.5-turbo-instruct on CF and text-davinci-003 on T.

**Method-by-method: type, settings and numbers**

| Method (venue) | Retrieval / in-context vs weight edit | Models | Dataset (version) and k | Reported multi-hop accuracy | Status / source |
|---|---|---|---|---|---|
| MeLLo (EMNLP 2023) | retrieval + decomposition + self-check (frozen LM, Contriever memory) | GPT-J, Vicuna-7B, GPT-3.5 | CF / T, k up to 3000 / 1868 | see table above | [PRIMARY] [MQuAKE](https://huggingface.co/papers/2305.14795) |
| PokeMQA (ACL 2024) | retrieval / in-context. Separates decomposition from a trained scope detector (DistilBERT pre-detector + conflict disambiguator); ELQ entity linking to Wikidata knowledge prompts | LLaMa-2-7B, Vicuna-7B, GPT-3.5-turbo-instruct | CF-3k, T; edited-num 1 and 3000 | LLaMa-2-7B, k=1, CF-3K: 44.13% multi-hop, 30.6% hop-wise (MeLLo 33.57% / 9.9%) | [SNIPPET]; repo [PRIMARY] [GitHub](https://github.com/hengrui-gu/pokemqa) |
| DeepEdit (2024) | decoding-time constraints + depth-first search (black-box, no weight edit) | – | introduced MQuAKE-2002 and MQuAKE-Hard to fix "knowledge-conflicting annotation mistakes" in MQuAKE-3K | not obtained | [SNIPPET] [arXiv](https://arxiv.org/abs/2401.10471) |
| RAE (CIKM 2024) | retrieval of edited fact chains by mutual-information maximisation + pruning; in-context | GPT-2, GPT-J, Falcon, Vicuna, Llama2 | CF-3k (3,000 edits), T | not obtained | repo [PRIMARY] [GitHub](https://github.com/sycny/RAE) |
| GMeLLo (EMNLP 2024 Findings) | KG + LLM (question → relation chain → KG query) + MeLLo | GPT-J-6B, Vicuna-7B | MQuAKE-CF, k=1…3000 | GPT-J: 76.3% (k=1) → **49.0% (k=3000)**, MeLLo 20.3% → 9.8%. Vicuna-7B at k=3000: **41.9%** (MeLLo 10.2%) | [SNIPPET] [ACL Anthology](https://aclanthology.org/2024.findings-emnlp.844/) |
| KEDKG (AAAI 2025) | external dynamic KG with conflict resolution + fine-grained retrieval/filtering (no weight edit) | Llama-2-7B (others not verified) | MQuAKE-CF-3K, T; batch 1 / 100 / all | +108.1% (batch 1), +96.2% (batch 100), +83.9% (all edits) over the best prior baseline. H-Acc 63.67% vs PokeMQA 30.60% (Llama-2-7B, batch 1). 4-hop M-Acc >50% vs <25% for others | [SNIPPET] [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/34655) |
| G-Walk (ICLR 2025, MQuAKE-Remastered) | graph walk over the edited KG (retrieval) | Mistral-7B-Instruct-v0.2, Llama-2-7b-hf (repo) | Remastered-CF-3k, 1000 edits | 57.37% vs MeLLo 14.63% | [SNIPPET] [ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf); repo [PRIMARY] [GitHub](https://github.com/henryzhongsc/MQuAKE-Remastered) |
| CHECK (IJCAI 2025, Simon & Ewetz; arXiv 2508.00914) | retrieval / in-context. Compiler-style "type safety": the output types of one hop must overlap the input types of the next, otherwise re-decompose | GPT-J, Vicuna-7B, Falcon-7B | MQuAKE-CF-3k, MQuAKE-2002, MQuAKE-Hard, MQuAKE-T | Average +22.8% over 5 frameworks. Gains over next best: +31.57 (CF-3k), +28.51 (2002), +24.79 (Hard), +16.77 (T). **78.69 = MQuAKE-T with GPT-J.** **42.27 = GPT-J on MQuAKE-CF-3k**; one search extract attributed 42.27 to MQuAKE-T, which conflicts. The edit batch k and whether "CF-3k" means the original or v2 could NOT be verified | [SNIPPET] [IJCAI](https://www.ijcai.org/proceedings/2025/0916.pdf); [arXiv](https://arxiv.org/abs/2508.00914) |
| HYPE (ACL Findings 2025; arXiv 2505.18343, "Lifelong Model Editing with Graph-Based External Memory") | hybrid: hyperbolic (Poincaré) knowledge graph + Möbius-addition **parameter** updates + gradient masking / periodic GNN reset | GPT-J, GPT2-XL | CF, CF+, MQuAKE (version / k not verified) | +10.18 pts multi-hop over the best baseline; +9.42 specificity | [SNIPPET] [ACL Anthology](https://aclanthology.org/2025.findings-acl.690/) |
| AcE (arXiv 2510.07896) | weight edit of neuron-level query→value pathways ("implicit subjects act as query neurons") | GPT-J, Qwen3-8B | multi-hop factual recall (MQuAKE; version / k not read) | +9.44% over SOTA on GPT-J, +37.46% on Qwen3-8B | [PRIMARY abstract] [AcE](https://huggingface.co/papers/2510.07896) |
| MCircKE (arXiv 2604.05876, Apr 2026, "Addressing the Reasoning Gap") | weight edit restricted to the causal circuit (attention heads + MLPs) for the reasoning task | – | MQuAKE-3K | numbers not obtained. A search extract says "accepted to EMNLP 2026 Findings" (unverified) | [SNIPPET] [arXiv](https://arxiv.org/abs/2604.05876) |
| CaKE (arXiv 2503.16356) | circuit-aware weight editing for multi-hop generalisation | – | – | not read | [HF listing](https://huggingface.co/papers/2503.16356) |

### Inferences
- **Retrieval vs weights:**
  - Retrieval / in-context / KG methods: MeLLo, PokeMQA, DeepEdit, RAE, GMeLLo, KEDKG, CHECK, G-Walk.
  - Weight edits: MEMIT, ROME, MEND, AcE, MCircKE, CaKE and (hybrid) HYPE.
  - GLAME (graph-augmented weight editing, 2024) could not be retrieved.
  - All strong 3,000-edit results are retrieval-based. No weight-editing method was found with more than ~10% multi-hop at k = 3,000 on MQuAKE-CF with GPT-J (MEMIT 5.4, MEND 6.1).
- **CHECK's 42.27% is most likely the k = 3,000 (all-edits-at-once) setting** (inference). The reported +31.57 gain on CF-3k implies a next-best of about 10.7%, which matches MeLLo- or PokeMQA-level numbers at k = 3,000 (MeLLo GPT-J 9.8–14.2). The same arithmetic gives a next-best of about 61.9% on MQuAKE-T.
  - If this reading is right, CHECK's comparison set evidently did not include GMeLLo (49.0% at k = 3,000, GPT-J, per snippet), or it used a different dataset version. The report writer should not call CHECK "best at 3,000 edits" without that caveat.
- **Best reported multi-hop accuracy at 3,000 edits:**
  - **GPT-J:** GMeLLo 49.0% [SNIPPET], then CHECK 42.27% [SNIPPET]. For comparison, MeLLo is 14.2% [PRIMARY, v3] and MEMIT 5.4% [PRIMARY].
  - **Larger / API models:** the only primary number obtained is MeLLo with GPT-3.5 at 30.9% (CF, k = 3,000). PokeMQA, DeepEdit and RAE GPT-3.5 or larger-model numbers were not obtained.
  - **On the corrected benchmark:** G-Walk 57.37% at 1,000 edits.
- The MeLLo GPT-J baseline differs between sources (38.9 → 14.2 in MQuAKE v3 vs 20.3 → 9.8 quoted in GMeLLo). This shows how dataset versions and prompts shift the baselines that every "+x%" claim rests on.

### Gaps
- CHECK's exact table (k, dataset version, the five baselines, Vicuna and Falcon numbers) could not be read; the IJCAI and arXiv PDFs were blocked. The 42.27 and 78.69 figures are only partly confirmed.
- No numbers were obtained for DeepEdit, RAE, GLAME, MCircKE, CaKE, or AcE's full table.
- KEDKG's absolute M-Acc at "all edits" and its model list were not read.
- No 2025–26 paper was found reporting a weight-editing method at k = 3,000 on MQuAKE-Remastered.

## Benchmark reliability: MQuAKE-Remastered, teacher-forced vs generative evaluation, other critiques

### Takeaway
- **Multi-hop editing numbers on the original MQuAKE are unreliable.** The Remastered audit (ICLR 2025 Spotlight) finds **33–76% of questions/labels corrupted** (edit contamination, missing information in instructions, conflicting edits, duplicates). The MQuAKE authors themselves re-issued CF-3k in Sep 2024 (v2), and DeepEdit had already proposed MQuAKE-2002 and MQuAKE-Hard.
- **Single-hop and lifelong numbers inflate heavily under teacher-forced evaluation.** Switching to live-decoding evaluation (WILD) cuts single-edit success from 96.8% (reported) to 38.5%, and 1,000 sequential edits fall to about 10%.
- **Independent re-runs** (EtCon, NAS, UltraEdit, the AlphaEdit reproducibility study) repeatedly find published lifelong editors collapsing at longer horizons or on newer models.

### Cited Findings

**MQuAKE-Remastered (ICLR 2025 Spotlight; Zhong, Lu, … Chaudhary, Hu)**
- Official repo: "We audit the MQuAKE benchmark … and find that **33–76%** of its questions and labels are corrupted due to unintentional clerical or procedural oversights." It releases MQuAKE-Remastered plus **G-Walk**, "a simple graph-walk-based approach … that does not exploit the quirks of the original dataset". [PRIMARY] [GitHub](https://github.com/henryzhongsc/MQuAKE-Remastered)
- HF dataset card: the fixes cover "edit contamination, missing information in question instructions, conflicting edits, and duplicate cases". [PRIMARY] [HF dataset](https://huggingface.co/datasets/henryzhongsc/MQuAKE-Remastered)
  - It releases four splits:
    - Remastered-CF-3k (same size as CF-3k).
    - Remastered-CF-9k (full size).
    - Remastered-CF-6334 (train/test splits for parameter-based editors, with edit_num ∈ {100, 1000, 3000, 6334}).
    - Remastered-T.
- The paper PDF was only reachable via search extract [SNIPPET] [ICLR 2025 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/f782860c2a5d8f675b0066522b8c2cf2-Paper-Conference.pdf):
  - Original MQuAKE-CF-3k-v2 still contains **17 duplicate cases and 633 cases with missing information** in the multi-hop question instructions.
  - Error types are named "Inner Contamination" and "Intra Contamination":
    - Fixing Inner Contamination matters most at high editing intensity (e.g., 3,000 edits).
    - Fixing Intra Contamination matters most at low intensity (e.g., 100 edits).
  - Remastered-CF-3k "fixed substantially more errors than MQuAKE-CF-3K-v2" while keeping 1,000 cases per 2-, 3- and 4-hop.
  - **G-Walk: 57.37% vs MeLLo 14.63%** on CF-3k in the 1,000-edit setting. The model was not verified; the repo supports Mistral-7B-Instruct-v0.2 and Llama-2-7b-hf.
- Repo scope: G-Walk and MeLLo on Mistral-7B-Instruct-v0.2 and Llama-2-7b-hf. [PRIMARY] [GitHub](https://github.com/henryzhongsc/MQuAKE-Remastered)
- The MQuAKE authors' own fix: "We have fixed this subset in 2024/9 to resolve an issue of knowledge conflict". [PRIMARY] [MQuAKE v3](https://huggingface.co/papers/2305.14795)
- DeepEdit's MQuAKE-2002 and MQuAKE-Hard "resolve knowledge-conflicting annotation mistakes in the popular KE benchmark MQuAKE-3K". [SNIPPET] [DeepEdit](https://arxiv.org/abs/2401.10471)

**"The Mirage of Model Editing: Revisiting Evaluation in the Wild"** (Yang, Sun et al., ICT-CAS / Baidu; arXiv 2502.11177) [PRIMARY] [Mirage](https://huggingface.co/papers/2502.11177)
- **New benchmark and protocol:** QAEdit (edits aligned with standard QA datasets) and WILD (Without Intervention, Live Decoding).
- **Single editing:** "current editing methods perform substantially worse than previously reported (38.5% vs. 96.8%)".
- **Four flaws in "synthetic" evaluation:**
  - identical edit and test prompts;
  - **teacher forcing**, which leaks both the content and the length of the ground truth;
  - **truncating output to the target length**;
  - a match-ratio metric that rewards partial matches.
  - Teacher forcing and target-length truncation cause the largest overestimation.
- **Sequential editing:** under WILD, "average success rates dropping to ∼10% for only 1000 samples".
- Methods tested include ROME and WISE on Llama-2-7b-chat and two other LLMs.

**Other critiques and re-evaluations**
- **EtCon:** "a significant gap exists between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning scenarios". Under a GPT-4.1 judge, AlphaEdit scores 0.0% on CounterFact and QAEdit (Qwen-2.5-7B-Instruct) and MEMIT collapses. [PRIMARY] [EtCon](https://huggingface.co/papers/2512.04753)
- **UltraEdit** adopts WILD (LLM-as-judge) as a complementary metric but reports token Exact Match as primary. [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679)
- **NAS / AlphaEdit / RLSEdit CounterFact metrics are teacher-forced probability comparisons** (P[o_new] > P[o_old] on the prompt). ZsRE uses top-1 token accuracy. [PRIMARY] [NAS appendix A.2](https://huggingface.co/papers/2602.02543)
- **WikiBigEdit:** knowledge-editing techniques transfer poorly to large-scale real-world lifelong updates, and RAG or continual fine-tuning with model merging can do better. [PRIMARY] [WikiBigEdit](https://huggingface.co/papers/2503.05683)
- **AlphaEdit reproducibility study** (Jul 2026): the original metrics reproduce, but there is a fluency/consistency reporting discrepancy, the advantage fails to generalise to Qwen2.5 / Gemma-2 / Phi-3, it degrades at longer horizons, and it harms BoolQ / HellaSwag / XSTest. [SNIPPET] [arXiv](https://arxiv.org/abs/2606.26783)
- **"Should We Really Edit Language Models? On the Evaluation of Edited Language Models"** (arXiv 2410.18785): performance deteriorates with large-scale edits, instruction-tuned models are more robust, and safety is reduced (HF listing description only). [HF](https://huggingface.co/papers/2410.18785)
- **"Suppressed, Not Erased"** (arXiv 2609.18985, Sep 2026): a representational trace of edited facts survives even weight-free editing. If the original object stays linearly decodable, "the edit certificate is incomplete". [SNIPPET] [arXiv](https://arxiv.org/html/2609.18985)
- **Titles only (not read):**
  - "Beyond Endpoint Scores: Time- and Capacity-Conditioned Evaluation of Continual Knowledge Updating" (arXiv 2609.03900, Sep 2026).
  - "The Butterfly Effect of Model Editing" (arXiv 2402.09656): few edits can trigger collapse.
  - "Model Editing at Scale leads to Gradual and Catastrophic Forgetting" (arXiv 2401.07453).
  - HF listings: [2609.03900](https://arxiv.org/pdf/2609.03900), [2402.09656](https://huggingface.co/papers/2402.09656), [2401.07453](https://huggingface.co/papers/2401.07453).

### Inferences
- **Yes, reported editing numbers collapse under realistic evaluation.** The documented drops are:
  - single edit: 96.8 → 38.5 (WILD);
  - 1,000 sequential edits: about 10% (WILD);
  - AlphaEdit's 94–99 Eff (teacher-forced, b100, a few thousand edits) becomes 0–16% in EtCon's judge-based lifelong evaluation on Qwen-2.5.
- **MQuAKE's lenient metric** (correct on any 1 of 3 paraphrases) plus dataset corruption means pre-2025 MQuAKE-CF-3k gains, especially "+x% over MeLLo/PokeMQA", should be treated as provisional unless re-run on Remastered or at least CF-3k-v2.
- **The trustworthiness ranking implied by the evidence:**
  1. WikiBigEdit (real diffs, auto-extending, multi-axis).
  2. QAEdit + WILD.
  3. MQuAKE-Remastered.
  4. MQuAKE-CF-3k-v2 / MQuAKE-2002 / MQuAKE-Hard.
  5. Original MQuAKE-CF-3k, CounterFact / ZsRE under teacher forcing (least trustworthy as absolute numbers; still useful for relative stress tests of collapse).

### Gaps
- The exact split of "33%" vs "76%" between CF-3k-v2 and CF-3k could not be confirmed from the primary paper. My unverified recollection is roughly 33% for v2 and 76% for the original CF-3k; confirm before citing.
- The full list of methods MQuAKE-Remastered re-benchmarked (and how much each changed after fixing) was not obtained.
- It is unknown whether G-Walk's 57.37 uses Mistral-7B or Llama-2-7B, and whether it is on Remastered-CF-3k.

## Novelty check: frozen LM + zero-init product-key slot memory, symbolic (subject × relation-rotation) keys, ROME-style targets, and a nightly exact joint ridge re-solve over all facts

### Takeaway
- **Every individual ingredient has published precedent:**
  - sparse / product-key memory slots on a frozen or partly-frozen LM (Sparse Memory Finetuning 2025; Ordo-M 2026 [SNIPPET]);
  - entity-name-addressed slots (Ordo-M [SNIPPET]);
  - keys from a separately encoded subject + relation (KBLaM);
  - ROME-style target vectors (ROME/MEMIT/AlphaEdit);
  - closed-form ridge solves for edits (MEMIT batch LS, UltraEdit per-turn ridge);
  - most importantly, **exact cumulative ridge least squares over all edits so far** (RLSEdit, Jan 2026, via recursive least squares on a dense MLP matrix);
  - memory-based deletion (Larimar selective forgetting; codebook removal in GRACE-style memories).
- **I found no paper that combines all of them.** In particular I found none that periodically re-solves *all* values of a *sparse, symbolically addressed slot memory* by exact ridge LS (CG), and that uses the result for order-independence, exact deletion by re-solve, and a residual-based capacity-overflow certificate.
- The combination, and the certificate/deletion/order-independence guarantees, look new as a package. The core "joint LS over all edits" idea is not new (RLSEdit, MEMIT lineage), and the "entity-addressed slot table on a frozen LM" idea has a very close Aug-2026 preprint (Ordo-M) whose details I could not verify.

### Cited Findings

**Prior art, closest first**
- **RLSEdit** (arXiv 2601.15686, Jan 2026) [PRIMARY] [RLSEdit](https://huggingface.co/papers/2601.15686)
  - Lifelong editing as "online regularized least squares" with W*_t = argmin Σ_{i≤t}‖K_i W − V_i‖² + λ²‖W − W0‖² + μ²‖K0 W − V0‖², solved via Woodbury with per-edit cost independent of history.
  - It gives deviation bounds and an asymptotic ridge-consistency result.
  - Tested at 10K edits (b100) on Llama-3-8B and Qwen2.5-7B.
  - It edits a **dense in-place MLP weight** with native activations as keys. In the text I read, it makes no claims of deletion, strict order-independence, or a capacity certificate.
- **Ordo-M: "An Externally Addressed Sparse Memory Grafted onto a Frozen Language Model"** (Aug 2026, found only on alphaXiv under a non-standard id) [SNIPPET] [alphaXiv](https://www.alphaxiv.org/abs/2608.ordo-m-sparse-memory-frozen-model)
  - A "table of trainable value slots inserted into the forward pass" of a frozen LLM.
  - "symbolic or lexical addresses derived from entity names in the prompt": the entity name is a unique key to a slot.
  - Deterministic addressing gives "zero drift" / "zero-collateral edits".
  - 32 measurement rounds on three real domains and two standard editing benchmarks. CounterFact recall is 89.0% at the entity's own address vs 2.2% at a foreign address. Aimed at single-consumer-GPU users.
  - Unknown: LS vs gradient values, relation binding, deletion, product keys.
- **Zenodo record 20258685** (May 2026), "Trainable Sparse Vector Memory for Frozen Language Models: A Controlled Synthetic Binding Study with Qwen2.5-0.5B" [SNIPPET] [Zenodo](https://zenodo.org/records/20258685)
  - Frozen Qwen2.5-0.5B plus a router MLP, trainable keys, a sparse value table, a value projection and a sigmoid gate. Values are injected as a **residual update after transformer layer 8**.
  - Tested on synthetic entity-code bindings: a 100-fact task reaches 1.00 exact match on seen facts and 0.00 on unseen facts.
- **Continual Learning via Sparse Memory Finetuning** (FAIR Meta + UC Berkeley, arXiv 2510.15103) [PRIMARY] [HF](https://huggingface.co/papers/2510.15103)
  - Memory-layer models (Berges et al. 2024, "Memory Layers at Scale", product-key style) update only the memory slots highly activated by new knowledge relative to their pretraining usage.
  - NaturalQuestions F1 drop: −89% for full fine-tuning, −71% for LoRA, −11% for sparse memory fine-tuning, at equal new-knowledge acquisition.
  - Values are trained by gradient descent inside a pretrained memory-layer model; they are not a zero-init add-on solved by LS.
  - 2026 follow-ups (titles only): "Improving Sparse Memory Finetuning" ([arXiv 2604.05248](https://arxiv.org/html/2604.05248)) and "Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning" ([arXiv 2605.03229](https://arxiv.org/pdf/2605.03229)).
- **UltraEdit** [PRIMARY] [UltraEdit](https://huggingface.co/papers/2505.14679)
  - Closed-form ridge update Δθ = (HᵀH + I)⁻¹HᵀV per turn, with lifelong feature normalisation (online whitening).
  - It is a ridge solve, but only over the current batch and in dense weights.
- **MEMIT / AlphaEdit** [PRIMARY] [AlphaEdit Eq. 14–15](https://huggingface.co/papers/2410.02355)
  - Batch least squares: Δ_MEMIT = R K1ᵀ(Kp Kpᵀ + K1 K1ᵀ + K0 K0ᵀ)⁻¹. Here K1 are the current edits' keys, Kp the previously edited keys and K0 the preserved keys.
  - AlphaEdit projects Δ onto the null space of K0.
  - Joint LS within a batch; sequential batches are not a full re-solve.
- **KBLaM** (ICLR 2025, Microsoft) [PRIMARY] [KBLaM](https://huggingface.co/papers/2410.10450)
  - Each KB triple becomes continuous key-value vectors "via pre-trained sentence encoders with linear adapters", injected by rectangular attention.
  - More than 10K triples into an 8B LLM on one A100; "dynamic updates without model fine-tuning or retraining".
  - Precedent for keys built from a separately encoded (name, property).
- **Larimar** (ICML 2024, IBM) [PRIMARY abstract] [Larimar](https://huggingface.co/papers/2403.11901)
  - A distributed episodic memory attached to an LLM with "dynamic, one-shot updates"; 8–10× editing speed-ups.
  - "mechanisms for selective fact forgetting, information leakage prevention".
  - Its Kanerva-machine-style least-squares write is from my recollection, not re-verified.
- **GRACE** [PRIMARY listing] [GRACE](https://huggingface.co/papers/2211.11031)
  - Caches layer activations in a discrete key-value codebook with a deferral radius.
  - It memorises but generalises poorly: Gen .08 at T = 1,000 on LLaMA-2-7B [PRIMARY, WISE Table 2] [WISE](https://huggingface.co/papers/2405.14768).
- **WISE** (side memory + router + sharding / merging), **MEMOIR** (residual memory, sparse sample-dependent masks, TopHash retrieval) and **MELO** (dynamically activated LoRA blocks indexed by a vector database). Descriptions from the NAS appendix A.4 [PRIMARY] [NAS](https://huggingface.co/papers/2602.02543) and [WISE](https://huggingface.co/papers/2405.14768).
- **MeG** (ICLR 2026): a single dynamic neuron whose weights a diffusion model generates per query, gated by a "familiarity network". [SNIPPET / README] [GitHub](https://github.com/RodeWayne/MeG-for-Knowledge-Editing)
- **Patent:** US 12450168, "Dynamic updating of content addressable associative memories for large language models". Assignee and claims were not checked; only the title appeared in a search result. [SNIPPET] [USPTO](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12450168)
- **Deletion caveat:** "Suppressed, Not Erased" (arXiv 2609.18985) argues that behavioural success does not certify erasure if the original fact remains linearly decodable. [SNIPPET] [arXiv](https://arxiv.org/html/2609.18985)

### Inferences
- **What seems genuinely new** (no prior art found, within the limits of a blocked-proxy search):
  1. **A periodic exact global re-solve of all slot values over every stored fact**, as a "sleep / consolidation" step on a sparse, symbolically addressed memory. RLSEdit is the closest, but it is recursive, in dense weights, and its targets V_i are computed against the then-current edited model.
  2. **Order-independence as a guarantee.** It follows if the keys are deterministic (snapped subject × relation rotation) and the per-fact targets are computed against the frozen base model rather than the evolving memory. That assumption must hold in the user's pipeline; if targets are optimised with the memory in the loop, order-dependence re-enters.
  3. **Exact deletion by re-solving without the fact.** Classical for ridge regression, but I found no knowledge-editing paper that frames and tests it this way. Note that deletion from the memory does not delete the base model's original belief ("Suppressed, Not Erased" caveat). What gets restored is the base model's behaviour, not erasure of that fact.
  4. **A residual-based certificate that flags capacity overflow** (a per-fact LS residual / conditioning test). No editing paper found uses this as a capacity monitor; closest is RLSEdit's residual-based bounds.
  5. **Relation binding by a relation-specific random rotation of a whitened, snapped subject code.** This is VSA/HRR-style binding. I found no knowledge-editing paper using it; KBLaM and Ordo-M use encoder- or lexical-name keys instead.
- **What is not new:** sparse/product-key memory slots attached to an LM, zero-init residual add-ons, entity-addressed slot tables on frozen LMs (Ordo-M, if its preprint holds up), gradient-optimised ROME-style value targets, and joint LS over edit keys (MEMIT lineage, RLSEdit).
- **Scientific risks the literature points to for this design:**
  - **Weak paraphrase generalisation for lookup memories:** GRACE's Gen .08, and WISE's "impossible triangle" of reliability, generalisation and locality under lifelong editing.
  - **No multi-hop propagation when the key is the prompt's explicit subject at one layer.** AcE shows that second hops depend on *implicit* subjects ("query neurons") inside the model, which a subject-snapped key never sees.
  - **Inflated results under teacher-forced evaluation** (Mirage). A credible claim should use WILD / generative evaluation, WikiBigEdit, and MQuAKE-Remastered rather than original CF-3k.
- **Background (my knowledge, not re-sourced here):** optimal linear associative memories solved by pseudo-inverse / least squares go back to Kohonen (1970s). T-Patcher (ICLR 2023) adds one trainable key-value neuron per error. Reviewers will likely cite both.

### Gaps
- **Ordo-M could not be opened** (alphaXiv blocked, non-standard id). Whether it uses product keys, LS value solves, relation binding, deletion or residual certificates is unknown. It is the highest-priority item to check manually before claiming novelty.
- The contents of the Zenodo 20258685 study, "Trained Persistent Memory for Frozen Decoder-Only LLMs" (arXiv 2603.22329), "… Encoder–Decoder LLMs: Six Architectural Methods" (arXiv 2603.16413), "Memory Grafting" (arXiv 2605.20948) and "Knowledge Offloading" (arXiv 2605.29075) were not read.
- Patent US 12450168 claims were not reviewed.
- Larimar's exact write rule (LS vs other) was not re-verified.
- There was no exhaustive search for "sleep / nightly consolidation" editing papers or "certified unlearning" in editing memories; the web-search budget was exhausted.
