# Knowledge editing as the "write" operation for a lifelong-learning LLM (ROME/MEMIT → AlphaEdit → UltraEdit/LocFT-BF; GRACE/WISE; MEND/RLEdit; multi-hop and evaluation failures)

Verification labels used throughout:
- **[P]** = number read directly from the primary paper's full text (arXiv HTML/PDF text via the Hugging Face paper mirror `hf://papers/<id>/paper.md`; arxiv.org itself was blocked by the proxy).
- **[S]** = number taken from a secondary source (a search-engine snippet or summary of the abstract). Not checked against the full paper.
- Unless stated otherwise, "Eff/Gen/Spe" = Efficacy / Generalization (paraphrase) / Specificity (locality), and "Rel/Gen/Loc" = Reliability / Generalization / Locality.

---

## 1. ROME and MEMIT: mechanism, edit quality, batch limits, compute per edit

### Takeaway
ROME and MEMIT write a fact (subject, relation, object) into the mid-layer MLP weights with a closed-form least-squares update. ROME changes one layer with a rank-one update; MEMIT spreads a batch of edits over several layers. For single edits or one batch they look very good: ROME scores 89–92 on CounterFact, and MEMIT keeps an 85.8 score after 10,000 batched edits on GPT-J. These numbers come from probability-comparison or teacher-forced metrics. A 10k batch took hours on a GPU, and the facts are stored in one direction only.

### Cited Findings
- **Mechanism (ROME).**
  - Causal tracing finds that "feedforward MLPs at a range of middle layers are decisive when processing the last token of the subject name". ROME then changes that MLP's weights with a rank-one update.
  - Edits work best "at the last subject token, where both specificity and generalization peak at middle layers" [P]. — [ROME, Meng et al. 2022, arXiv 2202.05262](https://huggingface.co/papers/2202.05262)
- **ROME on CounterFact (single edits) [P]:**
  - GPT-2 XL: Score 89.2, Efficacy (ES) 100.0, Paraphrase (PS) 96.4, Neighborhood/specificity (NS) 75.4, fluency GE 621.9 (unedited 626.6), consistency RS 41.9.
  - GPT-J: Score 91.5, ES 99.9, PS 99.1, NS 78.9. The unedited GPT-J had NS 83.0.
  - Fine-tuning on GPT-J reached 100% efficacy but NS 10.3: "nearly 90% of neighborhood prompts are incorrect".
  - CounterFact has 21,919 records, 42,876 paraphrase prompts and 82,650 neighborhood prompts.
  - Source: [ROME, 2202.05262](https://huggingface.co/papers/2202.05262)
- **Mechanism (MEMIT).**
  - MEMIT is "a scalable multi-layer update algorithm that uses explicitly calculated parameter updates to insert new memories", aimed at "thousands of associations for GPT-J (6B) and GPT-NeoX (20B)" [P].
  - Earlier work had edited "at most a few dozen facts; a recent study evaluates on a maximum of 75".
  - Source: [MEMIT, Meng et al. ICLR 2023, arXiv 2210.07229](https://huggingface.co/papers/2210.07229)
- **AlphaEdit's closed-form statement of the MEMIT update [P]:**
  - Δ_MEMIT = R K1ᵀ (Kp Kpᵀ + K1 K1ᵀ + K0 K0ᵀ)⁻¹.
  - R = V1 − W K1 is the residual for the new facts, K1 are the new keys, Kp are keys of previously edited facts, and K0 are keys of preserved knowledge (sampled Wikipedia).
  - Source: [AlphaEdit, 2410.02355](https://huggingface.co/papers/2410.02355)
- **MEMIT at 10,000 edits in one batch, CounterFact [P]:**
  - GPT-J: Score 85.8, ES 98.9, PS 88.6, NS 73.7, GE 619.9, RS 40.1. Unedited GPT-J: NS 83.5, GE 622.4.
  - GPT-NeoX-20B: Score 82.0, ES 97.2, PS 82.2, NS 70.8.
  - At the same 10k scale, ROME (applied sequentially) scored 50.3 (ES 50.2, RS 3.3) and MEND scored 23.1, which means essentially no change.
  - Source: [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
- **MEMIT at 10,000 edits, zsRE, GPT-J [P]:**
  - Efficacy 96.7, Paraphrase 89.7, Specificity 26.6 (unedited 27.0), Score 50.7.
  - Here "Efficacy" is "the proportion of cases where o is the argmax generation given p(s,r)", i.e. token-level argmax.
  - Source: [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
- **Scaling curves in the MEMIT paper [P]:**
  - "ROME performs well up to n = 10 but degrades starting at n = 32."
  - MEND "rapidly declines at n = 6, losing all efficacy before n = 1,000".
  - Source: [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
- **Compute per edit [P]:**
  - Runtime for 10,000 edits: MEND 98 s, fine-tuning ~29 min, MEMIT 7.44 h, ROME 12.29 h.
  - The authors say MEMIT's implementation "does not batch the independent z_i optimizations", so it could be parallelized.
  - Source: [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
- **Representational limit [P]:** the method covers only directional (s,r,o) relations. "Tim Cook is CEO of Apple" "must be processed separately from the opposite association" (the reverse direction). It does not cover spatial, temporal, mathematical, procedural or symmetric knowledge. — [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
- **Format brittleness [P]:** after a successful zsRE-style QA-format edit, "the model is unable to produce the correct answer in approximately 70% scenarios when prompted in a text completion format". — [Gupta et al. 2024, arXiv 2401.07453](https://huggingface.co/papers/2401.07453)

### Inferences
- Divide the MEMIT runtime by 10,000 edits and you get about 2.7 s per edit for MEMIT and about 4.4 s per edit for ROME on a GPU with GPT-J 6B. This is my arithmetic from the [P] totals, not a reported per-edit figure.
- Specificity already falls noticeably at 10k batched edits: NS goes from 83.5 to 73.7 on CounterFact.
- Edits are stored in one direction only (subject → object). A user who teaches "X is my dentist" would not automatically get "Who is my dentist? → X" from the same edit.

### Gaps
- No ROME/MEMIT per-edit timings on phone-class hardware (NPU or mobile GPU) were found.

---

## 2. Sequential/lifelong editing collapse: when do ROME/MEMIT (and others) break?

### Takeaway
When edits are applied one after another, the classic locate-then-edit methods break after only hundreds to a few thousand edits:
- ROME collapses somewhere between about 100 and 1,000 edits, and a single "disabling edit" can trigger it.
- ROME, R-ROME and MEMIT collapse "within the first few hundred updates" on real Wikidata edits.
- In realistic free-generation evaluation, ROME and MEMIT reach about 0% reliability after 1,000 single edits.

Two causes are named: over-optimized activations with unbounded growth of the edited matrix's norm, and interference that comes from knowledge superposition.

### Cited Findings
- **Gupta et al. 2024, CounterFact, GPT-2 XL and GPT-J [P]:**
  - Two phases: "an initial gradual but progressive forgetting phase followed by an abrupt or catastrophic forgetting".
  - ROME's efficacy stays "almost 100% until a point where it begins to decline. This point of decline can come as early as 100 edits, or as late as 1000 edits".
  - Neighborhood accuracy "consistently declines" before that point.
  - The collapse is caused by "disabling edits": after one, "new knowledge edits are no longer successful on the model, the model forgets all previously edited facts and is unable to perform any downstream tasks".
  - MEND "almost instantaneously forgets all previously edited fact[s]".
  - Source: [Model Editing at Scale leads to Gradual and Catastrophic Forgetting, arXiv 2401.07453](https://huggingface.co/papers/2401.07453)
- **Butterfly Effect [P]:** "even a single edit can trigger model collapse". In sequential editing on hard cases, "nearly all examined editing methods result in model collapse after only few edits". The authors released the HardEdit dataset. — [Yang et al. 2024, arXiv 2402.09656](https://huggingface.co/papers/2402.09656)
- **General abilities [P]:** editing gains "may come at the cost of a significant degradation of the model's general abilities", attributed to "overfitting to the edited facts". Their RECT regularizer keeps "over 94% editing performance". — [Gu et al. 2024, arXiv 2401.04700](https://huggingface.co/papers/2401.04700)
- **AlphaEdit paper, LLaMA3-8B, sequential edits in batches of 100 [P]:**
  - With baseline editors, general capability (GLUE/MMLU F1) degrades significantly after 2,000 edited samples: "all metrics are rapidly approaching zero".
  - After 2,000 edits, MEMIT on LLaMA3 CounterFact scores Eff 65.65 / Gen 64.65 / Spe 51.56 with fluency 437.43 (unedited 635.23).
  - MEMIT on LLaMA3 ZsRE scores 34.62 / 31.28 / 18.49, and ROME on LLaMA3 ZsRE scores 2.01 / 1.80 / 0.69.
  - Source: [AlphaEdit, 2410.02355](https://huggingface.co/papers/2410.02355)
- **WikiBigEdit, Llama-2-7b, real Wikidata edits [P]:**
  - ROME, R-ROME and MEMIT "rapidly degrade within the first few hundred updates, leading to model collapse".
  - WISE "avoids catastrophic failure" but "steadily declines within the first 10K updates, ultimately converging to pre-update knowledge levels".
  - Source: [Thede et al., ICML 2025, arXiv 2503.05683](https://huggingface.co/papers/2503.05683)
- **Mirage of Model Editing, 1,000 sequential single edits on QAEdit, synthetic vs WILD evaluation [P]:**

  | Method | Llama-2-7b-chat | Mistral-7b | Llama-3-8b |
  |---|---|---|---|
  | ROME | 0.114 → 0.001 | 0.059 → 0.001 | 0.034 → 0.001 |
  | MEMIT | 0.057 → 0.002 | 0.058 → 0.002 | 0.000 → 0.000 |
  | GRACE | 0.370 → 0.015 | 0.416 → 0.018 | 0.368 → 0.022 |
  | WISE | 0.802 → 0.195 | 0.735 → 0.060 | 0.526 → 0.072 |
  | FT-M | 0.973 → 0.531 | 0.960 → 0.454 | 0.925 → 0.229 |
  | Average WILD reliability | 0.124 | 0.089 | 0.065 |

  - The authors conclude that methods "fail drastically with only 1000 edits" (~10% average success).
  - Source: [Yang et al. 2025, arXiv 2502.11177](https://huggingface.co/papers/2502.11177)
- **Mechanistic cause 1: regularization (ENCORE) [P]:**
  - Degradation comes from "(1) over-optimization of internal activations and (2) continuous norm-growth of edited matrices" (MEMIT, Llama3-8B).
  - Fixes: Most-Probable Early Stopping plus a Frobenius-norm constraint, which enable "up to 10,000 sequential edits while maintaining original downstream performance".
  - The speed-up is reported as "42-61%" faster in the abstract and "41-62%" in the introduction (an internal inconsistency).
  - Source: [Gupta et al. 2025, arXiv 2502.01636](https://huggingface.co/papers/2502.01636)
- **Mechanistic cause 2: superposition [S]:**
  - Extending the closed-form linear-associative-memory solution to lifelong editing produces an "interference term" tied to superposition of knowledge representations.
  - "When knowledge superposition does not exist … the interference term vanishes, allowing for lossless knowledge editing".
  - Superposition is reported as universal across real LMs, with heavy-tailed, high-kurtosis distributions.
  - Source: [Hu et al., AAAI 2025, arXiv 2408.07413](https://arxiv.org/abs/2408.07413)
- **NeuralDB's summary of the state of the art [P]:** "existing methods can only edit hundreds of facts without compromising the models' general abilities". — [arXiv 2507.18028](https://huggingface.co/papers/2507.18028)
- **2026 follow-up (existence only) [S]:** a January 2026 paper, "Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse" (arXiv 2601.11042), exists. Its content was not accessible.

### Inferences
- For a phone assistant that gets a few facts a day, plain ROME or MEMIT would fail within weeks to months: roughly 100–1,000 edits, or a few hundred on realistic data.
- These methods are not a viable lifelong write path without the newer regularization and projection fixes.
- The "disabling edit" risk means one bad write can destroy the model. A deployed system would need checkpointing or rollback and a health check such as perplexity after each write. The Butterfly paper proposes perplexity as a surrogate.

### Gaps
- None of the collapse studies use sub-3B, phone-class models; results are for GPT-2 XL (1.5B), GPT-J 6B, and 7–8B Llama/Mistral.
- Collapse points under 4-bit or 8-bit quantized weights (the likely on-device format) were not found.

---

## 3. AlphaEdit and the 2025–2026 follow-ups: how far has sequential weight editing been pushed?

### Takeaway
Null-space projection (AlphaEdit, ICLR 2025) moved MEMIT-style editing from about 1k to about 3k sequential edits without losing general capability, but it too collapses at around 8k–20k edits.

The largest scales claimed in 2025–2026 are:
- **UltraEdit:** 2,000,000 edits in batches of 100, with about 80% Eff, about 76–79% Gen and about 76–78% Spe on its own benchmark (Exact-Match metric).
- **StableEdit (ICML 2026):** 2M sequential edits on Llama-3-8B [S].
- **LocFT-BF:** 100K edits under strict autoregressive (WILD) evaluation, with 99.8% reliability but only 57.6% generalization on Qwen2.5-72B.
- **NeuralDB:** 100,000 facts, using an in-FFN gated key-value module.

The papers conflict. LocFT-BF reports that UltraEdit and RLEdit fail beyond about 20K edits under WILD evaluation, while UltraEdit reports stability to 2M under its own metrics.

### Cited Findings
- **AlphaEdit mechanism [P]:**
  - The edit perturbation is projected onto the null space of the preserved-knowledge keys K0, so that (W + ΔP)K0 = W K0 = V0.
  - The closed-form result is Δ_AlphaEdit = R K1ᵀ P (Kp Kpᵀ P + K1 K1ᵀ P + I)⁻¹.
  - P "only needs to be computed once", adding "negligible additional time", and "a single line of additional code" gives a 36.7% average improvement to locate-then-edit methods.
  - Source: [AlphaEdit, Fang et al., ICLR 2025, arXiv 2410.02355](https://huggingface.co/papers/2410.02355)
- **AlphaEdit numbers after 2,000 sequential edits in batches of 100 [P]:**
  - LLaMA3-8B: CounterFact Eff 98.90 / Gen 94.22 / Spe 67.88 / Flu 622.49; ZsRE 94.47 / 91.13 / 32.55. MEMIT on the same setup: 65.65 / 64.65 / 51.56 and 34.62 / 31.28 / 18.49.
  - GPT-J: CounterFact 99.75 / 96.38 / 75.48; ZsRE 99.79 / 96.00 / 28.29.
  - GPT2-XL: CounterFact 99.50 / 93.95 / 66.39.
  - General capability is maintained "even after editing 3,000 samples".
  - Source: [2410.02355](https://huggingface.co/papers/2410.02355)
- **AlphaEdit at larger scale:**
  - RLEdit's abstract: AlphaEdit "experiences significant performance decline at 8,000 edits" [S]. — [RLEdit, arXiv 2502.05759](https://arxiv.org/abs/2502.05759)
  - UltraEdit's table at 20K edits (batch 100) [P]:
    - Mistral-7B-v0.3: 0.00 on every metric.
    - LLaMA-3-8B-Instruct: ZsRE 74.34 / 67.85 / 22.94; WikiBigEdit multi-hop reasoning 0.01.
    - On UltraEditBench, LLaMA-3 scores 4.51 / 3.16 / 2.78.
    - Source: [UltraEdit, arXiv 2505.14679](https://huggingface.co/papers/2505.14679)
- **RLEdit (hypernetwork trained with reinforcement learning over the whole edit sequence) [S]:**
  - Lifelong editing is treated as a Markov decision process whose rewards are editing losses, with "memory backtracking".
  - Reported "59.24% improvement while requiring only 2.11% of the time compared to most approaches".
  - Source: [arXiv 2502.05759](https://arxiv.org/abs/2502.05759)
  - RLEdit at 20K edits, as reported by UltraEdit [P]: LLaMA-3-8B ZsRE 91.34 / 89.68 / 41.94. — [2505.14679](https://huggingface.co/papers/2505.14679)
- **UltraEdit mechanism [P]:**
  - "Training-, subject-, and memory-free". Parameter shifts are computed "in one step using only a hidden state and its gradient".
  - The closed-form update is Δθ = (HᵀH + I)⁻¹ HᵀV.
  - A "lifelong normalization" keeps running mean and variance of the concatenated [hidden state ∥ gradient] features. Removing it drops Eff 84.47 → 36.15. Freezing its statistics collapses Eff to 1.11.
  - The paper claims more than 7× faster editing and 4× less VRAM than the previous SOTA. It calls itself the "only method currently capable of editing a 7B LLM on a 24GB consumer-grade GPU".
  - Source: [2505.14679](https://huggingface.co/papers/2505.14679)
- **UltraEdit scale results [P]:**
  - ULTRAEDIT* was run on 100K edits (ZsRE, FEVER), 500K (WikiBigEdit) and 2M (UltraEditBench). Baselines were run on 20K (17K for WikiBigEdit). Each turn has 100 samples.
  - Results at 2M edits on UltraEditBench (Eff / Gen / Spe):

    | Model | Eff | Gen | Spe |
    |---|---|---|---|
    | GPT-J | 81.65 | 76.80 | 76.44 |
    | Mistral-7B-v0.3 | 81.70 | 77.25 | 77.09 |
    | LLaMA-3-8B-Instruct | 83.45 | 79.11 | 78.05 |
    | Qwen2.5-7B-Instruct | 80.70 | 75.78 | 76.01 |

  - At 100K ZsRE edits, LLaMA-3-8B-Instruct scores 87.80 / 85.48 / 46.74.
  - The primary metric is "Exact Match" following AlphaEdit and RLEdit, with WILD LLM-as-judge as a complementary metric.
  - Source: [2505.14679](https://huggingface.co/papers/2505.14679)
  - Versions differ: a search summary of an earlier version said "up to 1 million edits" [S], while the current full text says 2M [P].
- **StableEdit ("More Edits, More Stable", May 2026; code repo tagged ICML'26) [S]:**
  - Recent long-horizon editors share "Lifelong Normalization (LN), which normalizes value gradients using running statistics"; "removing LN causes immediate performance collapse".
  - With ridge regression, LN gives "asymptotic orthogonality and bounded norms".
  - StableEdit adds warm-up plus full whitening and "successfully execut[es] 2 million sequential edits on Llama-3-8B".
  - Source: [arXiv 2605.11836](https://arxiv.org/abs/2605.11836); [GitHub MINE-USTC/StableEdit](https://github.com/MINE-USTC/StableEdit)
- **RLSEdit (Jan 2026) [P]:**
  - Recursive least-squares with soft constraints: deviation from the pre-trained weights plus deviation from an anchor mapping, updated with the Woodbury identity.
  - "Per-edit cost independent of history length".
  - "Stable scaling to 10K edits" on Llama-3 and Qwen2.5, "retaining early edits" and preserving GLUE and held-out reasoning/code.
  - Source: [arXiv 2601.15686](https://huggingface.co/papers/2601.15686)
- **HiEdit (ACL 2026) [S]:** hierarchical RL chooses which layers to perturb for each edit. It improves RLEdit "by an average of 8.48% with perturbing only half of the layers per edit". — [arXiv 2604.11214](https://arxiv.org/abs/2604.11214)
- **LocFT-BF / "Fine-tuning Done Right" (Sep 2025) [P]:**
  - Classic fine-tuning baselines were weak because of a "depth-first" pipeline: each edit is optimized to convergence, one sample at a time.
  - A "breadth-first" (epoch-based) pipeline with mini-batches plus localized tuning (MLP down/up-projection in later layers) instead achieved:
    - +33.72% average reliability (up to +58.50%) over the next-best method on 3K edits, evaluated with WILD autoregressive decoding.
    - 100K sequential ZsRE edits on LLaMA3-8B and Qwen2.5-7B with "near-perfect success rates" and preserved capability.
    - On Qwen2.5-72B with 100K edits: reliability 99.80%, generalization 57.60%, capability 62.54% (original 64.84%).
  - Contradiction with UltraEdit: "RLEdit and UltraEdit struggle to scale, with model capabilities decline sharply when edits exceed 20K", with "consistently low editing success rates" under WILD.
  - Efficiency: LocFT-BF, RLEdit and UltraEdit finish "each edit within one second"; other baselines are "nearly 50 times slower".
  - Source: [arXiv 2509.22072](https://huggingface.co/papers/2509.22072)
- **NeuralDB (Jul 2025) [P]:**
  - Linear locate-then-edit is reframed as querying a key-value database.
  - The linear perturbation is replaced by a "gated non-linear retrieval module" of edited facts inside the FFN, and appending, modifying and deleting entries is supported.
  - Performance drops "only 3% as the edited facts scale from 2000 to 10,000", and the method is shown at "100,000 edited facts (45x more edited facts than AlphaEdit)" on GPT-2 XL, GPT-J and Llama-3-8B with general abilities preserved.
  - Source: [arXiv 2507.18028](https://huggingface.co/papers/2507.18028)
- **MEMOIR (NeurIPS 2025) [S]:**
  - Residual-memory module; sparse, data-dependent masks confine each edit to a distinct parameter subset; at inference, sparse activation patterns are matched against those stored at edit time.
  - A search summary reported ZsRE with 1,000 edits: average 0.93 on LLaMA-3, 14 points above AlphaEdit; average 0.88 at 15,000 edits while MEMIT and AlphaEdit fell toward zero. Not verified from the full text.
  - Source: [arXiv 2506.07899](https://arxiv.org/abs/2506.07899)
- **REPAIR (Oct 2025) [P, abstract]:** closed-loop feedback plus dynamic memory management and knowledge fusion; "boosts editing accuracy by 10%-30% across multiple model families". — [arXiv 2510.01879](https://huggingface.co/papers/2510.01879)
- **EtCon (Dec 2025) [P, abstract]:** targeted proximal SFT (trust region) followed by a GRPO "consolidation" stage that aligns edited knowledge with chain-of-thought generation. It points to "a significant gap … between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning". — [arXiv 2512.04753](https://huggingface.co/papers/2512.04753)

### Inferences
- **Largest demonstrations found:**
  - 2M sequential edits: UltraEdit [P] and StableEdit [S], both on 7–8B models.
  - 100K edits under the stricter WILD evaluation, with capability retained: LocFT-BF [P].
  - 100K facts with an in-weight key-value module: NeuralDB [P].
- No method was found demonstrating more than 2M edits.
- Quality at 2M is "about 4 in 5 correct on the edit prompt, about 3 in 4 on paraphrases, about 3 in 4 unrelated facts preserved", measured by Exact Match. That metric is likely teacher-forced or token-level in the AlphaEdit/RLEdit tradition. This is an inference: the paper says it follows those protocols, and it also reports WILD separately.
- UltraEdit and LocFT-BF are directly contradictory about UltraEdit beyond 20K edits. The disagreement tracks evaluation protocol (Exact Match vs WILD autoregressive with natural stopping). Treat UltraEdit's 2M headline as an upper bound until reproduced under WILD.
- LocFT-BF's "breadth-first" pipeline iterates "over the entire dataset across epochs". A phone that truly never forgets would therefore have to keep every taught fact and periodically re-train on all of them. That is effectively a replay buffer, not a one-shot streaming write.
- NeuralDB, MEMOIR, WISE and GRACE keep the knowledge "in weights" (no context window), but as a growing, gated, retrieval-like parameter memory rather than a distributed change to the base weights.
- For the "answer from weights without context" requirement these do qualify. They behave like an internal key-value store with a router.

### Gaps
- Full-text numbers for StableEdit, MEMOIR and RLEdit (primary tables) could not be read because arxiv, OpenReview, NeurIPS and alphaxiv were blocked. Their numbers are secondary.
- A 2026 reproducibility study of AlphaEdit (arXiv 2606.26783) appeared in search results, but its content was not accessible.
- No study found that runs any of these at 1M+ edits on a 1–4B model or on device.

---

## 4. Memory/adaptor lifelong editors: GRACE, WISE, MELO, T-Patcher (and trade-offs)

### Takeaway
Adaptor/memory editors avoid overwriting the base weights by adding a codebook, a side memory or extra neurons, plus a router. They hold up far better on retention and locality: GRACE keeps locality at about 1.00, and WISE's locality is 1.00 at 3K edits. They pay in generalization to paraphrases (GRACE Gen 0.03 at 2–3K edits), and their parameters grow with the number of edits. Under realistic free-generation evaluation they collapse too: WISE at 1k sequential edits drops to 0.06–0.20 and GRACE to about 0.02.

### Cited Findings
- **GRACE mechanism [P]:** GRACE "writes new mappings into a pre-trained model's latent space, creating a discrete, local codebook of edits without altering model weights". It is "the first method enabling thousands of sequential edits using only streaming errors". — [Hartvigsen et al., NeurIPS 2023, arXiv 2211.11031](https://huggingface.co/papers/2211.11031)
- **GRACE numbers [P]:**
  - T5 (60M), zsRE, 1,000 sequential edits: Test Retention Rate (TRR) .69, Edit Retention Rate (ERR) .96, average .82, using 137 keys (7.30 edits/key). MEND scores TRR .25 / ERR .27 and FT .56 / .82.
  - BERT on SCOTUS: TRR .81 / ERR .82 with 252 keys.
  - GPT2-XL hallucination task, 1,392 edits: TRR perplexity 15.84, ERR perplexity 7.14, perplexity on already-accurate sentences 10.00, with 1,341 keys (1.04 edits/key).
  - Time per edit: .13 s (ROME .64 s, FT .26 s).
  - Source: [2211.11031](https://huggingface.co/papers/2211.11031)
- **WISE mechanism [P]:**
  - Editing long-term memory (weights) or working memory (retrieved activations) creates an "impossible triangle" of reliability, generalization and locality.
  - WISE uses "a dual parametric memory scheme": the main memory is frozen, and a side memory (a copy of a mid-to-late FFN layer) is edited, with a router.
  - Knowledge sharding puts edit shards in distinct random-mask subspaces, which are then merged.
  - Source: [Wang et al., NeurIPS 2024, arXiv 2405.14768](https://huggingface.co/papers/2405.14768)
- **WISE numbers [P]:**
  - ZsRE at T = 1,000 edits: average 0.83 on LLaMA-2-7B and 0.79 on Mistral-7B, "improvements of 18% and 11% over the nearest competitor".
  - Scaling to 3K edits on LLaMA-2-7B (Rel / Gen / Loc):

    | Method | Rel | Gen | Loc | Avg |
    |---|---|---|---|---|
    | WISE-Retrieve | 0.61 | 0.58 | 1.00 | 0.73 |
    | WISE-Merge | 0.58 | 0.56 | 1.00 | 0.71 |
    | WISE-Retrieve oracle routing | 0.75 | 0.70 | 1.00 | 0.82 |
    | GRACE | 0.96 | 0.03 | 1.00 | 0.66 |
    | MEMIT-MASS | 0.58 | 0.53 | 0.47 | — |

  - Capacity: "20% FFN parameters can accommodate at least 500 edited samples".
  - Overheads: WISE-Merge has a constant ~3% inference delay, 0.64% extra parameters and 4% extra VRAM. WISE-Retrieve adds ~7% inference time after 3K edits.
  - Setup detail: each edit prompt is augmented with 10 random token sequences of length 10.
  - Source: [2405.14768](https://huggingface.co/papers/2405.14768)
- **WISE/GRACE under stricter or larger evaluation [P]:**
  - Mirage (1,000 sequential edits, WILD): WISE 0.195 / 0.060 / 0.072 and GRACE 0.015–0.022 (Llama-2 / Mistral / Llama-3).
  - In single-edit WILD, "GRACE and WISE … exhibit the most significant decrease, with both reliability and generalization dropping below 5%", because they generate wrong content after the correct answer.
  - Source: [2502.11177](https://huggingface.co/papers/2502.11177)
  - WikiBigEdit: WISE converges to pre-update accuracy within 10K updates. — [2503.05683](https://huggingface.co/papers/2503.05683)
  - UltraEdit table, WISE at 20K ZsRE edits: GPT-J 34.13 / 33.14 / 26.81 and LLaMA-3 40.94 / 40.27 / 37.40. — [2505.14679](https://huggingface.co/papers/2505.14679)
- **MELO [P, abstract]:** a "neuron-indexed dynamic LoRA" that "dynamically activat[es] certain LoRA blocks according to the index built in an inner vector database". It reports SOTA on three sequential editing tasks (document classification, QA, hallucination) "while requir[ing] the least trainable parameters and computational cost". No numeric results were extracted. — [Yu et al., AAAI 2024, arXiv 2312.11795](https://huggingface.co/papers/2312.11795)
- **T-Patcher [S]:**
  - Adds "a few neurons in the last Feed-Forward Network layer" (one mistake, one neuron) and "can successively correct up to thousands of errors" with generality and locality maintained.
  - Cost per edit reported as 7.1 s (fact-checking) and 18.9 s (QA).
  - In the same study, MEND's sequential success rate was 0.04 on FEVER and 0.41 on zsRE.
  - Source: [Huang et al., ICLR 2023, arXiv 2301.09785](https://arxiv.org/abs/2301.09785)

### Inferences
- Codebook and side-memory editors are the closest existing match to a phone "never forget" memory that stays in weights: base weights are untouched, locality is about 1.0, and rollback is easy.
- Their weakness is exactly the user-facing one: recognizing a differently-phrased later question.
  - GRACE's key-matching fails on paraphrases (Gen 0.03).
  - WISE's routing is the bottleneck: 0.73 with learned routing vs 0.82 with oracle routing at 3K.
- Parameter and memory growth is linear in the number of edits, or per shard of about 500 edits for WISE. At 1M facts that could be substantial on a phone, though no one reports it.

### Gaps
- No MELO or T-Patcher primary tables were read.
- No memory-footprint numbers at 100K–1M edits were found for GRACE, WISE, MELO or T-Patcher.

---

## 5. MEND and learned/amortized editors (hypernetworks: fact → weight delta in one forward pass)

### Takeaway
Hypernetwork editors such as MEND are fast at edit time, about 0.01 s per edit (98 s for 10k), but classic MEND fails outright in sequential and large-batch settings. MALMEN raised batch capacity by "hundreds of times". RLEdit, which trains the hypernetwork with RL over whole edit sequences, reaches tens of thousands of lifelong edits. Hypernetworks are also the only weight-based route to multi-hop "propagation" so far (PropMEND), but their absolute accuracy is low and they generalize poorly out of domain.

### Cited Findings
- **MEND [P]:** small auxiliary networks "learn to transform the gradient obtained by standard fine-tuning, using a low-rank decomposition of the gradient". It "can be trained on a single GPU in less than a day even for 10 billion+ parameter models" and was "the only approach … that effectively edits the behavior of models with more than 10 billion parameters" at the time. — [Mitchell et al., ICLR 2022, arXiv 2110.11309](https://huggingface.co/papers/2110.11309)
- **MEND at scale [P]:**
  - At 10,000 batched CounterFact edits, MEND scores 23.1, which leaves the model "nearly unchanged". — [MEMIT, 2210.07229](https://huggingface.co/papers/2210.07229)
  - Sequentially, it "almost instantaneously forgets all previously edited fact[s]". — [2401.07453](https://huggingface.co/papers/2401.07453)
  - In the Mirage sequential run it scored 0.000. — [2502.11177](https://huggingface.co/papers/2502.11177)
  - In AlphaEdit's 2,000-edit run on LLaMA3, MEND's ZsRE Eff was 0.91. — [2410.02355](https://huggingface.co/papers/2410.02355)
- **MALMEN [P, abstract/introduction]:**
  - Replaces MEND's summing of parameter shifts with a least-squares aggregation solved via the normal equation, and separates hypernetwork and LM computation to allow arbitrary batch size.
  - Edits "up to thousands of facts" on BERT-base, GPT-2, T5-XL (2.8B) and GPT-J (6B); it is "capable of editing hundreds of times more facts than MEND" and "outperforms … MEMIT".
  - It notes that prior sequential meta-learning "only scale[s] to few, e.g., 10, updates", because hypernetworks "overfit to the present state of the LM".
  - Source: [Tan et al., ICLR 2024, arXiv 2311.04661](https://huggingface.co/papers/2311.04661)
- **Why hypernetworks drift in lifelong use [P]:** "the hypernetwork remains fixed while the underlying model continues to evolve as edits accumulate. This growing mismatch can lead to degraded editing performance over time." — [UltraEdit, 2505.14679](https://huggingface.co/papers/2505.14679)
- **RLEdit [S; P via UltraEdit table]:** hypernetwork plus RL over the full edit sequence. At 20K edits it reaches LLaMA-3-8B ZsRE 91.34 / 89.68 / 41.94 and UltraEditBench 85.69 / 81.88 / 65.64. Under WILD, LocFT-BF reports that it degrades sharply beyond 20K. — [2502.05759](https://arxiv.org/abs/2502.05759); [2505.14679](https://huggingface.co/papers/2505.14679); [2509.22072](https://huggingface.co/papers/2509.22072)
- **Hypernetwork scaling laws (Jul 2026) [P, abstract]:**
  - Hypernetworks generate a fixed LoRA for "train-time knowledge injection" from a large fact corpus. MegaWikiQA has "tens of millions of multi-hop question-answer examples across 39 knowledge domains".
  - Results: "broadly predictive power law scaling along all architecture axes" and "reliable OOD generalization to unseen entities and relations at increasing scales", with "steeper scaling exponents in all OOD evaluations" than LoRA or full fine-tuning.
  - Source: [arXiv 2607.19604](https://huggingface.co/papers/2607.19604)

### Inferences
- A trained hypernetwork on the phone could make each edit a single forward/backward pass, the cheapest write operation available.
- But MEND-style editors need retraining as the base model drifts, and only RL-trained variants (RLEdit, HiEdit) have been shown at 10^4 scale, with disputed results beyond 20K.
- The 2026 scaling-law work suggests amortized injection improves predictably with hypernetwork size. It studies batch (train-time) injection, not lifelong streaming.

### Gaps
- MALMEN's exact maximum batch size and scores were not extracted from its tables.
- No numbers were found on hypernetwork editors at more than 100K lifelong edits.

---

## 6. Multi-hop and ripple failures: are edited facts used in reasoning? Any weight-based fixes?

### Takeaway
No. Edited facts are recalled well in the exact form they were written, but they are rarely used as premises. In MQuAKE, multi-hop accuracy on GPT-J falls from 40.5% before editing to 7.0–8.0% after ROME, MEMIT or MEND, even with 88–96% edit-wise success, and falls further with more edits. RippleEdits finds only 38–66% accuracy on logical consequences, and a simple in-context baseline beats all weight editors. Retrieval-based MeLLo fixes much of it but is not a weight method. Weight-based propagation methods (PropMEND) roughly double accuracy on non-verbatim questions, from 12.7% to 22.4%. At scale, even the best 2025 lifelong editors reach only about 29–36% on WikiBigEdit multi-hop.

### Cited Findings
- **MQuAKE-cf, per-instance (≤4 edits) [P]:**

  | Model / method | Edit-wise success | Multi-hop | Multi-hop (CoT) |
  |---|---|---|---|
  | GPT-J, before editing | — | 40.5 | 39.5 |
  | GPT-J, ROME | 88.3 | 7.4 | 19.9 |
  | GPT-J, MEMIT | 96.2 | 7.0 | 11.8 |
  | GPT-J, MEND | 73.2 | 8.0 | 11.7 |
  | GPT-J, FT | 46.9 | 1.5 | — |
  | Vicuna-7B, before editing | — | 30.2 | — |
  | Vicuna-7B, MEMIT | — | 4.9 | — |
  | Vicuna-7B, ROME | 99.7 | 6.9 | — |

  - The authors: methods "are rather hard coding them into the model by updating weights locally".
  - Source: [Zhong et al., EMNLP 2023, arXiv 2305.14795](https://huggingface.co/papers/2305.14795)
- **MQuAKE-t, real Wikidata 2021→2023 changes, GPT-J [P]:**
  - Before editing: multi-hop 34.3 (CoT 46.8).
  - After editing: ROME 0.3 (CoT 11.3), MEMIT 0.3 (4.8), MEND 16.0 (38.2), while ROME and MEMIT had 100.0 edit-wise success.
  - Injecting k = 100 / 1,000 / 3,000 instances at once made multi-hop "further drop".
  - Source: [2305.14795](https://huggingface.co/papers/2305.14795)
- **MeLLo [P, qualitative]:** stores edits in an external memory and prompts the model to decompose questions and self-check against retrieved edits. It "outperforms MEMIT and MEND significantly across all the settings", and the gap widens with GPT-3.5. It is not weight-based. Exact numbers from its Table 5 were not extracted. — [2305.14795](https://huggingface.co/papers/2305.14795)
- **RippleEdits [P]:**
  - Tests logical generalization, compositionality I/II, subject aliasing, preservation and relation specificity.
  - Data: 54 relations and 18–26 test queries per edit. Section 4.2 lists 2,000 Recent + 1,000 Random + 1,000 Popular edits (4,000), while the abstract says "5K" (internal inconsistency).
  - Existing editors (MEND, ROME, MEMIT on GPT-2 XL, GPT-J, LLaMA-7B and GPT-NeoX-20B) show "low average accuracy of 38−66 across all models".
  - "A simple in-context editing baseline obtains the best scores".
  - Source: [Cohen et al., TACL 2024, arXiv 2307.12976](https://huggingface.co/papers/2307.12976)
- **Weight-based propagation fix, PropMEND [P]:**
  - A MEND-style hypernetwork meta-trained so gradient updates on a fact let the model answer multi-hop questions about it.
  - RippleEdit non-verbatim questions: 22.4% vs 12.7% for the next best ("almost 2×").
  - Controlled RippleEdit: 64.0% in-domain but 17.7% on the hardest out-of-domain setting (unseen relations or entities).
  - Source: [Liu et al. 2025, arXiv 2506.08920](https://huggingface.co/papers/2506.08920)
- **Multi-hop at lifelong scale [P]:**
  - WikiBigEdit "Reasoning" (multi-hop) after about 17K–500K edits:
    - UltraEdit: 29.27 (GPT-J), 35.80 (Mistral-7B), 35.64 (LLaMA-3-8B).
    - RLEdit: 25.94, 22.41, 28.13.
    - AlphaEdit: 0.07, 0.00, 0.01.
  - Source: [2505.14679](https://huggingface.co/papers/2505.14679)
  - In WikiBigEdit's own study, even RAG leaves multi-hop "near pre-update levels"; "only 10% of evaluation samples retrieve both facts correctly". — [2503.05683](https://huggingface.co/papers/2503.05683)
- **Consolidation approach [P, abstract]:** EtCon argues edited facts are "insufficiently integrated into LLMs' inference-time behavior under autoregressive generation". It adds a GRPO stage over CoT trajectories. — [2512.04753](https://huggingface.co/papers/2512.04753)

### Inferences
- An on-device "write" that uses any current weight editor would store facts as retrievable associations but not as usable premises.
- "My sister moved to Lyon" would probably not be reflected in "What country does my sister live in?" or "What's the weather where my sister lives?" (inference from MQuAKE and RippleEdits).
- Chain-of-thought helps: MQuAKE ROME goes from 7.4 to 19.9 with CoT. Explicitly recalling the edited fact first partially compensates.
- Weight-based propagation (PropMEND, EtCon, hypernetwork scaling) is an active 2025–2026 direction, but no method reaches pre-edit multi-hop levels at scale.

### Gaps
- MeLLo's exact numbers and RippleEdits' per-criterion tables were not extracted (the tables were images in the HTML conversion).
- No "knowledge propagation" method was found evaluated on thousands of lifelong edits.

---

## 7. Evaluation critiques: are editing success numbers inflated? Real free-form QA performance

### Takeaway
Yes, substantially. Standard synthetic evaluation uses the same prompt as the edit, teacher forcing, truncation at the target length, and token-match ratios. Under it, editors score about 96%. With realistic autoregressive generation, natural stopping and an LLM judge, the same edits score about 38–44%, and sequential editing collapses to about 10% at 1k edits. Newer critiques show that edited facts fail simple negation and fact-checking reformulations, which suggests shortcut learning rather than knowledge integration.

### Cited Findings
- **Mirage / WILD [P]:**
  - Single-edit average success "38.5% vs. 96.8%". Synthetic evaluation around 96% vs WILD 43.8% on ZsRE and 38.9% on QAEdit.
  - The four flaws named: identical prompts for editing and testing, teacher forcing, "output truncation" at the ground-truth length, and match-ratio metrics.
  - "Teacher forcing and target length truncation cause the most significant overestimation."
  - Source: [2502.11177](https://huggingface.co/papers/2502.11177)
- **Mirage controlled numbers, Llama-3-8b, 3,000 QAEdit samples [P]:**
  - Teacher forcing vs autoregressive decoding, context-guided prompt: FT-M 0.937 → 0.800; ROME 0.930 → 0.851; MEMIT 0.907 → 0.786; GRACE 0.412 → 0.036; WISE 0.838 → 0.592.
  - Ground-truth-length truncation vs natural stop (LLM judge, context-free): FT-M 1.000 → 0.202; ROME 0.954 → 0.478; MEMIT 0.886 → 0.461; GRACE 0.992 → 0.301; WISE 0.700 → 0.046.
  - Failure modes: edited models continue past the answer with "meaningless repetition", "irrelevant" or "incorrect information". Example: "Wilhelm Conrad Röntgen Wilhelm Conrad Röntgen …".
  - Source: [2502.11177](https://huggingface.co/papers/2502.11177)
- **"Is Model Editing Built on Sand?" (Oct 2025) [S]:**
  - Editing's "apparent reliability … rests on a fragile foundation"; the objective "encourage[s] exploiting hidden shortcuts, rather than utilizing real semantics".
  - "All nine state-of-the-art methods collapse entirely on all four datasets when tested with simple negation".
  - On ZsRE with Llama-3, edited models "still output the target word 80-90% of the time when prompted with a negation".
  - "A method with an 84% standard success rate dropped to just 29.9% accuracy in a fact-checking format".
  - Source: [arXiv 2510.00625](https://arxiv.org/abs/2510.00625)
- **Format sensitivity [P]:** about 70% of zsRE QA-format edits fail when queried as text completion. — [2401.07453](https://huggingface.co/papers/2401.07453)
- **Stricter evaluation reorders methods [P]:** under WILD, fine-tuning done right (LocFT-BF) beats AlphaEdit, RLEdit and UltraEdit by 33.72% average reliability at 3K edits. — [2509.22072](https://huggingface.co/papers/2509.22072)
- **EtCon [P, abstract]:** "a significant gap exists between their performance in controlled, teacher-forcing evaluations and their real-world effectiveness in lifelong learning scenarios". — [2512.04753](https://huggingface.co/papers/2512.04753)
- **RAG vs editing at scale [P]:**
  - On WikiBigEdit, RAG "vastly outperforms specialized knowledge editing techniques", "nearly tripling accuracy on training edits", with rephrase 78.64% and personas 66.16%. Inference time "at most doubles" after all of WikiBigEdit.
  - "Knowledge editing techniques like WISE and MEMIT … cannot match simple low-rank adapter-based finetuning (LoRA-FT) at scale", and LoRA + interpolation merging gives "consistent gains across all evaluation axes".
  - Source: [2503.05683](https://huggingface.co/papers/2503.05683)

### Inferences
- The headline editing numbers (ROME ~100%, MEMIT 98.9%, AlphaEdit 98.9%, UltraEdit 80%+ at 2M) should be read as upper bounds on a phone assistant's real recall of taught facts.
- The realistic single-edit baseline is about 40% correct in free generation (Mirage WILD), and methods must additionally stop generating cleanly.
- The strongest method under realistic evaluation is essentially "careful, localized, mini-batch fine-tuning with replay of all edits" (LocFT-BF). That points toward a periodic consolidation design rather than instant surgical edits.

### Gaps
- The "Built on Sand" full tables were not accessible (numbers are secondary).
- No independent reproduction of UltraEdit or StableEdit at 1M+ edits under WILD evaluation was found.

---

## 8. Benchmarks: CounterFact, zsRE, WikiBigEdit, UltraEditBench, UnKE/long-form, QAEdit, MQuAKE, RippleEdits

### Takeaway
Classic benchmarks are small and synthetic: CounterFact about 21.9K and zsRE about 1K–20K samples in common use. The ones that test lifelong editing at scale are WikiBigEdit (502K real Wikidata QA pairs, 8 timesteps) and UltraEditBench (more than 2M Wikidata5M-derived pairs). QAEdit plus WILD tests realistic QA. Long-form and unstructured editing (UnKE, AnyEdit's EditEverything) is a separate, less mature line.

### Cited Findings
- **CounterFact [P]:** 21,919 records, 20,391 subjects, 749 objects, 42,876 paraphrase prompts, 82,650 neighborhood prompts, 62,346 generation prompts. — [2202.05262](https://huggingface.co/papers/2202.05262)
- **Benchmark size survey [P]:** "existing benchmarks are small … CounterFact at 20K, ZsRE at 1K, SelfCheckGPT at 600". WikiFactDiff has more than 450K pairs but "only a 20K subset … is commonly used". — [2503.05683](https://huggingface.co/papers/2503.05683)
- **WikiBigEdit [P]:**
  - 502,382 QA pairs, 6.9M tokens, 8 timesteps from Wikidata snapshots (February–July 2024), 27k–69k edits per step, with "monthly factual changes exceeding 100k".
  - About 76% of raw changes were filtered out.
  - Evaluation axes: update, rephrase, personas, multi-hop and locality.
  - "Lower entity frequency correlates with reduced performance."
  - Source: [2503.05683](https://huggingface.co/papers/2503.05683)
- **UltraEditBench [P]:** more than 2M editing pairs built from Wikidata5M triples, with questions generated by GPT-4o-mini. Each pair has an editing instance (subject in the question), an equivalent (paraphrase) instance and an unrelated instance, and the answer is excluded from the question. Dataset: huggingface.co/datasets/XiaojieGu/UltraEditBench. — [2505.14679](https://huggingface.co/papers/2505.14679)
- **QAEdit [P]:** derived from three widely used QA datasets. Sequential tests used samples "incorrectly answered by all pre-edit LLMs"; the WILD framework is task-agnostic. — [2502.11177](https://huggingface.co/papers/2502.11177)
- **MQuAKE [P]:** MQuAKE-cf has 9,218 questions (a 3K subsample is used) covering 2-, 3- and 4-hop questions with 1–4 edits. MQuAKE-t has 1,868 real temporal edits from Wikidata 2021-04 vs 2023-04. — [2305.14795](https://huggingface.co/papers/2305.14795)
- **Long-form editing [P, abstract]:** AnyEdit identifies an "efficacy barrier" from editing a single token's hidden state. Its autoregressive chunk-wise editing "outperforms strong baselines by 21.5% on benchmarks including UnKEBench, AKEW, and … EditEverything". UltraEdit also reports on UnKE. AnyEdit is excluded from lifelong comparisons because it is "not designed for lifelong editing" (per UltraEdit). — [AnyEdit, ICML 2025, arXiv 2502.05628](https://huggingface.co/papers/2502.05628); [2505.14679](https://huggingface.co/papers/2505.14679)
- **Other benchmarks used by the lifelong papers [P]:**
  - FEVER, SelfCheckGPT hallucination, Temporal (out-of-distribution emerging entities) and SCOTUS. — [WISE 2405.14768](https://huggingface.co/papers/2405.14768); [GRACE 2211.11031](https://huggingface.co/papers/2211.11031)
  - MedEditBench. — [LocFT-BF 2509.22072](https://huggingface.co/papers/2509.22072)
  - HardEdit for collapse-inducing edits. — [2402.09656](https://huggingface.co/papers/2402.09656)

### Inferences
- For a phone "learn once, never forget" evaluation, the closest public proxies are:
  - WikiBigEdit: real, temporal, includes multi-hop and personas.
  - UltraEditBench: scale.
  - QAEdit/WILD: realistic decoding.
- None tests personal, user-specific facts or dialogue-taught facts. That would need a custom benchmark.

### Gaps
- The KnowEdit/EasyEdit comprehensive-study tables (arXiv 2401.01286) were not read in this pass.
- The UnKE paper (arXiv 2405.15349) was not available on the Hugging Face mirror.

---

## 9. Compute per edit and deployability constraints relevant to a phone

### Takeaway
Per-edit compute ranges from well under a second on a datacenter or consumer GPU (UltraEdit, RLEdit, LocFT-BF, GRACE) to several seconds (MEMIT, ROME, T-Patcher). All published results are GPU-based on 1.5B–72B models. No on-device (phone) editing results were found. Precomputed covariance or projection matrices (MEMIT, AlphaEdit) and hypernetwork pretraining (MEND, RLEdit) are extra fixed costs.

### Cited Findings
- **Per-edit and total times [P]:**
  - MEMIT 7.44 h and ROME 12.29 h for 10k edits on GPT-J; MEND 98 s. — [2210.07229](https://huggingface.co/papers/2210.07229)
  - GRACE .13 s per edit and ROME .64 s per edit on GPT2-XL. — [2211.11031](https://huggingface.co/papers/2211.11031)
  - T-Patcher 7.1 s (fact-checking) and 18.9 s (QA) per edit [S]. — [2301.09785](https://arxiv.org/abs/2301.09785)
- **Fast lifelong editors [P]:**
  - LocFT-BF, RLEdit and UltraEdit take "within one second" per edit; the others are "nearly 50 times slower". — [2509.22072](https://huggingface.co/papers/2509.22072)
  - UltraEdit claims more than 7× faster editing and 4× less VRAM than prior SOTA, and is the "only method … capable of editing a 7B LLM on a 24GB consumer-grade GPU". — [2505.14679](https://huggingface.co/papers/2505.14679)
- **Fixed precompute [P]:**
  - AlphaEdit's projection P "only needs to be computed once". — [2410.02355](https://huggingface.co/papers/2410.02355)
  - Locate-then-edit methods such as ROME and AlphaEdit "require computing large covariance and projection matrices, incurring substantial memory and time costs that make scaling to larger models practically infeasible". — [2509.22072](https://huggingface.co/papers/2509.22072)
- **Recursive per-edit cost [P]:** RLSEdit's per-edit cost is "independent of history length and scaling only with the current edit size". — [2601.15686](https://huggingface.co/papers/2601.15686)
- **Memory-based overhead [P]:** WISE-Merge adds 0.64% parameters, 4% VRAM and about 3% inference latency. — [2405.14768](https://huggingface.co/papers/2405.14768)

### Inferences
- Among current published methods, the realistic phone "write" candidates are those with sub-second, closed-form or one-pass updates and no large precomputed matrices: UltraEdit/StableEdit-style normalized least squares, RLSEdit's recursion, or LocFT-BF-style localized fine-tuning of one MLP projection.
- A LocFT-BF-like approach implies storing every taught fact and periodically re-running epochs over all of them ("sleep"-style consolidation) to keep 99%+ reliability.
- Everything here assumes full-precision (or at least bf16-class) weights with gradient access. How closed-form weight edits interact with quantized, NPU-deployed weights is unstudied in the sources found.

### Gaps
- No paper found measures editing latency, energy or memory on mobile SoCs, or with quantized (INT4/INT8) weights.
- No paper found reports multi-year retention (edits surviving later model updates or re-quantization).
