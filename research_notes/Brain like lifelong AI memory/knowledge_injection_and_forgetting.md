# Knowledge Injection and Forgetting: the data and training-signal side of permanent learning in LLMs

Scope: how an LLM can learn a new fact into its weights from very few exposures so the fact is usable later (recall, paraphrase, reverse direction, reasoning), what makes weight updates forget old knowledge, and brain-inspired consolidation (CLS, sleep/replay) applied to LLMs. Written for the design of an on-device AI that must learn from one telling or one web page, keep it permanently, and not lose older knowledge.

Verification tags used throughout:
- **[P]** = number read by me in the primary paper text (arXiv HTML/markdown served via the Hugging Face papers mirror `hf://papers/<id>`, because arxiv.org, huggingface.co, openreview.net and most publisher sites were blocked by the network proxy for direct fetch).
- **[S]** = secondary: from a search-engine summary of the abstract or proceedings page, a blog or social post, or an AI summary. Treat these as less reliable.
- Links go to the canonical arXiv/ACL/PMLR page even when the text was read through the HF mirror.

---

## 1. Physics of Language Models Part 3.1 / 3.2 / 3.3 (Allen-Zhu & Li): augmentation, exposures, capacity, quantization

### Takeaway
Seeing a fact is not enough. It has to appear in varied forms during training (paraphrases, reordered sentences, multiple write-ups); otherwise the model memorizes the text but cannot answer questions about it (9.7% accuracy without augmentation vs 96.6% with 5 varied biographies). Once facts are extractable, a sufficiently trained transformer holds about **2 bits of knowledge per parameter**. That capacity needs about **1000 exposures per fact**; facts seen only about **100 times** are stored at about **1 bit/param**. int8 quantization keeps the capacity, but **int4 cuts it to about 0.7 bit/param**. Models also cannot do inverse lookup ("which person has attribute X?"), and cannot compare or classify stored facts without chain-of-thought.

### Cited Findings
**Part 3.1: Knowledge Storage and Extraction** ([arXiv 2309.14316](https://arxiv.org/abs/2309.14316)) [P]
- Setup: synthetic biographies of 100k individuals (birth date, birth city, university, major, employer, work city), plus Llama-rewritten versions ("bioR"). Models were pretrained from scratch, then QA-finetuned on a fraction p of people and tested out-of-distribution on the rest. Example runs: 124M model pretrained 540 passes on bioS; 302M model 1000 passes on bioR; 682M model 1350 passes on bioS/bioR. — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- Abstract: "for knowledge to be reliably extracted, it must be sufficiently augmented (e.g., through paraphrasing, sentence shuffling, translations) *during pretraining*. Without such augmentation, knowledge may be memorized but not extractable, leading to 0% accuracy, regardless of subsequent instruction fine-tuning." — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- Mixed training (biographies plus QA for some people in the same pretraining run) gives out-of-distribution QA accuracy of **86.6%** on bioS single and **77.7%** on bioR single. — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- Pretrain-then-finetune with no augmentation (bioS single): QA finetune reaches **33%** on birthdate and does poorly on the other attributes. With **5 diverse biography entries per person** (different wording plus sentence shuffling), QA accuracy on held-out people rises **from 9.7% to 96.6%**. Multiplicity alone (multi5, no permutation) goes 9.7% → **41%**. Permutation alone (single+permute1 → single+permute5) goes **4.4% → 70%**. Translating English to French gives about **40%**. "More augmentation ⇒ better." — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- Mechanism (probing): with augmentation, attribute knowledge is stored almost linearly on the entity-name tokens. Without it, the knowledge is spread over the whole biography text and "knowledge extraction nearly impossible no matter how one finetunes it". In multi5+permute, probing accuracy for all six attributes is near 100% from the first position. — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- "Celebrity helps minority": augmenting only a subset of people (celebrities) also raises extraction accuracy for the non-augmented people. — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)
- Recommendations: "(1) rewrite the pretraining data — using small, auxiliary models — to provide knowledge augmentation, and (2) incorporate more instruction-finetuning data into the pretraining stage before it becomes too late." — [arXiv 2309.14316](https://arxiv.org/abs/2309.14316)

**Part 3.2: Knowledge Manipulation** ([arXiv 2309.14402](https://arxiv.org/abs/2309.14402)) [P]
- Models "excel in knowledge retrieval but struggle even in the simplest classification or comparison tasks unless Chain of Thoughts (CoTs) are employed during both training and inference." "Their performance in inverse knowledge search is virtually 0%, regardless of the prompts." This holds "even when such knowledge is perfectly stored in the models, despite adequate training and sufficient model size". The authors report GPT-4 and Llama-3 failing similar tests (as of May 10, 2024). — [arXiv 2309.14402](https://arxiv.org/abs/2309.14402)

**Part 3.3: Knowledge Capacity Scaling Laws** ([arXiv 2404.05405](https://arxiv.org/abs/2404.05405)) [P]
- "language models can and only can store **2 bits of knowledge per parameter, even when quantized to int8**". "A 7B model can store 14B bits of knowledge, surpassing the English Wikipedia and textbooks combined based on our estimation." Here knowledge is flexibly extractable, not verbatim memorization. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Exposures: "Achieving a 2bit/param capacity requires each knowledge piece to be visited **1000 times** during training, termed 1000-exposure." **With 100 exposures, an undertrained GPT2's capacity falls to 1 bit/param**. "Rare knowledge, encountered only 100 times during training, is stored at a 1bit/param ratio." — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Architecture: at 1000 exposures, 2 bit/param looks universal (even without MLP layers). At 100 exposures, LLaMA/Mistral (gated MLP) have **1.3× lower** capacity than GPT2 with rotary embeddings. The authors translate a 1.3× difference into accuracies of about 90% vs 70%. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Quantization (GPTQ): "Quantizing to **int8 does not compromise model capacity** (even for models on the boundary of 2bit/param); however, **quantizing to int4 reduces capacity to 0.7bit/param**." — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- MoE: with 32 experts, capacity is only 1.3× lower than dense while using 8.8% of parameters at inference. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Junk data: at a **1:7 useful:junk token ratio**, capacity for useful knowledge drops **20×** even with 100 exposures. The loss factor improves to **3×/1.5×/1.3× at 300/600/1000 exposures**. **Prepending a domain/source token** (like "wikipedia.org") to useful data improves the 20× loss to **2×**. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)
- Data definitions: bioS(N) regenerates each biography on the fly with random template selection and ordering (50 templates per attribute). bioR(N) has 40 LLaMA2 rewrites per person. For bioR, 1000/100 exposures mean only 25/2.5 passes. — [arXiv 2404.05405](https://arxiv.org/abs/2404.05405)

**Related capacity estimate (different definition)**
- Morris et al. 2025, "How much do language models memorize?": GPT-family models have "an approximate capacity of **3.6 bits-per-parameter**" for *unintended memorization* of random data. Models memorize until capacity fills, then "grokking" begins. The study trained hundreds of models from 500K to 1.5B parameters. [P] — [arXiv 2505.24832](https://arxiv.org/abs/2505.24832)

### Inferences
- For a phone-sized model (say 1–4B parameters), the Part 3.3 constant suggests a ceiling of about 2–8 Gbit of extractable factual knowledge in fp16/int8 when trained to saturation, and about 0.7 bit/param (about 0.7–2.8 Gbit) if the weights holding that knowledge are int4. This comes from synthetic data. That int4 weights cut stored knowledge by about 3× has not been checked on real LLMs. It still means an on-device design that keeps learned facts in int4 base weights is likely to lose capacity; keeping the "learning" parameters (adapters or memory slots) at int8 or higher is a design implication.
- "Learn after being told once" conflicts with the 100–1000-exposure regime. The only known workaround is to turn one exposure into many varied training examples (paraphrases, reorderings, QA pairs, implications, both directions), replayed over time. Sections 3–5 quantify this.
- A pretrained model is already near capacity. So new facts compete with old ones for bits, and forgetting is expected unless new capacity is added (memory layers, adapters, parameter growth; see sections 6 and 8).
- The Part 3.2 inverse-search result means a fact learned as "A → B" will not support "which X has B?" unless the reverse form is also trained (see section 2).

### Gaps
- Exact Part 3.1 "celebrity helps minority" accuracies and Part 3.2 per-task numbers were not captured (figures). The 3.3 capacity experiments train from scratch; no Physics-of-LM paper measures capacity for facts added *after* pretraining into an already-full model.
- Quantization effects beyond GPTQ int8/int4 (e.g., QAT, 3-bit, mixed precision) on stored-knowledge capacity: not found.

---

## 2. Reversal curse (Berglund et al. 2023): numbers and fixes

### Takeaway
Finetuning on "A is B" gives near-zero ability to answer "B is ?". The model does not even raise the likelihood of the correct answer above a random name, at any model size. Paraphrasing in the same direction does not help. Fixes all put the reverse direction into the training data: explicit bidirectional augmentation, or reverse training (entity-preserving word reversal), which doubles tokens.

### Cited Findings
- Experiment 1 (GPT-3-175B finetuned on fictitious "name ↔ description" facts, with **30 paraphrases per fact**): same-direction exact-match accuracy is **96.7%** (DescriptionToName) and **50.0%** (NameToDescription). Reverse direction is "close to 0%", no higher than outputting random names. Log-probability of the correct name vs a random name shows no significant difference (t-tests and KS tests). [P] — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)
- The same near-0% reversal held across a hyperparameter sweep for GPT-3-350M and Llama-7b. A larger ablation (100 examples × 100 paraphrases = 40,000 documents, GPT-3-350M, 10 epochs) did not fix it. Prompt-tuning Llama-7b also showed no reverse generalization. [P] — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)
- Real-world: GPT-4 answers "Who is Tom Cruise's mother?"-type questions **79%** of the time vs **33%** for the reverse (abstract). The body of v4 says GPT-4 scores **28% on "Child"** (the parent direction is 100% by construction). Note this discrepancy. [P] — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)
- Experiment 3 (instructions): all Llama-1 models got >80% on QuestionToAnswer and <7% on AnswerToQuestion, which is chance level. [P] — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)
- The abstract states the curse "is not alleviated by data augmentation" (same-direction paraphrases). [P] — [arXiv 2309.12288](https://arxiv.org/abs/2309.12288)
- Fix, reverse training (Golovneva et al. 2024): train on both {x} and {REVERSE(x)} (2N samples), "whereby all words are used twice, doubling the amount of available tokens". Variants are token, word, entity-preserving, and random-segment reversal. "Data-matched reverse-trained models provide superior performance to standard models on standard tasks, and compute-matched reverse-trained models provide far superior performance on reversal tasks." On a symbolic reversal task, "standard training completely fails". Entity-preserving reversal is needed for multi-word entities, and random-segment reversal works when the max segment length k is at least as long as the entities. [P] — [arXiv 2403.13799](https://arxiv.org/abs/2403.13799)
- Allen-Zhu & Li Part 3.2 independently found inverse knowledge search "virtually 0%" (section 1). [P] — [arXiv 2309.14402](https://arxiv.org/abs/2309.14402)
- EntiGraph's authors attribute part of raw-corpus CPT's failure to "limited diversity of knowledge representations … such as the reversal curse". [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- SCoL (2026) sees a reversal-curse-like mismatch in practice: a LoRA "Batch TTT" adapter trained jointly on 100 SQuAD passages plus implications scored *worse* than prompting alone (26.52% vs 28.17%). The authors attribute this to the training format (declarative) not matching the query format (QA). [P] — [arXiv 2605.07076](https://arxiv.org/abs/2605.07076)

### Inferences
- In an on-device learner, every learned fact should be written out in both directions (and ideally as QA in both directions) before any weight update. Otherwise the fact can only be reached from its original subject.

### Gaps
- Exact numeric results of reverse training on the celebrity/biography reversal tasks were in tables I could not extract. Quantified bidirectional-augmentation results at the level of individual facts (e.g., how many reversed paraphrases are needed) were not found.

---

## 3. Synthetic continued pretraining / EntiGraph (Yang et al. 2024)

### Takeaway
To teach an 8B model a small corpus (1.3M tokens), the authors expanded it about 350× into **455M synthetic tokens** by generating text about entity relationships. QA accuracy grew **log-linearly** with synthetic token count, rising from 39.49% to **56.22%**. That recovers **over 80% of the gain RAG gives** (60.35%), and the approach combines with RAG (62.60%). Continued pretraining on the raw corpus does worse than the base model. Plain rephrasing scales poorly.

### Cited Findings
- Setting: QuALITY, **265 books, 1.3M tokens**. **455M synthetic tokens** generated with gpt-4-turbo. **Llama 3 8B** continually pretrained for **2 epochs with replay of RedPajama**. [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- Closed-book QA: **39.49% (base) → 56.22% (EntiGraph CPT)**. Accuracy "scales log-linearly up to 455M tokens". The authors model it as a mixture of exponentials with three stages: linear growth, log-linear growth, plateau. [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- "Raw CPT performs even worse than Llama 3 8B Base." The Rephrase baseline was stopped at **38M tokens** because it "scales poorly". "For synthetic CPT to scale, the synthetic data must be sufficiently diverse." [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- Reference points: GPT-3.5 closed-book **44.81%**, GPT-4 closed-book **51.30%**. Open-book with the full document: GPT-3.5 **72.60%**, GPT-4 **86.09%**. [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- RAG (text-embedding-3-large + FAISS + Cohere rerank, Recall@8 = 99.63%): Llama 3 8B Base + RAG **60.35%**, EntiGraph CPT + RAG **62.60%**. "Adding RAG … improves accuracy by 20.86% (39.49% → 60.35%) … EntiGraph CPT improves accuracy by 16.73% (39.49% → 56.22%). Hence, EntiGraph continued pretraining provides > 80% of the absolute performance improvement of RAG." [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- Human analogy in the paper: "a 13-year-old human acquires knowledge from fewer than 100M tokens, while state-of-art open-source language models are trained on 15T tokens." [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- EntiGraph-trained instruct models produce fewer false claims in summaries than Raw- or Rephrase-trained models. [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- Later comparison: Active Reading's authors did not compare to EntiGraph "since it was shown to underperform synthetic QA (see their Appendix D)". [P] — [arXiv 2508.09494](https://arxiv.org/abs/2508.09494)

### Inferences
- For "read one web page once", EntiGraph-style numbers imply a synthetic-to-source expansion of about 100–350× to reach most of RAG's accuracy. For a 2k-token page that is roughly 0.2–0.7M generated tokens per page. That is likely too much to generate on a phone per page, so cheaper recipes (SEAL-style implications, QA generation, Active Reading strategies) and batched overnight "sleep" generation are needed. This is an extrapolation from a corpus-level result; the paper does not measure per-page scaling.

### Gaps
- No per-document or per-fact scaling curve (how many synthetic tokens per single fact). The 455M-token experiment used a GPT-4-class generator. How well a small on-device generator works as the EntiGraph generator was not measured here (SEAL and Active Reading suggest self-generation can work; see sections 4–5).

---

## 4. SEAL: Self-Adapting Language Models (Zweiger, Pari, et al. 2025)

### Takeaway
A 7B model can learn to write its own training data ("implications" of a passage). After 2 rounds of RL (ReST-EM), LoRA-finetuning on those self-edits raises no-context SQuAD accuracy from **32.7–33.5% to 47.0%**, beating GPT-4.1-generated synthetic data (46.3%). Training on the raw passage alone barely helps (+0.8 points). Under **sequential** self-edits SEAL forgets steadily; a 2026 follow-up measured only **1.30% retention** after 100 sequential passages. Each RL reward evaluation costs **30–45 s** (one finetune plus one evaluation).

### Cited Findings
- Model and setup: **Qwen2.5-7B**, SQuAD passages. Self-edits are "implications" of the passage used as SFT data with **LoRA** (single-passage) or **full finetuning** (CPT). **2 rounds of ReST-EM**, batches of **50 contexts × 5 sampled self-edits**. [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Single-passage no-context SQuAD accuracy: base **32.7%**; train on passage only **33.5%**; passage + base-model synthetic implications **39.7%**; passage + **GPT-4.1** implications **46.3%**; **SEAL 47.0%**. "Two iterations suffice for SEAL to overtake GPT-4.1 data; subsequent iterations yield diminishing returns." The policy converges to "distill[ing] the passage into easily learnable atomic facts". [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- CPT regime (n=200 and n=2067 passages, 5 self-edits per passage aggregated): SEAL reaches **58.2% at n=200**, above its single-passage score. GPT-4.1 data slightly outperforms SEAL in CPT. The exact n=2067 figures are in an image table and were not extracted. [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Prompting for longer self-edits helps, and RL adds a similar margin on top (Appendix B.11). [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Catastrophic forgetting: in a continual stream where each passage triggers a self-edit, "performance on earlier tasks gradually declines as the number of edits increases … still susceptible to catastrophic forgetting. Still, it can perform multiple updates without complete collapse." Suggested fixes: reward shaping to penalize regressions, null-space-constrained edits, representational superposition, or RL instead of SFT in the inner loop. [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Cost: "each self-edit evaluation takes approximately **30–45 seconds**". [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Few-shot ARC subset: success **72.5%** vs 20% (self-edits without RL) vs 0% (no adaptation), still below Oracle TTT. [P] — [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Independent measurement of forgetting (SCoL, 2026, Qwen2.5-7B-Instruct, stream of **100 SQuAD passages**): "SEAL retains only **1.30%** accuracy after sequential incorporation". SCoL has the model choose which ≤10 layers to update with LoRA, trained by meta-RL. Without a forgetting term (λ=0), retention is **14.64%**; with the forgetting reward, **20.46%**, and immediate accuracy also improves. The no-update prompting baseline is 28.17%. SCoL's chosen layers are sparse and line up with high-Fisher-information layers. [P] — [arXiv 2605.07076](https://arxiv.org/abs/2605.07076)
- "Language Models Need Sleep" (Behrouz & Mirrokni, Google, 2026) builds its "Dreaming" phase on SEAL with ReST-EM. It reports the best no-context SQuAD results in both the single-passage and CPT (n=200 passages, 974 questions) settings. Exact numbers were in an image table I could not extract. [P] — [arXiv 2606.03979](https://arxiv.org/abs/2606.03979)

### Inferences
- SEAL-style self-generated implications are the most direct published analogue of "told once, learn it". But the verified numbers show that (a) single-passage accuracy tops out around 47% on easy SQuAD facts, and (b) naive sequential application forgets almost everything (1.3% retention over 100 passages). Retention-aware rewards help (about 20% retention) but are far from "never forget". Some combination of replay, isolated parameters, and gating (sections 6 and 8) is needed.

### Gaps
- Exact SEAL forgetting-matrix values (Figure 6 heatmap) and CPT n=2067 numbers were not extractable as text. Energy and latency of SEAL-style updates on mobile hardware were not found.

---

## 5. Other knowledge-injection results (FT vs RAG, hallucination, acquisition dynamics, active reading/self-study, self-distillation)

### Takeaway
Across studies: (1) plain finetuning on documents injects little; (2) RAG beats it (Ovadia: about 0.88 vs 0.50 on new current events); (3) new facts are learned more slowly than known ones, and once learned they increase hallucination (Gekhman); (4) each exposure adds a small probability increment that then decays by a power law (Chang); (5) teaching the QA format first (PIT, +17.8%), generating diverse study material (Active Reading: 16% → 66% on SimpleWikiQA), or on-policy self-distillation (SDFT: 89% vs SFT 80% on new 2025 facts, with close to perfect out-of-distribution accuracy) greatly improves how much of the knowledge becomes usable.

### Cited Findings
**Fine-tuning vs RAG (Ovadia et al., EMNLP 2024)**
- "While unsupervised fine-tuning offers some improvement, RAG consistently outperforms it, both for existing knowledge encountered during training and entirely new knowledge … LLMs struggle to learn new factual information through unsupervised fine-tuning, and … exposing them to numerous variations of the same fact during training could alleviate this problem." [P, abstract] — [arXiv 2312.05934](https://arxiv.org/abs/2312.05934)
- Current-events (new knowledge) task: RAG **0.875 (Mistral) / 0.876 (Orca)** vs fine-tuning **0.504 / 0.511**. [S, from a social-media summary of the paper] — [X post summarizing 2312.05934](https://x.com/IntuitMachine/status/1750205032509681825); paper: [ACL Anthology 2024.emnlp-main.15](https://aclanthology.org/2024.emnlp-main.15/)

**New knowledge via fine-tuning and hallucination (Gekhman et al., EMNLP 2024)**
- Setup: closed-book QA on EntityQuestions (12 relations in-distribution, 7 held out), **PaLM 2-S** base, exact match. Fine-tuning sets are built with controlled proportions of examples "unknown" to the model. [P, setup section] — [arXiv 2405.05904](https://arxiv.org/abs/2405.05904)
- "LLMs struggle to acquire new factual knowledge through fine-tuning, as fine-tuning examples that introduce new knowledge are learned significantly slower than those consistent with the model's knowledge"; "as the examples with new knowledge are eventually learned, they linearly increase the model's tendency to hallucinate." [S, abstract via search] — [ACL Anthology 2024.emnlp-main.444](https://aclanthology.org/2024.emnlp-main.444/)

**Acquisition dynamics during pretraining (Chang et al., NeurIPS 2024)**
- Method: inject fictional knowledge into intermediate OLMo checkpoints (1B, 7B) and track per-step log-probability on 1,800 probes at three depths: memorization, semantic generalization (paraphrase), and compositional generalization. [P] — [arXiv 2406.11813](https://arxiv.org/abs/2406.11813)
- "Factual knowledge acquisition occurs by accumulating the small increase of probability induced by updating the model with a minibatch containing the factual knowledge"; later checkpoints show "no significant improvement in the ability to acquire" facts; "effectivity is greater in the 7B model than in the 1B model"; "a **power-law relationship** between training steps and forgetting of memorization and generalization"; "LLMs trained with **duplicated** training data exhibit **faster forgetting**"; "**larger batch sizes** can enhance the models' robustness to forgetting". This explains poor long-tail knowledge and the benefit of deduplication. [P] — [arXiv 2406.11813](https://arxiv.org/abs/2406.11813)

**Pre-instruction-tuning (Jiang et al. 2024, "Instruction-tuned LMs are better knowledge learners")**
- Llama-2 continually pretrained on new documents until perplexity reaches 1 answers only **27.6%** of questions about them, rising to **30.3%** after instruction tuning (the "perplexity curse"). Instruction-tuning on QA *before* training on documents (PIT) "outperform[s] standard instruction-tuning by **17.8%**". [P] — [arXiv 2402.12847](https://arxiv.org/abs/2402.12847)

**Active Reading (Lin et al., Meta FAIR, 2025; ICLR 2026)**
- The model proposes document-specific study strategies (paraphrasing, knowledge linking, active recall, analogical reasoning) and then applies them to produce self-generated training data. [P] — [arXiv 2508.09494](https://arxiv.org/abs/2508.09494)
- Llama 3.1 8B base, **20,000 steps**, about **4B words** generated per method, with **10% DCLM pretraining data mixed in "to prevent model degradation"**. SimpleWikiQA (3,449 Wikipedia-grounded SimpleQA questions): base **7.42%**; repeat **15.92%**; paraphrase **25.74%**; synthetic QA **47.87%**; Active Reading (task-agnostic) **63.33%**; headline **66%** (+313% relative over vanilla finetuning). FinanceBench: 26% (+160% relative). Paraphrase and synthetic QA plateau as tokens grow; Active Reading keeps improving up to 4B words. [P] — [arXiv 2508.09494](https://arxiv.org/abs/2508.09494)
- At scale: Meta **WikiExpert-8B** was trained on **1 trillion** Active-Reading tokens of Wikipedia. It outperforms DeepSeekV2 (236B) and Llama 3.1 405B on factual QA and is competitive with DeepSeekV3 (671B). [P] — [arXiv 2508.09494](https://arxiv.org/abs/2508.09494)
- "Self-generated data" beats "data generated by larger models". [S, paper-notes summary] — [papernotes ICLR2026](https://en.papernotes.org/ICLR2026/llm_pretraining/learning_facts_at_scale_with_active_reading/)

**Self-Distillation Fine-Tuning (Shenfeld, Damani, Hübotter, Agrawal, Jan 2026)**
- SDFT: the same model conditioned on the demonstration (or the source document) acts as teacher. The student (no context) is trained with reverse-KL on its own samples, which makes the update on-policy. [P] — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897)
- Knowledge acquisition: Wikipedia articles on **2025 natural disasters** (after the knowledge cutoff), about **200K tokens**, turned into a QA SFT set about **5× larger** than the corpus. On **Qwen2.5-7B-Instruct**: continued pretraining "performs poorly"; SFT reaches **80%** strict accuracy; **SDFT 89%**, "nearly clos[ing] the gap to the oracle RAG model". On out-of-distribution questions SDFT achieves "close to perfect accuracy, while SFT's performance remains low". It also shows less forgetting of prior capabilities, and 3 skills learned sequentially without regression. [P] — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897)
- Counterpoint (July 2026): SDPO, a dense on-policy self-distillation method, "exhibits stronger forgetting and can even collapse" in continual post-training, while GRPO "better preserve[s] prior capabilities"; "on-policy data alone is insufficient for continual learning." [S, abstract from HF fallback page] — [arXiv 2607.01763](https://arxiv.org/abs/2607.01763)

**Gated per-memory LoRA (MEGa, Pan, Hahami, Zhang, Sompolinsky 2025, CLS-inspired)**
- Each memory (a short event paragraph) gets its own LoRA adapter plus a stored embedding key. At inference, query-to-key similarity gates the adapters. Setup: Llama-3.1-8B-Instruct; 20 partitions × **50 sequential memories** (fictional character events, about 42 words each, **9 paraphrases** per sample during finetuning) and Wikipedia 2024 events (after the Dec-2023 cutoff). [P] — [arXiv 2504.21239](https://arxiv.org/abs/2504.21239)
- QA accuracy: MEGa **72.53%** (fictional) / **78.03%** (Wiki 2024) vs RAG **82.57% / 88.83%**. Continual-learning finetuning baselines "show severe catastrophic forgetting" (L2 regularization helps only a little). MEGa shows "only mild forgetting". The gate picks the correct memory 85.0% / 87.8% of the time. [P] — [arXiv 2504.21239](https://arxiv.org/abs/2504.21239)

**Hypernetwork-generated adapters (2026)**
- Scaling laws for hypernetwork-based "train-time knowledge injection" on MegaWikiQA (tens of millions of multi-hop QA from Wikidata5M): power-law scaling on all architecture axes and "steeper scaling exponents in all OOD evaluations" than LoRA or full fine-tuning. [P, abstract] — [arXiv 2607.19604](https://arxiv.org/abs/2607.19604)

### Inferences
- The strongest per-fact recipes so far generate many diverse, model-written study items: implications, QA, both directions, links to known entities. They train on them on-policy or with context distillation, mixing in about 10% general data. Raw-text finetuning is consistently weak.
- Gekhman's hallucination finding suggests facts the model does not know should be taught *with* their context/source cues and paired with "I don't know" or calibration data. Otherwise fitting unknown facts teaches the model to guess.
- Chang's power-law forgetting plus the Physics 3.3 exposure numbers support **spaced re-exposure** (replay) rather than one burst: each exposure's gain decays, so a fact must be revisited at growing intervals until it is consolidated.

### Gaps
- No study found that measures the minimum number of synthetic variants per *single* fact for durable (weeks-later) recall in a pretrained 1–8B model. Sparse-memory FT uses a batch of N paraphrases per fact (section 6) but does not ablate N. Primary Ovadia and Gekhman tables could not be read.

---

## 6. Forgetting: what causes it, and how much replay / constraint prevents it

### Takeaway
Forgetting tracks **how far the model's output distribution moves** (KL from the base model on the new task: R² = 0.96 on a toy task, 0.71 for LLMs). On-policy updates (RL, self-distillation) move it less. Tools that measurably reduce forgetting: **replay** (as little as 1% helps; 5% for mild data shifts, 25% for strong shifts in LLM continual pretraining; 30–60% used in production CPT), **LR re-warm/re-decay or infinite schedules**, **LoRA** (learns less, forgets less), **orthogonal subspaces** (O-LoRA), **freezing bottom layers** (spurious forgetting), and **sparse memory-slot updates** (NQ F1 drop of 11% vs 89% for full finetuning). Some apparent forgetting is lost task alignment, not lost knowledge, and 10 examples can restore it.

### Cited Findings
**RL's Razor (Shenfeld, Pari, Agrawal 2025)** [P] — [arXiv 2509.04259](https://arxiv.org/abs/2509.04259)
- "Even when SFT and RL achieve the same performance on the new task … SFT often achieves new-task gains by erasing prior knowledge, while RL better preserves old skills." "The degree of forgetting is determined by the distributional shift, measured as the KL-divergence between the fine-tuned and base policy evaluated on the new task."
- Forgetting vs KL: quadratic fit **R² = 0.96** (ParityMNIST toy) and **R² = 0.71** in the LLM experiments.
- LLM setup: Qwen 2.5 3B-Instruct trained on Open-Reasoner-Zero math, SciKnowEval Chemistry L-3, or ToolAlpaca. GRPO used a binary reward and *no* explicit KL penalty. Forgetting was measured on HellaSwag, TruthfulQA, MMLU, IFEval, Winogrande and HumanEval. Also robotics with OpenVLA-7B.
- An "oracle SFT" distribution that minimizes KL while being fully accurate forgets *even less than RL*. So the advantage comes from KL-minimality, not from RL as such.

**LoRA Learns Less and Forgets Less (Biderman et al., TMLR 2024)** [P] — [arXiv 2405.09673](https://arxiv.org/abs/2405.09673)
- Llama-2-7B on code and math; continued pretraining (about 20B tokens: StarCoder-Python, OpenWebMath) and instruction finetuning (about 100K pairs). "In the standard low-rank settings, LoRA substantially underperforms full finetuning. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain." LoRA "mitigates forgetting more than common regularization techniques such as weight decay and dropout". "Full finetuning learns perturbations with a rank that is **10-100× greater** than typical LoRA configurations." In CPT the gap is not closed even at high rank.

**O-LoRA (Wang et al., EMNLP Findings 2023)** [P] — [arXiv 2310.14152](https://arxiv.org/abs/2310.14152)
- Each task is learned in a new low-rank subspace kept orthogonal to earlier tasks' LoRA subspaces (earlier LoRAs are frozen). No replay data is stored, and task IDs are not needed at test time. The paper reports beating prior state of the art on continual-learning benchmarks and better preservation of generalization to unseen tasks. Exact benchmark numbers were not extracted.

**Spurious forgetting (Zheng et al., ICLR 2025)** [P] — [arXiv 2501.13453](https://arxiv.org/abs/2501.13453)
- "Performance drops often reflect a decline in task alignment rather than knowledge loss." Examples: LLaMA-2-7B-Chat safety drops 100% → 0% after 10 "identity shifting" examples and recovers to 99% after finetuning on 10 safety examples. Finance QA drops 75% → 0% after Science QA and recovers to 72% after training on *irrelevant* tasks.
- Most alignment damage happens in the **first ~150 optimization steps** of a new task, and the bottom layers matter most. **Freezing bottom layers** raises sequential-finetuning accuracy from **11% to 44%**; other continual-learning techniques (regularization, generative replay, merging, gradient-based) peak at 22%.

**Scale (Ramasesh, Lewkowycz, Dyer, ICLR 2022)** [S]
- "Large, pretrained ResNets and Transformers are significantly more resistant to forgetting than randomly-initialized, trained-from-scratch models, with robustness systematically improving with scale of both model and pretraining dataset size." — [Google Research page](https://research.google/pubs/effect-of-scale-on-catastrophic-forgetting-in-neural-networks/)

**Replay ratios and LR schedules for LLM continual pretraining (Ibrahim et al., TMLR 2024)** [P] — [arXiv 2403.08763](https://arxiv.org/abs/2403.08763)
- 405M and 10B decoder-only models, new datasets of 100B+ tokens each; weak shift (Pile→SlimPajama) and strong shift (English→German). "LR re-warming, LR re-decaying, and replay of previous data is sufficient to match the performance of fully re-training from scratch on all available data."
- Replay sweep: 1%, 5%, 10%, 50% (plus 0.5% for the weak shift and 25% for the strong shift). "Even the lowest tested replay of **1%** significantly reduces forgetting." 1/5/10% have little impact on downstream performance. **50%** replay adapts noticeably worse to the new data. Chosen defaults: **5% replay (weak shift), 25% replay (strong shift)**. Related work cited in the paper found replay "as little as 1%" sufficient in LLM finetuning (Scialom et al. 2022).
- Also: re-warming the LR is needed to adapt but itself causes forgetting. Lowering the max LR reduces forgetting. "Infinite" LR schedules (constant plateau plus final decay) avoid re-warming.
- Production usage cited in the paper: Glorioso et al. 2024 (Zamba) used LR re-warm/re-decay with **60% replay** over a 50B-token decay phase. DeepSeek-AI 2024 used **30% replay** of pretraining data to continually pretrain DeepSeek-V2 on 6T tokens (a large gain in code, with most natural-language ability retained).

**Time-continual web-scale pretraining (TiC-LM, Apple, ACL 2025)** [S]
- Built from 114 Common Crawl dumps. "Autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6×)". Replay is "crucial to avoid forgetting on generic web data but less so on specific domains". — [arXiv 2504.02107](https://arxiv.org/abs/2504.02107); [Apple ML Research](https://machinelearning.apple.com/research/tic-lm-web-scale)

**Mixture/dose laws** [P, abstracts]
- CMR scaling law: a power law links loss, general:domain mixture ratio, and tokens, giving a predictable "Critical Mixture Ratio" for continual pretraining with replay. — [arXiv 2407.17467](https://arxiv.org/abs/2407.17467)
- Knowledge infusion scaling law: injecting too much domain knowledge causes "memory collapse". Each model has a **critical collapse point** beyond which knowledge retention "sharply degrade[s]", and these points scale with model size, so they can be predicted from smaller models. — [arXiv 2509.19371](https://arxiv.org/abs/2509.19371)

**Sparse memory finetuning (Lin et al., Meta FAIR, Oct 2025)** [P] — [arXiv 2510.15103](https://arxiv.org/abs/2510.15103)
- A 1.3B model with a memory layer replacing the FFN at layer 12 of 22: pool of **1M slots**, k=32 accesses per token, 4 heads, value dim 1024, so **32,768 active parameters** vs 50M in the original FFN. Only the **top t=500** memory slots ranked by TF-IDF (accessed on this batch vs 1000 background DCLM batches) are updated; everything else is frozen.
- Learning 1000 TriviaQA facts one at a time (each fact paraphrased N times to fill a batch of 64): "NaturalQuestions F1 drops by **89%** after full finetuning on new facts and **71% with LoRA**, sparse memory finetuning yields only an **11%** drop with the same level of new knowledge acquisition." The same pattern holds when learning from a stream of Active-Reading-augmented documents (100 SimpleQA questions). Naive full finetuning of the memory layer still forgets. LoRA sweep: rank {32, 128, 256}; full-FT LR sweep {2e-6 … 5e-5}.
- Motivation cited: "as little as 0.01% of model parameters are responsible for model performance on a particular task" (grafting).

**Chang et al. 2024 (pretraining)**: forgetting of injected facts follows a **power law** in steps; **deduplication** and **larger batch** slow forgetting. [P] — [arXiv 2406.11813](https://arxiv.org/abs/2406.11813)

### Inferences
- For an on-device learner the best-supported anti-forgetting stack is: (1) keep updates **on-policy / KL-small** (self-distillation from the context-conditioned model; RL-style rewards); (2) put new facts in **sparse, isolated capacity** (memory slots chosen by TF-IDF, per-memory gated LoRA, orthogonal LoRA) rather than dense shared weights; (3) **replay** a small stream of old material (1–10%, more if the new data is far from the old distribution); (4) monitor KL on a fixed probe set as a cheap forgetting alarm, using the verified KL-forgetting correlation.
- The numbers suggest dense updates of any kind (full FT, LoRA) are the dominant forgetting risk for per-fact learning. Sparse memory FT's 11% vs 71–89% drop is the largest measured reduction in the reviewed literature for a stream of single facts.

### Gaps
- No verified per-fact numbers for O-LoRA or EWC on fact streams. No verified long-horizon (thousands to millions of facts) sparse-memory-FT results; a follow-up "Improving Sparse Memory Finetuning" exists ([arXiv 2604.05248](https://arxiv.org/abs/2604.05248)) but was not read. Ramasesh et al. exact effect sizes were not retrieved (secondary summary only).

---

## 7. Continual knowledge updating benchmarks with news/temporal data: parametric update vs retrieval

### Takeaway
Across benchmarks from 2022 to 2026, retrieval absorbs new facts fastest and most reliably. Parametric updates can work (TemporalWiki: training only on monthly diffs matches full-snapshot training at 12× less compute) but struggle to *overwrite* outdated facts (EvolvingQA). Parameter expansion is needed to retain and learn at once (CKL). The best parametric numbers on genuinely new 2024–2025 facts now approach RAG: SDFT 89% vs SFT 80% (RAG oracle slightly higher); MEGa 72.5–78% vs RAG 82.6–88.8%; EntiGraph 56.2% vs RAG 60.4%.

### Cited Findings
- **StreamingQA** (Liška et al., ICML 2022): questions asked on a given date and answered from **14 years of time-stamped news**, evaluated quarterly as models read new articles. "Parametric models can be updated without full retraining, while avoiding catastrophic forgetting." For semi-parametric models, "adding new articles into the search space allows for rapid adaptation, however, models with an outdated underlying LM under-perform those with a retrained LM." Parametric updates are especially beneficial for questions about higher-frequency named entities. [S] — [PMLR v162 liska22a](https://proceedings.mlr.press/v162/liska22a.html)
- **TemporalWiki** (Jang et al., EMNLP 2022): consecutive English Wikipedia/Wikidata snapshots. "Training a language model on the diff data through continual learning methods achieves similar or better perplexity than on the entire snapshot with **12 times less computational cost**". [S] — [ACL Anthology 2022.emnlp-main.418](https://aclanthology.org/2022.emnlp-main.418/)
- **CKL** (Jang et al., ICLR 2022): InvariantLAMA (retain), UpdatedLAMA (update), NewLAMA (acquire), and the FUAR metric (Forgotten / (Updated + Acquired)). T5 and GPT-2 continually pretrained on new corpora. "CKL exhibits unique challenges … where **parameter expansion is necessary** to reliably retain and learn knowledge simultaneously." [P] — [arXiv 2110.03215](https://arxiv.org/abs/2110.03215)
- **EvolvingQA** ("Carpe Diem", NAACL 2024): evolving Wikipedia with UNCHANGED/NEW/EDITED sets. Existing continual-learning baselines "suffer from updating and removing outdated knowledge"; "models fail to rectify knowledge due to small weight gradients"; numerical and temporal changes are especially hard. [S] — [ACL Anthology 2024.naacl-long.302](https://aclanthology.org/2024.naacl-long.302/)
- **TiC-LM** (2025): 114 CC dumps; replay plus meta-schedules match re-training at 2.6× less compute (see section 6). [S] — [arXiv 2504.02107](https://arxiv.org/abs/2504.02107)
- **Post-cutoff news-like facts into weights, 2024–2026 (primary numbers):**
  - SDFT, 2025 natural-disaster Wikipedia (about 200K tokens), Qwen2.5-7B-Instruct: SFT 80% → SDFT 89% strict accuracy, "nearly clos[ing] the gap to the oracle RAG model"; CPT on raw text "performs poorly". [P] — [arXiv 2601.19897](https://arxiv.org/abs/2601.19897)
  - MEGa, Wikipedia 2024 events, Llama-3.1-8B-Instruct: 78.03% QA vs RAG 88.83%. [P] — [arXiv 2504.21239](https://arxiv.org/abs/2504.21239)
  - EntiGraph (book corpus, not news): 56.22% closed-book vs 60.35% RAG. [P] — [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
  - Ovadia current-events: RAG about 0.875 vs FT about 0.50. [S] — [X summary](https://x.com/IntuitMachine/status/1750205032509681825)
  - SCoL, streaming SQuAD (100 passages): best retention after the full stream is 20.46% (vs 28.17% for prompting with no update). Sequential parametric consolidation is still weak for long streams. [P] — [arXiv 2605.07076](https://arxiv.org/abs/2605.07076)

### Inferences
- A fast episodic store (retrieval over raw pages and facts) is needed as the immediate, reliable path. Parametric consolidation can then catch up in the background. This matches CLS theory (section 8) and the StreamingQA/EntiGraph finding that the two approaches add together.
- Overwriting outdated facts (EvolvingQA's "EDITED") is a harder, separate problem from adding new ones. A design that "never loses older knowledge" also needs a way to supersede stale facts.

### Gaps
- Exact StreamingQA, TemporalWiki, EvolvingQA and CKL numbers (F1/EM tables) could not be read (arXiv blocked; HF mirror lacked these papers). I found no 2025–2026 paper that reports a daily or weekly "news → weights" system over months with measured long-term retention.

---

## 8. Complementary learning systems and sleep/replay consolidation applied to LLMs

### Takeaway
CLS theory holds that a fast hippocampal store records specifics and a slow neocortical store learns regularities through interleaved replay. Replay avoids catastrophic interference. Schema-consistent information can be integrated quickly, and replaying only *similar* old items (SWIL) speeds integration. LLM work in 2025–2026 now implements this directly: gated per-memory adapters (MEGa), learned sparse update locations with retention rewards (SCoL), and explicit "sleep" phases that distill fast memory into slower or larger parameters and "dream" synthetic rehearsal data (Behrouz & Mirrokni 2026). Measured gains are real but small-scale, mostly SQuAD-sized.

### Cited Findings
- Kumaran, Hassabis, McClelland (Trends in Cognitive Sciences, 2016), "What Learning Systems do Intelligent Agents Need? Complementary Learning Systems Theory Updated": fast hippocampal and slow neocortical systems; interleaved learning through replay to avoid catastrophic interference. [S] — [Semantic Scholar](https://www.semanticscholar.org/paper/What-Learning-Systems-do-Intelligent-Agents-Need-Kumaran-Hassabis/84dccdefebec40941af9d0cd4d415d7f2e5f1959)
- McClelland, McNaughton, Lampinen (Phil. Trans. R. Soc. B, 2020): "information consistent with prior knowledge can be integrated very quickly". Integrating new items that are *not* schema-consistent needs gradual interleaving with old items. **Similarity-weighted interleaved learning (SWIL)** focuses replay on previously known items in the same branch of the knowledge hierarchy, "resulting in faster integration with less interleaving overall". Projection onto known dimensions can be learned quickly without interleaving; new dimensions need gradual interleaved learning. [S] — [Royal Society](https://royalsocietypublishing.org/rstb/article/375/1799/20190637/23829/Integration-of-new-information-in-memory-new); [author PDF](https://web.stanford.edu/~jlmcc/papers/McCMcNaughtonLampinen20IntegrNewInfoCLS.pdf); follow-up in deep nets: [PNAS 2022 SWIL](https://pmc.ncbi.nlm.nih.gov/articles/PMC9271163/)
- **MEGa** (2025) is explicitly CLS-inspired: fast per-memory gated LoRA plus embedding-key gating. Mild forgetting vs "severe" forgetting for continual-learning finetuning baselines over 50 sequential memories; QA 72.5–78.0% vs RAG 82.6–88.8% (section 5). [P] — [arXiv 2504.21239](https://arxiv.org/abs/2504.21239)
- **"Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"** (Behrouz & Mirrokni, Google; on OpenReview since Sept 2025, arXiv June 2026). Sleep has two stages. (1) **Memory consolidation / "Knowledge Seeding"**: upward distillation from a smaller self (fast, high-frequency modules) into newly unlocked parameters in slower, low-frequency modules, via "Generalized Distillation" (on-policy distillation plus RL-based imitation), with periodic parameter (de)activation "while ensuring the stability of old parameters". (2) **Dreaming**: RL-generated synthetic curricula (built on SEAL with ReST-EM) "to rehearse new knowledge and refine existing capabilities". Evaluated on factual knowledge incorporation (SQuAD single-passage and CPT n=200 / 974 questions: best among compared methods including SEAL; exact numbers only in a table image), few-shot ARC (Llama-3.2-1B, **80% success**, above the other methods), long context, and continual learning. The authors argue their design "is more robust to catastrophic forgetting" than ICL-only or SEAL-style sequential updates. [P] — [arXiv 2606.03979](https://arxiv.org/abs/2606.03979)
- **"Do Language Models Need Sleep? Offline Recurrence for Improved Online Inference"** (May 2026): during sleep, the model makes N offline recurrent passes over accumulated context and writes into SSM fast weights with a learned local rule, keeping wake-time latency unchanged. Longer sleep (larger N) improves performance, most on examples needing deeper reasoning. Tested on cellular automata, multi-hop graph retrieval, and math reasoning. This is consolidation into *fast weights*, not permanent weights. [P, abstract] — [arXiv 2605.26099](https://arxiv.org/abs/2605.26099)
- **SCoL** (2026): meta-RL teaches the model where to write (≤10 layers) with reward = acquisition − λ·forgetting. Retention over 100 SQuAD passages: 1.30% (SEAL) → 14.64% (λ=0) → **20.46%** (with retention reward). Selections line up with high-Fisher layers. It also transfers from shorter meta-training streams to longer LongBench v2 streams. [P] — [arXiv 2605.07076](https://arxiv.org/abs/2605.07076)
- Brain-analogue design evidence from standard LLM practice: small replay fractions (1–25%) prevent most forgetting in continual pretraining (Ibrahim et al., section 6), and Active Reading and EntiGraph both interleave 10% or more general data during injection. This is a coarse, un-targeted form of CLS interleaving. [P] — [arXiv 2403.08763](https://arxiv.org/abs/2403.08763); [arXiv 2508.09494](https://arxiv.org/abs/2508.09494); [arXiv 2409.07431](https://arxiv.org/abs/2409.07431)
- Other 2025–2026 items found but **not read** (titles only, from search): "SCM: Sleep-Consolidated Memory with Algorithmic Forgetting for Large Language Models" ([arXiv 2604.20943](https://arxiv.org/abs/2604.20943)); "Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning" ([arXiv 2504.14727](https://arxiv.org/abs/2504.14727)); "Wake-Sleep Consolidated Learning" ([arXiv 2401.08623](https://arxiv.org/abs/2401.08623)); "Beyond Inference-Only Deployment: Comparing Weight-Based Consolidation Against Cascading Compaction" ([arXiv 2605.24657](https://arxiv.org/abs/2605.24657)); "Position: Modular Memory is the Key to Continual Learning Agents" ([arXiv 2603.01761](https://arxiv.org/abs/2603.01761)); "A Hippocampus for Linear Attention (HOLA)" ([arXiv 2607.02303](https://arxiv.org/abs/2607.02303)).

### Inferences
- A consolidation schedule consistent with the evidence:
  - **Wake**: store the raw page/fact verbatim in an episodic store (retrieval gives immediate, near-100% recall), and optionally write a per-memory gated adapter or sparse memory slots (MEGa / sparse memory FT).
  - **Sleep** (charging, idle): (a) generate diverse self-edits for each new episode: implications, QA, reverse direction, links to known entities (SEAL / Active Reading / reverse training); (b) train on-policy or by self-distillation from the context-conditioned model (SDFT, KL-small, RL's Razor), writing into sparse or new capacity; (c) interleave replay of old material, weighted toward semantically similar old facts (SWIL), plus a small general-data stream (1–10%); (d) check retention on a held-out probe set and use KL as a forgetting alarm.
  - Repeat each episode's replay over several nights on a spacing schedule, because single-exposure gains decay by a power law (Chang) and about 100–1000 exposures are needed for full-capacity storage (Physics 3.3).
- This combined schedule is my synthesis. No paper reviewed has tested it end to end, and the published CLS-for-LLM results are all small-scale (50–200 passages or memories).

### Gaps
- No LLM paper found that measures long-term (weeks/months, thousands+ of facts) retention under a sleep/replay schedule. No quantitative guidance was found for replay frequency *per fact* (e.g., number of nights, interval growth) in LLMs. Human spacing literature was out of scope. Exact results tables for "LMs Need Sleep", SCM, and semi-parametric consolidation were not extracted. Tadros et al. 2022 (sleep-like replay in ANNs) was not verified.
