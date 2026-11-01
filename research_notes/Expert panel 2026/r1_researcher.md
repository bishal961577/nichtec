# Round 1 — CS/AI Research Scientist

## 1. Frontier (Nov 2026) and what the team hasn't reviewed
Nobody has a write operator that turns a few exposures into knowledge the model uses like pretraining knowledge. Storage and order-freedom largely solved. Recurring finding: what decides whether a fact gets used is the **variety of data it was written with**, not the update method.
- **Weight writes.** O'Neill et al. 2026 (2607.11020), Qwen3: after 20 sequential writes, earlier facts retained 1% with repeated statements vs 46% with 24 varied restatements; data in later writes sets retention whatever the method (SFT, off/on-policy distillation; LoRA/full FT). "Forgotten" facts still visible in logprobs. Sparse memory FT loses 11% on NQ vs 89% full FT (2510.15103); SDFT forgets less than SFT (2601.19897).
- **Amortized writers.** Doc-to-Atom (2606.12400) compiles each knowledge unit into a micro-LoRA with a provenance key; a router assembles per query — very close to the team's keyed memory, overlaps its novelty claim. Augenstein et al. (Google, Sept 2026, 2609.24979) per-user LoRAs on device. Doc-to-LoRA's released hypernetwork targets Gemma-2-2b-it; long inputs chunked and LoRAs concatenated, but adapters from separate documents at different times never combined → accumulation over time untested.
- **Frozen weights + memory.** "Learning on the Job" (2607.22157): deployment feedback → natural-language rules; 1.6× single-trial success from verdicts, 2.6× from corrections on τ-bench banking; memory transfers across models. Cheap baseline the app must beat. Evo-Memory (2511.20857) benchmarks experience reuse.
- **On device.** "Continual On-Device Personalization of SLMs from Hindsight" (NeurIPS 2026): self-distillation from later user activity, on a real phone, beats ICL, RAG, reward learning, rejection FT. = the companion idea in miniature.
- **Multi-hop editing.** Best: retrieval + critic with backtracking (CARE, ACL Findings 2026), circuit-aware editing (CaKE). MQuAKE-Remastered (ICLR 2025) already published the 33% contamination.
- **Architectures.** TTT-E2E (2512.23675), Nested Learning/Hope: LM and long-context gains, no new-knowledge-in-reasoning shown.
- **Latent composition.** Balesni (2411.16353): two-hop over two fine-tuned facts fails without CoT even with 30 paraphrases per fact; succeeds if one fact is pretrained or facts share a document.

## 2. Team conclusions
Solid: (a) right, but textbook CF/cold-start (Kang 2023 2305.06474: zero-shot LLMs lag CF) — frame as careful replication. (c) holds for *latent* use. (d) true.
Overstated/confounded:
- **Thinking was off in every Qwen3.5 run.** Qwen3.5 are hybrid reasoning models. So (b) and (e) were measured on single non-thinking samples at temperature 0.8. Evidence supports "unverified non-thinking guesses contradict evidence; external checking fixes it" (Huang 2310.01798 already showed). Re-run the 37/96% contradiction metric with thinking on before keeping (b).
- **(c) for explicit chains rests on GPT-J** (own multi-hop 23.4%). Comparison numbers (MEMIT 5.4, MEND 6.1, MeLLo 14.2) are from the 2023 paper on the *original contaminated* data; team ran v2. Not the same data. MeLLo no longer strongest.
- **Test 6 and context-vs-memory used 0.5B/1.5B.** Bridge-entity readability improves with size (Yang 2402.16837). 43→13% is partly weak-reader.
- **Keyed memory = name-span matching + relation classifier on a triplet benchmark** = structured retrieval. 96/98.7% depends on MQuAKE's clean names. Free-text personal memory (nicknames, pronouns, no schema) untested.
- Credit: pre-registration, leakage-safe popularity, label audit, reporting own wrong predictions.

## 3. Open directions (EV per $)
**1. Is the two-hop curse caused by how few facts the bridge entity has? (~$15, ~2 H100 h)** Hypothesis: paraphrase doesn't help (Balesni), but giving the new *bridge* entity a rich profile of M other facts makes it compose like a pretrained entity (fine-tuning analogue of Karmim's exposure bound). No M dose-response found on pretrained LLMs (check Karmim 2026, 2506.13479). Qwen3.5-4B LoRA, 200 fictional people; A→B (new bridge) and B→C in strictly separate docs, C never with A; M ∈ {0,4,16,64} extra facts about B; MC logprobs, no CoT; normalize by native no-CoT two-hop on real facts of matched popularity. Pass: M=64 ratio ≥ 0.5, ≤ 1 pt loss on held-out native QA. Stop: ratio < 0.15 at every M → latent integration via weights structurally blocked at this scale; commit to explicit chains.
**2. Does instant internalization survive time? (~$20)** Released Doc-to-LoRA on Gemma-2-2b; 200 synthetic life-log docs over simulated days; single-doc and cross-doc (day 1 ∘ day 30) questions; full context vs top-k RAG vs merged vs routed adapters. Pass: at 200 docs single ≥ 70% of context, cross-doc with CoT ≥ 50%. Stop: single < 50% at 50 docs.
**3. Nightly exactly-rebuildable consolidation (conditional on #1, ~$30).** Retrain LoRA from scratch nightly (fixed seed, sorted order) on varied restatements generated from the exact memory → weights a deterministic function of memory contents: order-free, exact deletion by retraining without. Memory = source of truth, checks the student. Pass: student no-CoT use ratio ≥ +0.15 over memory alone, ≤ 1 pt forgetting. Novelty low (SISA, O'Neill, SDFT, Cartridges); only design here that keeps accountability while adding integration.
**Must-do control first:** re-run the GPT-J use ratio on Qwen3.5-27B with thinking on; re-run a MeLLo/CARE-style baseline on v2.

## 4. Odds
Breakthrough 3–5%. Solid pre-registered publishable result 45–60%. Biggest risk: the app reproduces hindsight-SDFT and "Learning on the Job" without beating them.
Cited from memory, unchecked: Kang 2023, Huang 2024, Yang 2024, MQuAKE-Remastered.
