# Round 2 — CS/AI Research Scientist

## 1. Corrections
- Baselines: **I was wrong.** MEMIT/MEND/MeLLo from MQuAKE v3 Table 5 on CF-3k-v2 = same file. Retracted. Still: MeLLo not the strongest (CARE, ACL Findings 2026); GPT-J 23.4% ceiling limits what the run can show.
- Doc-to-Atom real (2606.12400, Diao et al., 10 Jun 2026) but claim too broad: micro-LoRA + provenance key + router overlaps "keyed values + router", but abstract silent on sequential writes, deletion, order-independence, multi-hop. Team novelty survives if stated as exact joint re-solve, order-freedom, exact deletion, capacity certificate, 2,764 lifelong edits.
- Hindsight paper real, weaker: NeurIPS 2026 SLM-Agents *workshop* poster (I-Ju Lin, Zhang-Wei Hong), not main track. Still relevant.
- O'Neill, Augenstein, Learning on the Job abstracts fetched in round 1, check out.

## 2. Novelty/feasibility of others' proposals
- Arch P2: **partly open.** KBLaM (2410.10450) already lets every attention layer read knowledge tokens from an external store (frozen LLM, adapters trained on synthetic KBs, >10K triples). "Read at several depths" done. Open: pointer-chain values and latent two-hop with both facts new. ~$5 check first: released KBLaM on two-hop with both facts in KB (knowledge tokens can't attend to each other → expect fail). Fail → P2 clean extension; succeed → novelty shrinks. Real risk: OOD relation pairs (Grokked).
- Neuro P1: **open**, same family as my #1. Missing prior art: Feng, Russell, Steinhardt (ICML 2025, 2412.04614) — fine-tuned fact generalizes as first hop if early layers trained, as second hop if late layers; fails when implications come before facts. Transfer test must vary hop position and order. Prediction: D→B transfers poorly unless D also seen in compositions.
- F-B: parts done (Generative Agents reflections cite evidence; sleep-time compute 2504.13171; TMS Doyle 1979), combination not. Product engineering, little research novelty. On MQuAKE forward chaining covers only first-hop edits; hop-2 edits need reverse lookup (reversal curse). Over the notebook's own facts reverse lookup is easy.
- Neuro P2: mostly known. Olausson et al. ICLR 2024: at equal compute self-repair seldom beats fresh sampling → predicts always-restart ≈ gated. Low priority, only with thinking on.

## 3. Founder answers
- Q1 partly yes: 9.7% and 19.5% = knowledge conflict (Override Gap). Personal facts avoid it. But two-hop curse shown on fictional conflict-free facts → latent composition stays hard. Personal facts bring their own conflicts: updates ("Ana moved"), ambiguous names.
- Q2: no published 27B-class number on 3–10K personal facts. LongMemEval-S (~115K tokens): long-context LLMs drop 30–60% relative vs evidence only (2410.10813). NoLiMa: at 32K, 10/12 models < 50% of short-context score without lexical overlap; GPT-4o 99.3→69.7 (2502.05167). Expect shared-word lookups hold; rewording, combination, latest-value updates degrade. Must measure.
- Q6: within-user AUC over chosen movies mostly measures quality; companion decides what to suggest; selection bias (Schnabel 2016). Keep stated preferences (notebook) separate from revealed taste (population model).
- Q7: measured. Learning on the Job 2.6× from corrections; CIPHER/PRELUDE (2404.15269) learns from user edits; PrefEval (ICLR 2025) LLMs follow stated preferences poorly over long conversations.
- Q10: 4B int4 runs on a phone, 100K KV cache and prefill don't. Split: small exact per-person layer on phone, model in confidential cloud (Private Cloud Compute-style).

## 4. Merged ranking (EV/$)
1. **FIRST: notebook-reading baseline at ship size** (founder A + arch rerun + my control). 4B & 27B, thinking on/off; 10/100/1K/5K; fictional personal vs MQuAKE conflicting; worded-like-fact vs reworded; two-hop; updates; top-k page-selection arm. 1–2 H100 h, $0. Pass: 27B ≥ 85% reworded single-hop at 5K, two-hop with thinking ≥ 0.8× own-fact rate → v1 = notebook + retrieval.
2. Population lens (arch P1 + neuro P3 + founder C). < 1 H100 h. Heavy prior art (AlphaRec, DropoutNet); settles taste architecture.
3. Data structure for latent composition (neuro P1 + my #1): arms separate docs; rich B profile M ∈ {0,4,16,64}; co-replay; D→B transfer; × first-hop vs second-hop position (Feng). 3–6 A100 h.
4. Highest ceiling: arch P2, gated by KBLaM check. Only proposal that could give exact, deletable memory that also composes latently. ~20% clear win synthetic, 5–10% transfers to real.
5. F-B: build as product, measure retraction correctness.
6. Neuro P2, then Doc-to-LoRA accumulation (tied to Gemma-2-2B, off app path).

## 5. Odds
Breakthrough ~3%. Solid actionable result 65–75% (up; #1 and #2 cheap, near-certain to settle app architecture — engineering answers). Novel publishable (#3 or #4) 25–35%.
