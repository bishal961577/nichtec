# Round 2 — ML Architect

## 1. Agree / disagree
- Founder's 9.7%: number right, reading wrong. 0.5B question form; cloze form 88.5%; GPT-J pasted 84.0% question form. = small model failing to read a pasted fact into a question, not belief refusal. (Moderator verified: FIXING_THE_BOTTLENECKS.md:602, 633, 822.) Belief conflict shows elsewhere: GPT-J chain old answer 19.5%, 19.3% with pasting, on sub-questions GPT-J wrote itself. Neuro's "encoding specificity" = my "lookup miss"; neuro's fix cheapest: store each fact under the model's own sub-question wordings, log every step.
- Thinking off confirmed: confounds (b), (e), and (c). (b), (e) unproven until rerun.
- Researcher #1 richer bridge composes: sceptical. Grokked Transformers (Wang 2024): every entity had many facts, yet composition on entities never seen in compositional examples stayed near chance. Compositional exposure mattered, not facts per entity. Pass ~25%. Run it as one arm alongside neuro co-replay → "how many facts" vs "seen in compositional contexts".
- Neuro "sum of per-fact terms = filing cabinet": half right. Joint ridge has cross terms, but they resolve interference, never produce a derived fact. Derivation needs nonlinear computation, live (reasoning over retrieved facts) or offline (sleep). Order-freedom + integration reconciled by researcher #3: every derived product a deterministic function of the stored *set*, rebuilt from journal; delete = rebuild without.
- Prediction for F-B: derivations easy forward from a new object, hard backward ("what pointed to Apple?" → reversal curse). Derived pages should help when the edit is the first hop, little when a later hop. Split by which hop was edited.

## 2. Answers to founder
- Q1: yes for override (no old belief fights personal facts). But "my sister → Ana → Ana's job" is the hard two-hop case with both facts new: easier to override, just as hard to compose without steps; with steps + retrieval should be easy.
- Q2: predict fine for direct recall with shared words; falls off with no shared words or two steps (NoLiMa). Cloud with cached prompt viable; phone 50–100K tokens per turn not.
- Q3: for v1 weights give users nothing the notebook can't. One real argument: **unprompted relevance** — notebook acts only when queried; integrated knowledge shapes every answer. Founder's unprompted-recall gap is the strongest case for integration; nobody measures it.
- Q4: right — latent composition not a product requirement. Downgrade my P2 from app need to science bet.
- Q8: yes, hidden assumption and cause of interference. Facts → exact notebook; preferences → population map + personal coordinates; episodes → journal; skills → slow distillation. Forcing all into one weight update makes them interfere.

## 3. Merged ranking
| # | Merged | Breakthrough | App | Cost | Clean |
|---|---|---|---|---|---|
| M1 | Re-measure at ship sizes: F-A sweep (personal vs conflicting, 10→5,000) + researcher control (thinking on, 27B use ratio) + store under own wordings + step logging | ~0 | Very high | 1–2 H100 h | High |
| M2 | Population taste map + exact personal coords + LLM places new items (Arch P1, F-C, Neuro P3) | ~0 | High | <1 h | High |
| M3 | What makes a new bridge composable in weights (Res #1 richness + Neuro P1 co-replay + D→B transfer) | Low–med | Med | 3–6 GPU h | Med–high |
| M4 | Nightly consolidation with sources: derived pages (F-B), then rebuildable LoRA (Res #3) if M3 passes | Low | High | ~$10–30 | Med |
| M5 | Arch P2 multi-depth exact memory | Highest, still low | Low near term | 1–2 wk | Med (fail ambiguous) |
| M6 | Doc-to-LoRA accumulated over time (Res #2) | Low | Med | ~$20 | High |
| M7 | Surprise-gated restart (Neuro P2), only if contradictions persist with thinking on | Low | Low | 2–4 h | Med |
Gap nobody filled: learning from corrections, and unprompted recall. A small simulated-conversation benchmark worth more to the app than M5–M7.

## 4. First / biggest bet
(a) M1: 4B and 27B, thinking on, 10/100/1,000/5,000 facts in prompt, fictional personal vs conflicting world edits, single- and two-hop, with/without shared words. Pass: 27B ≥ 85% single-hop and two-hop use ratio ≥ 0.8 at 5,000 personal facts → v1 = notebook + retrieval/long context, the bar every memory method must beat. Fail: shows where it breaks (count, no shared words, belief conflict).
(b) Still P2, revised: add identity ("zero-hop") training task (Identity Bridge, confirmed); run *after* M3. If co-replay builds composable entities in weights, P2 matters less; if nothing works in weights, P2 is the only remaining route to integration that keeps exact deletion. <3%.
Changed mind: P1 drops from decisive diagnostic to cheap app engineering (CF already answered (a); textbook).
Citations: Grokked Transformers, NoLiMa, MemN2N from memory (fairly confident); MesaNet, AlphaRec, Identity Bridge confirmed by search.
