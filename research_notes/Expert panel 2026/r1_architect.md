# Round 1 — ML Architecture Expert

## 1. Diagnosis
Central fact: **a transformer reads only one kind of new information in an integrated way — its context window.** ICL works because training taught every layer to read that channel. Post-deployment information can go three places; each gives something up:
- **Context/retrieval**: integrated, composes with reasoning; not persistent, dilutes (43/27/13% at 10/100/1,000 pasted facts; CL-Bench memory systems worse than full history).
- **Bolt-on memory** (exact memory, GRACE, Engram tables): persistent, can be exact; no layer was trained to read it → addressable only by surface tokens (test 6), usable only where reasoning is written out (two-hop curse).
- **Weight updates** (FT, self-distillation, hypernetworks): integrated and persistent; don't commute, need backward pass, drift (MEND, MEMIT at 2–4K edits).
Integrated / persistent / cheap-and-safe: pick two. Cause: **memory integrates only if the network was trained with that memory in the loop** (Memory Layers, RETRO, Titans, MesaNet were pretrained with it). Frozen model + add-on memory is limited to the text interface.

Team conclusions:
- (a) representation in advance — **strongly agree**, the most important finding. CF is the template: slow representation from everyone + exact per-person solve (that ridge solve is SPARC working well: order-free, deletable, certifiable). Untested: does the LLM lack the representation, or can its verbal channel just not do statistics over 80 ratings? Proposal 1 decides.
- SPARC order-freedom holds **only for the fast layer**. Learning features can't be order-free in general (test 4 confirms). Right design: slow non-order-free representation under a fast exact layer. Stop looking for order-free feature learning.
- (b) can't check own conclusions — partly agree; forward pass generates, doesn't score. Code-as-judge was right; real-data failure was wrong hypothesis space (= (a)).
- (c) filing cabinet vs mind — **partly disagree.** With CoT, integration is largely solved: CODE 83.5% two-hop on 7–8B; retrieval into context works with competent models. 0.40 use ratio is held down by GPT-J's weak chain (23.4% ceiling) and lookup name collisions. Rerun on Qwen3.5-4B with per-step logging first. Genuinely unsolved: latent composition through a new middle entity — in a fixed-depth net facts sit at one depth while the bridge entity is computed later → Proposal 2.
- (d) skills — agree unaddressed, no small-team architectural opening; best is verified procedure libraries (Voyager, AWM, ReasoningBank, ACE); only enter where an automatic checker exists.
- (e) weak lesson extractors — true, so don't ask them to extract in words; read hidden states with a linear model.

## 2. Proposals
**P1 — Is the representation missing, or just unreadable? (cheap, decisive)**
Embed each MovieLens title with frozen Qwen3.5-4B/27B hidden states (mean-pooled middle layer).
- A (no other people): per person ridge/logistic on top PCA components, popularity as offset, fit on own 80 ratings.
- B (population lens): linear map hidden→32 dims fit on the other ~5,700 users; each person's 32-vector solved exactly from own ratings.
Mechanism: AlphaRec (ICLR 2025) — LLM title embeddings carry collaborative signal once a linear map is learned from many users. If so, the app's fast learner = small exact vector per domain on LLM states: rank-1 updates, order-free, deletable, phone-cheap, and scores **new text-only items** (CF can't) → "knows today's world."
Prior art: not new as a method (content-based filtering, AlphaRec, cold start). Value = diagnostic + app architecture template.
Pass: A ≥ +0.02 AUC over popularity, CI excludes 0. B ≥ 0.72 on rated movies and ≥ 0.68 on movies held out of the lens fit. Stop: A ≤ +0.005 (representation really missing); B ≤ 0.60 on held-out. Side: does the 4B-vs-27B gap vanish when reading states instead of words? Cost < 1 H100 hour, $0 API, 2–3 days.

**P2 — One exact memory read at several depths (is the two-hop curse a storage-depth problem?)**
Frozen Qwen2.5-0.5B/1.5B + **one** KV table written by the exact ridge solve, read at 3 depths (e.g. layers 6/12/18) with learned query projections and small read LoRAs. Values can be keys (pointer chains): hop 1 returns a middle-entity code, a later read fetches its fact. Meta-train read/write projections on synthetic worlds of fresh random people; facts live **only** in memory; middle entities are new people; final answers known entities.
Mechanism: fact stored at one depth, bridge entity computed later; a memory shared across depths removes the mismatch; reading circuit trained on the interface, not entity-specific weights → should transfer to unseen entities (Karmim: weight-stored new entities don't compose).
Prior art: MemN2N (2015) multi-hop on bAbI; MemReasoner, Identity Bridge, CaKE (2025); Larimar, KBLaM (single-depth, weak multi-hop); MesaNet (exact ridge within one sequence). Not found: persistent exact order-free post-deployment memory read at several depths inside a pretrained LM, tested on latent two-hop with both facts new.
Pass: held-out entities single-hop ≥ 90% (gate); latent two-hop both-new ≥ 50%, ≥ 30 pts over identical single-depth memory, ≥ 35% on relation pairs never composed in training; compare vs facts LoRA-fine-tuned into weights (expected near chance). Stop: multi-depth beats single-depth by < 10 pts or < 20% absolute after 3 H100 hours tuning. Only if pass → real MQuAKE without chain (2.5% now). Cost 1–2 weeks eng, 3–6 H100 hours.

## 3. Odds
P1: ~70% clear answer either way, ~0% breakthrough (it's an app architecture decision). P2: ~30% multi-depth clearly beats single-depth on synthetic, ~10% transfers to real facts noticeably, <3% breakthrough. Overall ~3–5% breakthrough. Realistic payoff: sound two-speed architecture for the app (population representation + exact per-person layer + retrieval into context) and one clean answer on where latent composition fails.

Sources cited: MesaNet arXiv 2506.05233; AlphaRec arXiv 2407.05441; Identity Bridge arXiv 2509.24653; MemReasoner arXiv 2503.07903; CaKE arXiv 2503.16356; Persistent Memory for Continuous Latent Reasoning arXiv 2606.07720; ALEX AAAI 2026.
