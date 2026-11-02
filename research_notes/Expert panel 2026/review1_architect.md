# Test 1 review — Architect
Verdict: run WITH CHANGES. Part A as written not worth the window; A2 and Part B worth it as they stand.
Part A ~80% PASS on 27B; PASS wouldn't show whole-notebook = v1 (hard parts removed). STOP likely a serving artifact (cap, cache). Either way team builds notebook + page selection (needed on phone anyway) → Part A decides almost nothing. A2 (27B use ratio, thinking on) and B (contradiction with thinking) test held conclusions.
Flaws:
1. Critical — text search solves it: unique full name in every question, prefix-free names; two-hop questions contain the relation note verbatim ("Signe Larsen's neighbour"); second hop uses the fact template word ("pet"). Reword removes overlap only for the attribute, not the key. NoLiMa's drop comes from removing overlap on the key. Name search + 9 attribute words ≈ 100%.
2. Critical — predictable: unique-key retrieval at ~50K = RULER multi-key needle; strong 30B-class near ceiling (from memory, fairly sure). Predicts reword ≥ 95%, ratio ≥ 0.9.
3. High — template notes exclude pronouns, roles, nicknames, two Anas, "might move", corrections, small talk; name collisions ruled out by the filter.
4. Medium — update Q gives it away: "now works as" / "work as now".
5. Medium — 10-note denominator mostly not two-hop: A has the asked attribute only ~4/9 of the time; otherwise the 10-note pile holds a single pet note → answered without resolving the relation. Ratio compares real two-hop at 5,000 with mostly single-hop at 10.
6. Medium — ignores cost: 50K prompt on a phone = minutes per turn; 4B at 1,000/5,000 answers a question that never arises. 27B decision should weigh tokens and latency.
7. Low–med — prefix caching on hybrid attention: SGLang reuses linear-attention state only at stored boundaries; warm-up ends in a different question → may never hit. Uncached: ~43M prompt tokens for personal 5,000 arm; window could close before A2/B. Budget, not validity.
8. Low — 6,144 cap mixes budget with reading.
9. Low — strict scoring: "Porto (moved from Lisbon)" wrong.
10. Low — power fine; problem is ceiling.
Fixes: BM25 top-20 arm read by 4B (minutes); remove full name from half the questions (first names shared by 3–4, roles "the Porto nurse", alias notes "Paula goes by Polly"); updates in original template; two-hop only where A has a different value of the same attribute; report accuracy excluding cut-offs + rerun at 16K cap; lenient "first value named" secondary score; drop 4B above 100 notes and thinking-off at 5,000.
Replacement if one window: LongMemEval-S (500 q, ~115K-token real chat histories; extraction, multi-session, temporal, knowledge updates, abstention). 27B think: full history (stratified 250, ~29M prompt tokens), evidence sessions only, BM25/embedding top-k; 4B: evidence-only and retrieval. Lines: full ≥ 0.9× evidence-only → long context v1 in cloud; retrieval ≥ full → page selection v1 everywhere. Run A2 and B unchanged.
