# Round 2 — Neuroscientist

Self-corrections: working-memory argument leaned on 0.5B 43/27/13%. The 9.7% is also 0.5B; GPT-J pasted got 84% question form. Belief conflict survives only in multi-hop (19.5% old). Founder hunch A is the right check.
Source confidence: Balesni v4 high; MATSIR and 2604.04182 medium-low (snippets); others' 2026 citations unchecked.

## 1. Agree / disagree
- **Architect "integrates only if trained in the loop": half right.** Brain co-develops hippocampus/cortex, but the main trick: hippocampus doesn't hand cortex foreign vectors; it **reinstates a cortical pattern the cortex itself produced** — no special reader needed. The 92% for pasting only the matching fact is that reinstatement. Two routes: train a reader, or **write memories in the format the reader already reads**. Team's slot values are foreign vectors fitted to output the answer → only work in the text interface. Caveat both ways: episodes encoded separately then reinstated don't bind (brains or separately cached text). **Integration needs co-activation** on either route.
- **Multi-depth memory**: sound diagnosis (bridge resolved too late for where fact is stored), but the brain fixes depth with **recurrence in time** (hippocampal output loops back as next cue; REMERGE), not parallel fixed-depth reads. Three depths ≈ at most three hops. Add a loop arm: one read position, late state fed back for a second pass, meta-trained (Coconut-style; expect training needed).
- **Researcher #1 vs my P1: competing hypotheses → one factorial.** H1 richness (many facts about B); H2 entity-level transfer (another chain through B learned co-active); H3 relation schema (this relation pair composed on other entities). Tse: richness ≠ schema; schema = consistent relational structure new items slot into; 64 unrelated attributes = associations. Predict H1 small, H2 ~50% pass, H3 fails (Grokked OOD), richness × replay interaction.
- **Researcher #3 rebuildable consolidation: better than the brain for this goal.** Retrain-from-scratch on everything = McClelland 1995 fully interleaved ideal → no catastrophic interference by construction. Brain can't delete a consolidated memory by removing the episode (H.M.); this design can = accountability. Cost grows with memory; McClelland 2013: schema-consistent items need little interleaving → schedulable. But more data/rewordings control recall, not composition (Balesni 30 paraphrases fail). Student needs co-occurrence documents → depends on the factorial.
- Thinking off: agreed; my P2 must be crossed with thinking on/off.

## 2. Founder questions
- Q5: child's map = concept categories + mutual exclusivity + social gaze + syntax; still little kept from one exposure without repetition/sleep. Map for knowing a person splits: **facts** — LLM already holds a generic map of a person's life (sister, job change, scan); personal facts fit. **Preferences** — no such map in text knowledge; must come from population data (MovieLens). Caution: composing two personal facts ("my sister's husband's job") is fresh+fresh, Balesni's hardest case.
- Q8: yes hidden assumption. Brain: facts/events hippocampus+cortex; habits striatum; values via RPE; arbitrated by reliability (Daw 2005). App: exact notebook, preference vector in population space, skills trained slowly. New problem: **arbitration** (which store answers).
- Q9: **don't gate encoding on surprise.** LLM can't predict your sister's name or allergy → filters almost nothing personal; would drop important plausible facts ("penicillin allergy"). Brain keeps boring-but-important facts because goal-relevant and repeatedly recalled. Instead: log everything in cheap journal; gate *promotion* (notebook/weights/derived pages) on novelty vs memory (CA1 mismatch), importance category, retrieval frequency; use surprise at retrieval = corrections = reconsolidation.
- Q11: probably mostly the loop. Old rule in context = perseveration (task set never inhibited). Fresh copy = reset = my "always-restart" arm; cross with thinking on/off.

## 3. Merged ranking
| Proposal | Breakthrough | App | Cost | Clean |
|---|---|---|---|---|
| A. Founder A + researcher controls: 27B context personal vs conflicting, thinking on, logged chains | none | very high | ~1 H100 h | ~90% |
| B. Factorial (below) | medium | med-high | ~6 A100 + 1 H100 h, ≤$15 | ~70% |
| C. Arch P2 + loop arm | highest | medium | 1–2 wk, 3–6 H100 h | ~45% |
| D. Population lens on LLM states for cold items | low | high (prefs) | <1 h | ~80% |
| E. Surprise-gated reset vs fresh copy vs patch × thinking | low | low-med | 2–4 h | high |
| F. Rebuildable student / derived pages with receipts | depends on B | high | medium | — |

## 4. First / highest ceiling
(a) A (decides v1 in an hour), then B same week. (b) C: a trained reader over an exact, deletable memory that composes fresh facts is the session's real target; B tells C what to train on.

**Merged design B.** Qwen3.5-4B LoRA, one training run, two seeds. 420 fictional bridge people B in 7 cells of 60; each B has 2 test chains A→B→C (A new person, C real entity) → 120 items/cell. A→B and B→C only in separate documents, each worded 8 ways. 10 relations. Score: 10-option MC by logprob, no CoT, chance 10%.
Cells: 1 baseline M=0, relation pair never composed anywhere; 2 M=16; 3 M=64; 4 sibling chain A′→B→C′ co-replayed in one doc, M=0; 5 same at M=64; 6 relation pair co-replayed for 100 other untested bridges, this B M=0; 7 positive control: test chain itself in one doc.
Gates: both single hops ≥ 90% no CoT in every cell; cell 7 beats cell 1 by ≥ 20 pts; native QA within 1 pt; use ratio vs native two-hop on matched real relations.
Pass (Holm): H1 cell3−cell1 ≥ 15 and monotone 1→2→3; H2 cell4−cell1 ≥ 15 and ≥ 30% of cell-7 gain; H3 cell6−cell1 ≥ 15; interaction replay gain at M=64 ≥ 10 pts larger than at M=0.
Stop: cells 2–6 all within 5 pts of cell 1 while cell 7 passes → latent integration through weights is only same-document caching at 4B → explicit chains + nightly derived pages.
Add-on: Patchscope readout of the layer where B is resolved per cell (tests storage-depth explanation for free).
