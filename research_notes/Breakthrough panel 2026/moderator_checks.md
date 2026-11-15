# Moderator's checks of round-1 claims

Every panel member reads this before round 2.

## Claims about this repository (re-run by the moderator)

| Claim | By | Check | Verdict |
|---|---|---|---|
| B's dev index has 49 wrong fact rows of 781 (6.27%): 15 filed under the anchor of "X's flatmate", 9 earlier-message pronouns, 18 sent to someone not mentioned, 7 under the teller of second-hand news | founder | Rebuilt the index from `results/messy_index_dev_dev1500/write.jsonl` and re-classified the rows | **Confirmed.** 49 of 781. The moderator's classification gives 14 / 9 / 19 / 7, a one-row boundary difference. |
| Perfect pages, thinking off, by kind: two-hop 23.5%, retractions 26.7%, roles 33.3%, last meeting 33.3%, nicknames 58.8% | researcher | Recomputed from `answers.jsonl` | **Confirmed exactly.** |
| Test 1: the 4B with thinking off scored 95.1% on the hard set with perfect pages, in about 3 tokens per answer | researcher | `reports/Notebook reading at shipping size 2026.md`, lines 979–984 | **Confirmed.** |
| Test 1: the 4B reading the whole 5,000-note notebook scored 20.4% (thinking on) and 24.7% (thinking off) | founder, architect | README | **Confirmed.** |
| Test 2's writer uses about 154 tokens per message (15 completion + 139 prompt) | neuroscientist | dev summary | **Confirmed.** |

## Outside papers (abstract pages opened by a separate fetch, 15 Nov 2026)

| Citation | By | What the abstract says | Verdict |
|---|---|---|---|
| DISCERN, arXiv 2611.15729 | researcher | "Preserve at write, resolve at read"; never merges confusable entities; entity resolution 0.800 "where no baseline exceeds 0.40". Benchmark is synthetic and author-built (ENT-310). | **Confirmed.** Caveat: entities in an enterprise setting, not personal chat. |
| Block-Attention, arXiv 2409.15355 | architect | Blocks encoded independently; after fine-tuning, "comparable to full-attention models" across 11 benchmarks. | **Partly.** The specific HotpotQA and 2Wiki numbers are not in the abstract; they may be in the paper's tables. |
| User as Code, arXiv 2606.16707 | researcher | Typed Python objects, append-only log; on aggregate questions retrieval "collapses (6–43%)" while it stays "near-perfect (99%)"; can surface unsolicited alerts. | **Confirmed.** |
| AfterDeleteBench, arXiv 2610.31366 | researcher | Qwen2.5-7B: a single query recovers the deleted secret "in roughly two thirds of cases"; provenance-aware closure "eliminates observed recovery in the tested positive-baseline settings". | **Confirmed**, with the "tested settings" qualifier. |
| "Subtract or Replay?", arXiv 2607.27539: "in a 48B hybrid, 12–49% of a record's contribution depended on later writes" | architect | That ID is a different paper: "Can an AI Assistant Really Forget? Auditable Deletion from Addressable Memory" (Gemma 3, 4B). No 48B model, no 12–49% figure. | **Not supported. Treat as a false citation** until the architect gives a correct source. The general point (a store written with the delta rule makes one message's trace depend on later ones) is a [recall]-level argument, not a checked finding. |

## Notes for round 2

- **The architect must withdraw or correct the "Subtract or Replay?" claim.** Arguments that rest on it (M2's "exactness of deletion is a property of how the store was written") must stand on reasoning alone.
- Most 2026 sources are preprints read at abstract level only. Treat their numbers as the authors' claims.
- **Four of five members chose the same root without coordination:** the model doing identity and bookkeeping in its head, with write-time decisions final. The fifth (the engineer) has not reported yet.
  - The founder: remove recurrence.
  - The researcher: bind first, read fast.
  - The architect: write locally, read globally.
  - The neuroscientist: a memory that re-reads itself.

  They are the same underlying model with different instructions. Agreement is weak evidence.

## The engineer's round 1 (checked after it arrived)

| Claim | Check | Verdict |
|---|---|---|
| CPU simulation: with a perfect describer, code resolves who is who with 0% wrong filing. At 10% simulated describer error: 1.03% wrong, "Kofi's flatmate" 85.9%, worst-person forgetting 92.0%. Order invariance holds. Deleting nickname messages unbinds all 228 nickname mentions. Use-before-definition world: write-time binding leaves 39% and 37% unfiled, re-solve leaves 6% and 2%. | Re-ran `engineer_scripts/resolver_sim.py` (copied into this folder) | **Reproduced exactly.** It is a simulation from the generator's truth with injected errors, written after reading the generator (the designer's advantage, which the engineer states). It does not show that the 4B can write the descriptions. |
| B's link rows were right 57.1% of the time; B's writer put lure values into 51.9% of lure messages | `results/messy_index_dev_dev1500/result.json` | **Confirmed** (link precision 57.14%, lure intrusion 51.91%). |
| Test 2's world uses "Kofi's flatmate" only after the relation was stated, nicknames only after they were introduced, and bare first names only when clear at that moment | `brain/messy_world.py` lines 598–617 (`link_known`, `nick_known`, `known_first`) | **Confirmed.** Test 2 therefore cannot measure deferred reference (use before definition). |
| PowerInfer-2 measured 0.257 J per token on a phone | not re-checked | Treat as the engineer's citation. |

## Convergence, updated

All five members, independently, put the root at the same place: **the model decides identity while writing, with the memory in view, and the decision is final.** The remedy has three names:
- the founder: remove recurrence;
- the engineer: describe, don't decide;
- the researcher: bind first.

The architect's "write locally" and the neuroscientist's "re-read" are close variants. Round 2 must say whether this is real convergence or one model agreeing with itself.

## Round 2 checks

| Claim | By | Check | Verdict |
|---|---|---|---|
| Of B's 86 wrong hard-set answers: 26 wrong person, 17 a person split into several ids, 7 a fact never written, 18 filed right but not found, 18 wrong with every needed message in hand. Write-time identity explains about half (43 of 86). | founder | Re-ran `founder_scripts/r2_gap.py` (copied into this folder) | **Reproduced exactly.** One cause per question, writing causes counted first, so the writing share is the generous figure. |
| About 16% of true facts appear in no index row from their own message; 5.6% are filed under the wrong person | neuroscientist | Consistent with the dev summary's "facts filed right 78.2%" (100 − 78.2 = 21.8 ≈ 16 + 5.6) | **Consistent.** Not independently recomputed. |

## Correction to the moderator's own round-1 check: the architect's citations

The round-1 check read only the **current** version of arXiv 2607.27539 and called the citation false. After the architect's round-2 correction, a second fetch read the earlier version and the companion paper:

| Citation | What the pages say | Verdict |
|---|---|---|
| arXiv 2607.27539**v2** (13 Aug 2026), "Subtract, Transport, or Replay? Auditable Deletion from Language-Model Memory" | Abstract: "the corpus-pooled raw recurrent contribution changes by 12-49% with the suffix". It names native Kimi Delta Attention (KDA) but gives no model size. Its v3 comments say the KDA work moved to "a separate companion paper". | **Supported** for the 12–49% sentence. |
| Companion paper arXiv 2609.06872 (same single author, Vy Labs; v3, 25 Sep 2026) | Full text: "We evaluate the four strategies on the released 48B Kimi Linear hybrid". Replay "matched the never-stored reference … across all 80 evaluation and calibration contexts". The 12–49% (additive writes: none) and 0.7–32.9% figures are reported in sections VI-A and VI-B, per a model-extracted summary, not read verbatim. | **Mostly supported.** |
| Block-Attention, arXiv 2409.15355**v3**, Table 1 | Llama3-block-ft HQA 68.5, 2wiki 73.3; full-attention baseline HQA 69.5, 2wiki 73.4 | **Supported** for v3. The architect notes v5 gives larger gaps (72.3 vs 74.8; 72.2 vs 73.2). |

**Revised verdict:** not a false citation, but a bad one. The original lacked a version number, and its paraphrase ("12–49% of a record's contribution depended on later writes") was stronger than the text ("changes by 12–49% with the suffix"). The underlying finding is a single-author preprint on synthetic contexts. The moderator's "Treat as a false citation" in round 1 was itself too strong, because it rested on one version of the page.

## Round 2 checks (continued)

| Claim | By | Check | Verdict |
|---|---|---|---|
| One misread role definition files on average 13 extra facts under the wrong person (median 9, worst 53). At a 10% misreading rate for role definitions only, wrong filing averages 0.70% (90th percentile 2.0%, worst 4.3%). | architect | Re-ran `architect_scripts/definition_errors.py` | **Reproduced exactly.** |
| With anchor errors included and errors concentrated on hard messages, wrong filing rises: 1.88% at 10%/10%, 2.65% at 5%/30%, 3.6–3.8% with real-chat-like references | engineer | Re-ran `engineer_scripts/r2/adversarial_sim.py` | **Reproduced exactly.** |
| Test 1, thinking off, two-step questions: 93% right with 10 notes, 40% with 100, 5% with 5,000. Most wrong answers give the named person's own value (10 of 14, 90 of 120, 98 of 190). | engineer | Re-ran `engineer_scripts/r2/twohop_errors.py` on test 1's saved answers | **Reproduced exactly.** |
| REALTALK: none of 14 named third parties named before being introduced; 4 of 14 role first, name later; about 200 "my <relation>" mentions, many never named | researcher | not re-run (hand count) | The researcher's count. |
