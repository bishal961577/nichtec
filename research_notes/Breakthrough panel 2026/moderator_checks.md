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
