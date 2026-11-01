# Expert panel: what is stopping the breakthrough (November 2026)

The question put to the panel: what is stopping an AI that understands and grows from what it remembers?
That means an AI that learns after deployment, improves with experience, uses new knowledge in reasoning,
stays accountable, and eventually runs one per person on a phone. Could a small team test something that
is neither already done nor already refuted?

**How the panel ran.**
- Four agents with role prompts: a machine-learning architecture expert, a curious founder, a senior
  neuroscientist and an AI research scientist.
- Each got the full research record and the team's resources.
- Round 1: each wrote alone. Round 2: each read the other three and answered.
- The moderator checked every claim about this project's own numbers against the repository.
- All four are the same underlying model with different instructions. Their range of views is narrower than
  four real people's, and some of their agreement is shared background rather than independent
  confirmation. What they added that the record lacked:
  - literature searches;
  - a confound in our own runs;
  - one merged experimental design.
- Transcripts: `research_notes/Expert panel 2026/` (`r1_*.md`, `r2_*.md`).

---

## 1. The answer, in plain words

**Nobody, here or at the large labs, has a way to turn a few new experiences into knowledge the model uses
as if it had always known it.**
- Storing facts is solved (this project's notebook: 96% after 2,764 facts, exact deletion).
- Using a stored fact *when asked about it* is solved.
- Two things are not solved:
  - combining two new facts without writing out the steps;
  - new knowledge shaping answers that never asked about it.

The panel agreed on why. **A new fact becomes part of the model's thinking only in two cases:**
- it was active *together with* the facts it must combine with, either in the same training text or in the
  same reasoning pass;
- the network was trained to read the memory it is stored in.

Facts stored one at a time do not bind to each other. That holds for our notebook, for fine-tuned weights
(Balesni et al.: 30 paraphrases per fact still fail), and, the neuroscientist added, for separately encoded
episodes in the brain.

**The uncomfortable second half.** The team concluded that "a good notebook plus the model reasoning over
it" is not enough. The evidence for that came from:
- a 0.5B model (pasting facts into the prompt: 43% → 13%);
- GPT-J (multi-hop, own-fact ceiling 23.4%);
- Qwen3.5 runs with thinking switched off.

**So we do not yet know whether a companion app needs the breakthrough at all.** The founder's summary
sentence:

> No one can yet turn a few new experiences into knowledge the model uses as if it always knew it. But
> for a companion we haven't shown we need that, because our evidence that "good retrieval plus a thinking
> model" falls short was measured on models too small and too unthinking to count.

---

## 2. Corrections to our own record

| Claim in our record | What the panel found | Checked against |
|---|---|---|
| (b) "LLMs can't check their own conclusions" (37% / 96% of guesses contradicted the evidence) | **Confounded.** Every Qwen3.5 run had thinking off, and proposals were sampled at temperature 0.8. The result shows "unverified non-thinking guesses contradict the evidence", which was already known (Huang et al. 2023). It must be re-run with thinking on before we keep it. | `brain/order_gap.py:61`, `:311` |
| (e) "Small models extract lessons poorly" | Same confound. Not established until re-run with thinking on. | same |
| Pasting facts collapses (43 / 27 / 13%) | Measured on Qwen2.5-0.5B. Larger models read context better (but see NoLiMa below). | `FIXING_THE_BOTTLENECKS.md:538-553` |
| The pasted matched fact gave 9.7% on real edits | That was the 0.5B model in question form. In fill-in-the-blank form it scored 88.5%, and GPT-J with the pasted fact scored 84.0% in question form. This is a small model failing to read context, not the model refusing a belief. Belief conflict does appear in multi-hop: 19.5% of answers were the old answer (19.3% with the fact pasted). | `FIXING_THE_BOTTLENECKS.md:602, 633, 822`; `results/reallm4_gptj/summary.md:203-204` |
| The keyed memory's 96% / 98.7% | Holds on MQuAKE's clean names. Free-text personal memory is untested: nicknames, pronouns ("she" = Ana), "my boss". | – |
| (a) Fast learning needs a representation learned in advance | **Solid.** All four agree it is the most important finding. It is also the textbook collaborative-filtering / cold-start result (Kang et al. 2023), so present it as a careful replication. | – |
| SPARC's order-freedom thesis | Holds for the fast layer only. Learning the features themselves cannot be order-free in general, and test 4 (CIFAR) confirms it. | – |

One panel claim was wrong and was retracted. The researcher said our MEMIT / MEND / MeLLo comparison used
the original, contaminated data. It did not: the references are from MQuAKE arXiv v3, Table 5, on
CF-3k-v2, the file we ran (`FIXING_THE_BOTTLENECKS.md:726-735`).

**Prior art the team had not reviewed.** The researcher checked abstracts for most of these. Those marked
(memory) were cited from memory and are not re-checked.

| Work | Why it matters here |
|---|---|
| **KBLaM** (2410.10450) | A frozen LLM reads an external knowledge store at every attention layer. "Read memory at several depths" is already done. |
| **Feng, Russell, Steinhardt** (ICML 2025, 2412.04614) | A fine-tuned fact generalizes as a first hop if early layers were trained, and as a second hop if late layers were. Any composition test must vary hop position. |
| **O'Neill et al. 2026** (2607.11020) | After 20 sequential writes, earlier facts are retained 1% with repeated statements vs 46% with 24 varied restatements, whatever the update method. The data sets retention, not the method. |
| **Doc-to-Atom** (2606.12400) | A micro-LoRA per knowledge unit, with a provenance key and a router. It overlaps "keyed values plus router". Its abstract says nothing about sequential writes, deletion, order-freedom or multi-hop. Our claim survives if stated as: exact joint re-solve, order-freedom, exact deletion, capacity certificate, 2,764 lifelong edits. |
| **Learning on the Job** (2607.22157) | Frozen model plus rules learned from deployment feedback: 1.6× from success/failure verdicts, 2.6× from corrections (τ-bench). The cheap baseline a companion must beat. |
| **Continual On-Device Personalization of SLMs from Hindsight** (NeurIPS 2026 workshop poster) | Self-distillation from later user activity, on a phone. The companion idea in miniature. |
| **NoLiMa** (2502.05167), **LongMemEval** (2410.10813) | Long-context reading drops sharply when the question shares no words with the fact. At 32K tokens, 10 of 12 models fall below half their short-context score; GPT-4o goes from 99.3 to 69.7. LongMemEval-S: 30–60% relative drop. |
| **Grokked Transformers** (memory) | Composition fails on entities never seen in compositional training examples, even when each entity has many facts. |
| **Olausson et al.** (ICLR 2024, memory) | At equal compute, self-repair rarely beats sampling fresh attempts. |

---

## 3. Where the panel agreed

1. **Separate stores, not one mechanism.** Each store needs a different representation and update rule.
   Forcing all of them into one weight update is what makes them interfere.

   | Store | Holds | How it learns |
   |---|---|---|
   | Exact notebook | facts | write and delete exactly |
   | Population map plus a personal vector | preferences | map learned slowly from many people; personal vector solved from the person's own data (the collaborative-filtering template) |
   | Journal | episodes | append everything |
   | Slow distillation | skills | trained slowly, over time |

   The new problem this creates is **arbitration**: deciding which store answers.
2. **Order-freedom and integration are compatible.** Make every derived product a deterministic function of
   the stored *set* of facts, rebuilt from the journal. Derived products here means derived pages or a
   nightly LoRA. Deleting a fact means rebuilding without it, which the brain cannot do.
3. **Integration needs nonlinear computation.** The ridge solve's cross terms resolve interference between
   facts, but they never produce a new fact. New facts come from reasoning, either live (over retrieved
   facts) or offline (a nightly pass that puts related facts together).
4. **One-exposure learning is mostly a myth** (Horst & Samuelson 2008; adult fast mapping failed to
   replicate). That reframes the goal rather than killing it. Recall after one exposure is what the
   notebook already does. Absorbing a fact into thinking can happen over the following nights.
5. **Don't decide what to keep by surprise** (founder Q9, neuroscientist). An LLM cannot predict your
   sister's name, so surprise filters out almost nothing personal. And a plausible fact like "penicillin
   allergy" would be dropped. Instead:
   - log everything (text is cheap);
   - gate *promotion* to the notebook, weights or derived pages on novelty against memory, importance and
     how often it is retrieved;
   - use surprise at retrieval time, meaning corrections.
6. **Run first: the re-baseline at shipping size** (section 5, test 1). All four chose it.

## 4. Where the panel disagreed

- **Is combining facts without written steps worth pursuing?**
  - Architect, neuroscientist and founder: not a product requirement. People and brains think out loud, and
    the architect downgraded his own multi-depth proposal from an app need to a science bet.
  - Founder: "Everyone says it isn't needed, then spends three proposals on it. Pick one position."
  - The one product argument for it (architect, Q3): **unprompted relevance**. A notebook acts only when
    queried, while integrated knowledge shapes every answer. Nobody measures that yet.
- **What makes a new fact composable?** Three competing hypotheses:
  - **H1, richness:** the middle entity has many other facts. Researcher for; architect sceptical, citing
    Grokked Transformers, pass ~25%.
  - **H2, entity transfer:** another chain through the same entity was seen together. Neuroscientist: ~50%.
  - **H3, relation schema:** this relation pair was composed for other entities. All predict failure.

  The neuroscientist turned these into one factorial experiment (section 5, test 3).
- **Multi-depth memory vs recurrence.** Neuroscientist: brains fix depth by looping, where retrieved output
  becomes the next cue, not by parallel reads at fixed depths. Three read depths give at most about three
  hops, so add a loop arm. Researcher: KBLaM already reads at every layer, so the open part is only
  pointer-chain values and latent two-hop with both facts new. A ~$5 check on released KBLaM comes first.
- **Is MovieLens the right test of "knowing a person"?** Founder and researcher: within-person AUC over
  movies the person chose to watch mostly measures movie quality, and the data is selection-biased
  (Schnabel et al. 2016). Keep stated preferences (notebook) separate from revealed taste (population
  model). And "my dad is sick" has no rating.

---

## 5. Merged plan, ranked

| # | Test | Breakthrough potential | Value to the app | Cost | Chance of a clean answer |
|---|---|---|---|---|---|
| 1 | **Re-baseline at shipping size** | ~0 | very high | 1–2 H100 h, $0 API | ~90% |
| 2 | **Population lens** | ~0 | high (preferences) | < 1 H100 h | ~80% |
| 3 | **What makes a new fact composable** (factorial) | low–medium | medium | ~6 A100 h + 1 H100 h, ≤ $15 | ~70% |
| 4 | **Trained reader over the exact memory** (moonshot) | highest on the list, still low | low near term | 1–2 weeks, 3–6 H100 h | ~45% (a fail is ambiguous) |
| 5 | **Nightly consolidation with receipts** | depends on 3 | high | ~$10–30 | medium |
| 6 | Surprise-gated restart vs fresh copy vs patching, × thinking on/off | low | low | 2–4 H100 h | high |
| 7 | Doc-to-LoRA adapters accumulated over time | low | medium | ~$20 | high |

What each test is:
1. **Re-baseline at shipping size:**
   - Qwen3.5-4B and 27B, thinking on and off.
   - 10 / 100 / 1,000 / 5,000 facts in the prompt.
   - Fictional personal facts vs conflicting world edits; questions worded like the fact vs reworded.
   - Single-hop, two-hop and updates; a top-k page-selection arm; chains logged step by step.
   - The scientist contradiction rate re-measured with thinking on.
2. **Population lens:** frozen LLM hidden states for each item, a linear map learned from other users, and
   each person's vector solved exactly. Scored on items held out of the lens fit (cold start).
3. **What makes a new fact composable:** design in section 6.
4. **Trained reader over the exact memory:** one exact, deletable memory, read at several depths plus a
   loop arm, with pointer-chain values. Gated by the KBLaM check and run after test 3.
5. **Nightly consolidation with receipts:**
   - derived pages, each linked to the facts it came from, withdrawn when a source changes;
   - later, a LoRA rebuilt nightly from the notebook.

   Product engineering. The architect predicts it helps when the edit is the first hop and not when it is a
   later hop (the reversal curse). Split results by hop.
6. **Surprise-gated restart:** run only if contradictions persist with thinking on.
7. **Doc-to-LoRA accumulation:** tied to Gemma-2-2B, not on the app's path.

**Proposed pass and stop lines for test 1.** These are a proposal; they get fixed in a committed
pre-registration before any run.
- **Pass:** at 5,000 personal facts in the prompt, Qwen3.5-27B with thinking on scores ≥ 85% on *reworded*
  single-hop questions, and ≥ 0.8× its own-fact rate on two-hop. Then version 1 of the app is notebook plus
  retrieval or long context, and that is the bar every memory method must beat.
- **Otherwise:** the breakdown shows where it fails: fact count, missing shared words, belief conflict or
  composition.

**Gap nobody filled.** No test covers what users actually notice:
- recalling the right memory unprompted;
- learning from corrections (measured elsewhere: 2.6× in Learning on the Job);
- resolving "she" to Ana in real chat.

The architect and the founder both judged a small simulated-conversation benchmark for these worth more to
the app than tests 4–7.

---

## 6. The factorial design (test 3), with the moderator's corrections

**The neuroscientist's design:**
- Qwen3.5-4B with LoRA, two seeds.
- 420 fictional bridge people in 7 cells of 60, each with 2 test chains A → B → C (A a new person, C a real
  entity), so 120 items per cell.
- A → B and B → C appear only in separate documents, 8 wordings each.
- Scoring: 10-option multiple choice by logprob, no chain-of-thought.

**Cells:**

| Cell | Condition |
|---|---|
| 1 | Baseline: no extra facts, relation pair never composed anywhere |
| 2 | 16 extra facts about B |
| 3 | 64 extra facts about B |
| 4 | A sibling chain A′ → B → C′ co-replayed in one document |
| 5 | Cell 4 plus 64 extra facts about B |
| 6 | The relation pair composed for 100 other, untested bridges |
| 7 | Positive control: the test chain itself in one document |

**Gates:**
- single hops ≥ 90% in every cell;
- cell 7 at least 20 points above cell 1;
- unrelated QA within 1 point.

**Pass lines (Holm-corrected):**
- H1: cell 3 − cell 1 ≥ 15, rising from cell 1 to 2 to 3;
- H2: cell 4 − cell 1 ≥ 15 and ≥ 30% of the cell 7 gain;
- H3: cell 6 − cell 1 ≥ 15;
- interaction: the replay gain is ≥ 10 points larger with 64 extra facts.

**Stop line:** cells 2–6 all within 5 points of cell 1 while cell 7 passes. Then, at 4B, putting facts into
weights only caches what was seen together. Use explicit chains plus nightly derived pages instead.

**Moderator's corrections before this could be pre-registered:**
1. **Leakage between cells.** The design trains all cells in one LoRA run. Composition documents in cells
   4–7, and the relation-pair practice in cell 6, can teach a general "compose" skill that also lifts cell
   1, which shrinks every difference. Fixes:
   - give each cell its own disjoint relation pairs (10 relations give 90 ordered pairs);
   - better, train each cell separately (7 cells × 2 seeds = 14 short LoRA runs on 4B).
2. **Power.** At 120 items per cell, a 15-point difference is about 2.3 standard errors. Under Holm
   correction across four tests, the strictest threshold needs about 2.5. Double the items to 240 per cell;
   the cost is small.
3. **Hop position.** Add Feng et al.'s first-hop vs second-hop factor to at least cells 1 and 4. Otherwise a
   null result could reflect which layers the LoRA trained rather than the hypothesis.

---

## 7. Odds, as each expert gave them

| | Breakthrough | Solid, usable result | Novel publishable result |
|---|---|---|---|
| Architect | ~3–5% overall; < 3% for test 4 | – | – |
| Researcher (round 2) | ~3% | 65–75% (tests 1 and 2 settle the app's design) | 25–35% (tests 3 or 4) |
| Neuroscientist | – | ~90% clean answer for test 1, ~70% for test 3 | – |

These are opinions, not measurements. This project's own forecasting record is 0 for 5.

## 8. What the panel did not deliver

- **No breakthrough idea.** Every proposal is either engineering with heavy prior art (tests 1, 2, 5) or a
  science bet with single-digit odds (tests 3, 4).
- **No benchmark for what users notice.** Every test still scores accuracy on direct questions.
- **Unchecked citations.** Several 2026 citations were checked only by the agent that gave them. KBLaM,
  Feng et al. and the NoLiMa / LongMemEval numbers decide designs, so each needs a check before a
  pre-registration relies on it.
