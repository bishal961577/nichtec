# Fixing the Bottlenecks: A Memory That Stores Exactly, Grows When Full, and Never Needs a Backward Pass

This follows the research report [`reports/Brain like lifelong AI memory.md`](reports/Brain%20like%20lifelong%20AI%20memory.md).
That report found the parts for a phone AI that learns into its own weights, and the problems that
stop them from working together. This document takes each problem, finds its mechanical cause,
and proposes a fix. The core fixes are checked in simulation (`memsim/`), and a prior-art check
was run on each one.

**What the fixes deliver:**
- **By construction, confirmed in simulation:**
  - Stored facts are never silently lost.
  - The memory is identical whatever order facts arrive in.
  - Deletion is exact.
  - When the memory fills up, a nightly check detects it and the memory grows to fix it.
- **Design proposals still to be tested on a real language model:**
  - Recognising a fact however the question is worded.
  - Learning on the phone without a backward pass.
  - Using new facts in multi-step reasoning.
  - Overriding strongly held old beliefs.
- **Not solved:** learning new skills, as opposed to facts.
- **Real-model tests (Qwen2.5-0.5B on a laptop GPU, section 4):**
  - **Test 1: the design failed as specified.** The lookup key did not tell people apart, so even
    the joint solve recalled only 36% of facts at 18% of capacity. Forward-only targets also failed.
  - **Test 2: taking the person half of the key from the person's own tokens fixed most of it.**
    - First-night recall: 96% (written wordings) and 44% (unseen), against 36% and 6% in test 1.
    - Known facts stayed 100% intact behind a novelty gate.
    - A per-answer target codebook built offline matched gradient targets, so the device needs no
      backward pass.
  - **Test 3: canonical, snapped keys make wording irrelevant and keep old facts.**
    - The person key is the name encoded on its own, snapped to the nearest person already written.
      The relation is snapped to the nearest known relation.
    - First-night facts are recalled 100% on written wordings and 96.5% on wordings never seen. The
      gap equals the relation classifier's error.
    - After 24,000 facts, the earliest facts are still recalled at 86.8% (written) and 78.7%
      (unseen). The usual methods fall to 2–8%.
    - Never-written people and known facts are untouched (0% false reads).
  - **Context vs memory, measured.** On the same unseen questions:
    - pasting 10 / 100 / 1,000 facts into the prompt scored 43% / 27% / 13%;
    - selecting the one matching fact into the prompt scored 92% at every size;
    - the memory, holding all 24,000 facts with no context, scored 83%.
  - **Open:**
    - Only a third of the slots are reachable, so recall still declines past about 9,000 facts.
    - Answers are single words from known lists.
    - The subject's name and the set of relations are given to the system rather than discovered.

---

## 1. The bottlenecks and their causes

| # | Bottleneck (measured) | Cause |
|---|---|---|
| B1 | Old facts erode as more facts are written. Sparse-slot finetuning was tested only to ~1,000 facts. | Each night's write changes slots that earlier facts also use, and nothing re-checks the earlier facts. Updates don't commute, so errors accumulate: the same failure this project measured with the delta rule (98.0% shuffled → 22.8% ordered). |
| B2 | Dense-weight editors hit a wall at thousands of edits. MEMIT and AlphaEdit collapse at 2–4K; the AlphaEdit reproduction declines at ~5K. | An edit is a new key→value pair in one weight matrix. Exact storage needs keys independent of everything the matrix already encodes, and that spare room ("null space") is small and fills up. |
| B3 | Facts aren't recognised when asked in different words. GRACE generalises 0.03, WISE 0.58. | The memory lookup is computed from the question's wording. A different wording selects different slots, which were never written for this fact. |
| B4 | The phone can't learn fast: ~3.8 training tokens/s measured, and no phone chip exposes a backward pass. | Gradient training needs backpropagation; phone AI chips accelerate only forward passes. |
| B5 | Written facts aren't used in multi-step reasoning: MQuAKE 40.5% → 7–8%. | The in-between entity is resolved late in the network ("hopping too late"), after the layer where the fact was stored. |
| B6 | Strong old beliefs aren't overridden: 68% on weak priors vs 16% on strong ones. | Written corrections push by a roughly fixed amount, while the model's confidence in the old answer grows with how well it knows it (Override Gap paper). |
| B7 | Skills don't transfer, only facts. | A skill is a computation spread across the network, not a lookup. |
| B8 | "Never forgets" can't be checked. | Gradient methods degrade silently, and nothing records what each fact should produce. |

---

## 2. The fixes

### Fix 1: Exact joint memory on a sparse slot layer (for B1 and B2)

**How it works.**
- Each fact is stored as a small set of **linear constraints** on the value vectors of a large
  product-key memory layer. Each constraint says: "for this lookup, the weighted sum of the
  selected slots' values must equal this target."
- The memory's **keys and the base model are frozen**, so the lookup for a given question never
  changes. That removes representation drift, the other forgetting source this project found.
- Each night, the memory is **re-solved as one ridge least-squares problem over every constraint
  ever written**, not just tonight's. The solver is conjugate gradient, started from yesterday's
  memory.

**Why this removes erosion.**
- The ridge problem has one exact solution, so the memory after N nights equals the solution for
  all N nights' facts together, regardless of order (the commutativity result from
  `AGI_BOTTLENECK.md`, now at the scale of a memory layer).
- When a new fact shares slots with an old one, the solver rebalances both. Nothing is
  overwritten in favour of the newest fact.

**Why this escapes the dense-weight wall.**
- For exact storage, capacity is about the number of *independent* constraints the parameters
  can satisfy.
- In a dense matrix, that is bounded by the key dimension left free by existing knowledge: a few
  thousand in practice.
- In a product-key layer, each slot is its own key dimension, so exact capacity rises to about the
  **number of slots**: about 1 million in a 2²⁰-slot layer.
- Caveat: this bound applies to exact storage. For top-1 (best-guess) retrieval a linear memory can
  hold more (about d²/log n, per 2605.05189); the exact-storage bound is the conservative one.

**Simulation** (`memsim/sim.py`):
- 65,536-slot product-key memory, 4 wordings written per fact, 5,000 new facts per night, up to
  60,000 facts (3.7 constraints per slot).
- Wordings share 70% of their slots.
- Measured: how closely the memory reproduces the stored answer for the stored wording
  (cosine; 1.0 = exact), and unseen-wording recall for the first 1,000 facts ever written.

| Facts written | Constraints per slot | Gradient updates (today's method): stored fidelity | Nightly batch editing: stored fidelity | **Joint solve: stored fidelity** | First 1,000 facts, unseen wording (gradient / batch / joint) |
|---|---|---|---|---|---|
| 5,000 | 0.31 | 0.915 | 0.996 | **1.000** | 0.954 / 0.988 / 0.989 |
| 10,000 | 0.61 | 0.847 | 0.850 | **0.999** | 0.916 / 0.932 / 0.960 |
| 15,000 | 0.92 | 0.792 | 0.702 | **0.992** | 0.867 / 0.743 / 0.880 |
| 20,000 | 1.22 | 0.746 | 0.570 | **0.971** | 0.804 / 0.461 / 0.812 |
| 40,000 | 2.44 | 0.608 | 0.235 | **0.837** | 0.548 / 0.015 / 0.704 |
| 60,000 | 3.66 | 0.515 | 0.100 | **0.732** | 0.347 / 0.001 / 0.576 |

- **Gradient updates erode from the start.** This is B1, reproduced.
- **Nightly batch editing collapses.** Its early facts fall to 1.5% recall at 40,000 facts: the
  published MEMIT-style failure (B2), reproduced by the same mechanism.
- **The joint solve keeps stored facts exact until about one constraint per slot.** Past that,
  it degrades evenly across old and new facts rather than wiping out the oldest. Fix 2 handles
  that point.
- Unseen-wording recall falls even for the joint solve. That is B3, handled by Fix 3.

**Exact deletion** (`memsim/extras.py`):
- Deleted 500 of 5,000 facts and re-solved.
- The result differs from a memory that never learned those facts by at most **0.00018**, where
  typical values are about 1.9. That is solver tolerance.
- Deleted facts' recall went from 1.0 to **0.0**; kept facts stayed at 1.0.
- The raw constraints are needed for this, which is why the design keeps the episode journal.

**Prior art:**
- RLSEdit (arXiv 2601.15686, Jan 2026) already does exact joint recursive least squares over all
  past edits, on a *dense* weight matrix: 10K edits, 89.9 efficacy on Llama-3-8B vs 66.8 AlphaEdit
  and 49.7 MEMIT. Dense weights keep it under the null-space limit in B2.
- Larimar does closed-form writes into a 512-slot memory.
- The prior-art check found no work applying exact joint re-solves to a product-key layer with
  10⁵–10⁶ slots, and no published argument that a slot layer raises the effective key dimension to
  the number of slots. The new part is **the combination**; the least-squares idea itself is not new.

### Fix 2: A nightly certificate, and growth when the memory fills (for B8, and B2 past capacity)

**How it works.**
- After each night's solve, compute the residual for every stored constraint: one extra pass over
  the memory.
- Any fact whose residual exceeds a threshold is listed. If many are listed, the memory is full,
  so add a new slot table; every lookup then reads from it too. Then re-solve.

**What it guarantees.** Every stored fact reproduces its target for its stored wordings within a
set tolerance, or it appears on a list the system acts on the same night. Forgetting becomes a
measured quantity that is repaired, not a silent drift.

**Simulation** (4,096-slot tables, 4 wordings per fact):

| Facts | Tables | Constraints per slot | Facts flagged | Stored fidelity | After adding a table: flagged / fidelity |
|---|---|---|---|---|---|
| 500 | 1 | 0.49 | 0 | 0.9995 | – |
| 1,000 | 1 | 0.98 | 0 | 0.9921 | – |
| 2,000 | 1 | 1.95 | 1,992 | 0.9144 | **2 / 0.9905** |
| 3,000 | 2 | 1.46 | 2,156 | 0.9583 | **2 / 0.9893** |
| 4,000 | 3 | 1.30 | 1,261 | 0.9696 | **3 / 0.9884** |

- **The capacity point is predictable:** about one constraint per slot.
- **The check catches the overflow the night it happens.**
- **Growth restores fidelity**, and the memory grows with flash storage, not with retraining.

**Prior art:** not found in editing or LLM-memory work. The nearest are GRACE's codebook
splitting, RLSEdit's error monitoring, and classical resource-allocating networks (Platt 1991;
kernel RLS with dictionary growth, Engel et al. 2004).

### Fix 3: Canonical keys (for B3)

**What the simulation shows.** Whether a fact is found under an unseen wording depends almost
entirely on **how many slots that wording shares with the stored wordings**:
- 36% shared → 55–70% recall.
- 70% shared → 97–99% recall while the memory is lightly filled.
- Writing more wordings **does not fix it**: 4, 8 and 16 wordings per fact gave 58.8%, 64.7% and
  60.2%.

So coverage isn't the answer; the lookup itself has to be the same for every wording.

**How it works.**
- Before writing or reading, the model turns the question into a **canonical key**, a short
  internal `entity | relation` form (for example "Tim Cook | birthplace"), and the lookup is
  computed from that.
- Every wording of the same question then produces the same key, 100% slot overlap, so unseen
  wordings recover exactly what Fix 1 stored.
- Each fact is also written under its reverse key ("Mobile, Alabama | people born here") against
  the reversal curse, as in BIRD (prior art).

**Cost.** A few generated tokens before answering, similar to a short thought step.

**Risk.** Keys are only as consistent as the model's canonicalisation. That is a measurable
error rate (Experiment 2 below), and it replaces an unfixable one.

### Fix 4: Forward-only writes (for B4)

**How it works.** The value each fact must produce needs no gradient:
- Run the frozen model **with** the fact in context and **without** it.
- The difference in hidden state at the memory layer, at the question/answer position, is the
  target the memory must add.
- The phone's nightly work becomes **forward passes plus a sparse linear solve**. Both run on
  hardware phones already accelerate: the inference path and memory bandwidth.

**Prior art.** Dherin et al. ("Learning without training", 2507.16003) prove an exact rank-1 weight
update that reproduces an in-context result without the context and without a backward pass. Using
such differences as *persistent least-squares targets in a keyed memory* was not found.

**Limitation.** A difference taken at one layer need not reproduce the whole in-context behaviour.
Fix 5's multi-depth writes, or a vendor-trained target network like Doc-to-LoRA's (0.09–0.55 s per
document), are the fallbacks.

**Phone math** (3B base, Snapdragon 8 Elite Gen 5 class; assumptions stated):
- **Per fact, finding the target:**
  - 2 keys (forward and reverse) × 2 runs (with and without the fact) × ~40 tokens ≈ 160 prefill
    tokens.
  - At the measured 1,770 tokens/s that is **~0.1 s**; at ~5 W, **~0.5 J**.
  - The measured gradient route takes ~18 minutes per fact at 3.8 tokens/s, so this is about
    **10,000× faster**.
- **Nightly solve:**
  - Each fact has about 4 constraints × 128 slots (4 heads × top-32) = 512 slot reads.
  - At 100,000 facts: 5.1×10⁷ reads × 1,024-dim fp16 rows ≈ 100 GB per memory pass.
  - Two passes per iteration at ~80 GB/s ≈ **2.5 s per iteration**, so about 4 minutes for 100
    iterations.
  - At 1 million facts: about 40 minutes.
  - Starting from yesterday's solution cuts iterations; the simulation needed 60–120.
- **Storage:**
  - Memory layer: 2²⁰ slots × 1,024 × int8 ≈ **1.07 GB**, growing ~1 GB per extra table.
  - Constraint journal: about 4 × 512 B of slot indices and weights plus a 2 KB target ≈
    **4 KB per fact**: 0.4 GB at 100K facts, 4 GB at 1M.
- **Capacity:** at ~4 constraints per fact, one 2²⁰-slot table holds about **250,000 facts** exactly,
  about 7 years at 100 facts/day. Growth adds more when the certificate asks for it.

### Fix 5: Using new facts in multi-step reasoning (for B5)

Three parts:
1. **Chained single-step recall.** Canonical keys (Fix 3) let the model break a multi-step question
   into single steps: "Apple | CEO" → Tim Cook, then "Tim Cook | birthplace" → Mobile. Each step is
   a direct lookup.
   - If single-step recall is p, two steps succeed with about p² (0.97² ≈ 0.94).
   - That is arithmetic, not a measurement. Retrieval-based decomposition (MeLLo) is prior art for
     the approach; doing it with parametric recall was not found.
2. **Tied multi-depth reads.** The same memory is read at several depths (early, middle, late),
   with constraints written at each. A step that resolves late still finds the fact.
   - Sharing memory across depths is prior (Berges et al.).
   - Editing shallow and deep layers is prior: IFMET; Redundant Editing, +15.5 points on two-step
     questions.
   - The combination for facts written after deployment was not found.
3. **Nightly consequence writing.** Likely consequences of new facts are generated and written as
   extra facts (prior: SEAL, PropMEND). Capacity from Fixes 1–2 makes this affordable.

### Fix 6: Overriding strong old beliefs (for B6)

**How it works.**
- Instead of a fixed push, each conflicting fact gets a **margin constraint**: the memory must
  raise the new answer's score above the old one by a margin *scaled to how confident the model is
  in the old answer*.
- The constraint is linearised through the upper layers with a forward-mode derivative (a
  Jacobian–vector product, about 2–3 forward passes, still no backward pass). So it stays inside
  Fix 1's least-squares solve.
- Conflicts are also written at several depths (Fix 5), so the old answer is suppressed where it
  forms.

**Prior art.** Margin objectives exist but all train by gradient (KDPO, KLOD). The
fixed-push-vs-growing-confidence failure is documented (Override Gap: layer boosting reaches
71–72.5% on the deepest conflicts). Margin constraints inside a closed-form write were not found.

### B7, skills: not solved here

A memory layer stores lookups, not computations. The honest plan:
- Procedures that can be described in words ("to file this form: …") are stored as facts and
  recalled into step-by-step reasoning.
- New abilities go into isolated per-skill adapters that are trained, frozen and never overwritten.
  They are routed by canonical key and trained with on-policy self-distillation, which forgets less
  (RL's Razor).

This is design only, with no new result behind it.

---

## 3. What is guaranteed, what is designed, what is open

| Property | Status | Evidence |
|---|---|---|
| Same memory whatever order facts arrive in | **Guaranteed by construction** | The ridge problem has one unique solution |
| No loss of stored facts below capacity | **Confirmed in simulation and on a real model (test 3)** | Qwen2.5-0.5B, earliest facts: 100% at 3K–6K facts, 98.6% at 12K, 86.8% at 24K (past the reachable-slot capacity); batch editing 8.4%, delta rule 2.2% |
| Exact deletion | **Confirmed in simulation; to chance level on a real model (test 3)** | Deleted facts 86% → 6.9% (a random answer from the relation's list is right 5–7% of the time); kept facts unchanged |
| Capacity overflow detected and repaired | **Detection confirmed on a real model; repair confirmed in simulation** | Test 1: the certificate flagged 99.9% of constraints the night storage failed |
| Base model's knowledge untouched | **Guaranteed for the weights** (frozen) | Unrelated behaviour still has to be measured, because memory outputs can fire on unrelated questions |
| Found under any wording | **Confirmed for known relations (test 3)** | Unseen wordings 96.5%, equal to the relation classifier's accuracy; 6% (test 1) → 44% (test 2) → 96.5% |
| Better than pasting facts into the prompt | **Confirmed on a 0.5B model (test 3)** | Same questions: 10 / 100 / 1,000 facts in the prompt 43% / 27% / 13%; one selected fact 92%; memory with 24K facts 83% |
| Learning without a backward pass | **Works for closed-vocabulary answers (test 2)** | An offline per-answer codebook gives 96.0% recall vs 96.2% for per-fact gradient targets; in-context differences (7.7%) and output-embedding directions (22%) fail |
| Base model's known facts untouched | **Confirmed with the novelty gate (tests 2 and 3)** | 44 of 44 known facts kept at every night; never-written people: 0% false reads in test 3 |
| Multi-step use of new facts | **Designed, untested** (Fix 5) | Chained recall: p² by arithmetic; multi-layer editing +15.5 points (prior) |
| Overriding strong beliefs | **Designed, untested** (Fix 6) | Failure cause documented; margin writes are new |
| Skills | **Open** | – |

The simulations use synthetic lookups and targets, not a language model. They confirm the storage
mathematics: erosion, capacity, deletion, growth. They cannot confirm anything that depends on how
a real model represents questions.

---

## 4. Real-model test 1: Qwen2.5-0.5B on a laptop GPU

**Setup** (`reallm/memtest.py`; results in `results/reallm/summary.md`; 52 minutes on an RTX 3050 laptop GPU):
- **Model and memory.** Frozen Qwen2.5-0.5B plus an empty memory: 65,536 slots after block 14 of
  24. The memory's output is added at the question's last position. At the start, the model is
  exactly unchanged.
- **Facts.** 24,000 made-up facts: 3,000 invented people × 8 relations (city, job, colour, pet,
  instrument, sport, language, car). They arrive 3,000 per night.
- **Wordings.** Each fact is written under 4 wordings. Two further wordings per fact are used only
  for evaluation; nothing else ever sees them.
- **Targets.** Each fact's target is the vector that makes the frozen model say the answer, found
  by 25 gradient steps.
- **Key.** The model's hidden state at the question's last position, block 14. It is projected by a
  linear map fitted once on 300 separate calibration people.
- **Metric.** Is the model's next token the answer?

**First night's facts, written wordings, after each night:**

| Facts written | Constraints per slot | Delta rule (gradient family) | Nightly batch least squares | Joint least squares (Fix 1) |
|---|---|---|---|---|
| 3,000 | 0.18 | 6.8% | 27.5% | **35.9%** |
| 6,000 | 0.37 | 7.7% | 9.5% | **24.0%** |
| 12,000 | 0.73 | 9.6% | 7.7% | **18.3%** |
| 24,000 | 1.47 | 8.2% | 8.8% | **13.2%** |

- Without the memory, the model gets 2.5% right.
- Adding each fact's target directly at the last position gives **100%**, on written *and* unseen
  wordings. What to store works; where it is stored is the problem.

**What held:**
- **The ordering the simulation predicted.** The joint solve is best at every night. Nightly batch
  editing collapses after its second night (27.5% → 9.5%), the MEMIT-style failure. Sequential
  delta-rule writes barely store at all.
- **The certificate (Fix 2) did its job.** On night 1 it flagged 99.9% of constraints as more than
  20% off: the failure was reported the night it happened, not discovered later.

**What failed, and why:**
1. **The key did not tell people apart.**
   - The joint solve recalled only 36% at 0.18 constraints per slot, far below capacity; the
     simulation stored facts exactly there.
   - Measured slot sharing: different people with the same relation shared **60%** of their slots.
     Different relations shared 0.4%.
   - The key map's strongest directions separate relations; their discriminant ratios are 5,902,
     3,380 and 1,934.
   - Rescaling the directions to spread people apart cut sharing to 0.3%. But then a fact's own
     written wordings found only 14% of each other's slots.
   - The state at the question's last position simply does not carry who the person is. This
     matches how transformers recall facts (Geva et al. 2023; Meng et al., ROME): a subject's
     identity is assembled at the subject's own last token in early-middle layers. The final
     position then pulls attributes out of it, and an invented person has none to pull.
2. **Unseen wordings: 5–6% for every method,** near the base rate. Their keys found 25% of the
   written wordings' slots.
3. **Damage when the memory is read at every position.**
   - With the joint solve, known facts the base model answers correctly (37 of them) fell to 38–46%.
   - On unrelated text, the change in next-token predictions (KL divergence) was 0.37–0.55 per
     token. "Output nothing here" constraints halved that, to 0.18–0.23.
   - The values grew large (up to 64) because the solver was trying to satisfy colliding
     constraints.
4. **Deletion was not exact.**
   - With colliding keys, the re-solve did not converge in 600 iterations. The result differed from
     a never-learned memory by 2.3, with values up to 64.
   - Deleted facts fell from 13.7% to 8%, not to the 2.2% base rate.
5. **Forward-only targets (Fix 4) failed.** The in-context-minus-no-context difference, at its best
   scale, made the model say the answer for 7.7% of facts. The gradient target does so for 100%.

**Conclusion.** On a real model, the storage mathematics is not the bottleneck; the address is. The
sparse-memory results in the simulation assumed each fact gets its own slots. With keys read from
the question's last position, that assumption is false for the case that matters most: new facts
about new people.

**Test 2** (`reallm/memtest2.py`) changes three things:
- **The key.** The person half of the product key comes from the subject's last token, early in the
  network. The relation half comes from the question's last position. So each slot is a
  (person, relation) pair: Fix 3's canonical key, read from the model's own states rather than
  generated text.
- **A novelty gate.** The memory is read only when the person's key is close to one already
  written. This follows GRACE's deferral radius.
- **Two targets that need no backward pass on the device.**
  - The answer token's output-embedding direction.
  - A per-answer codebook of gradient targets, computed once offline on other people.

It uses the same 24,000 facts.

### Real-model test 2: person × relation keys

(`reallm/memtest2.py`; results in `results/reallm2/summary.md`; 57 minutes on the same laptop.) The
person half of the key is taken from the subject's last token at block 3. It was chosen
automatically from blocks 3, 5, 7 and 9, using written wordings only.

**Slot sharing, before and after:**

| | Test 1 | Test 2 |
|---|---|---|
| Different people, same relation: shared slots | 60% | **3.5%** |
| A fact's left-out written wording: its slots found | 89% | 96% |
| Unseen wording: its slots found among the written ones | 25% | **70%** |

**First night's facts after each night (joint least squares):**

| Facts written | Written wordings | Unseen wordings | Nightly batch LS, written | Delta rule, written |
|---|---|---|---|---|
| 3,000 | **96.2%** | **43.7%** | 85.5% | 6.8% |
| 6,000 | 89.6% | 28.0% | 13.6% | 3.7% |
| 12,000 | 72.0% | 18.7% | 10.1% | 1.3% |
| 24,000 | 51.8% | 11.1% | 9.3% | 3.5% |

**Damage and gate.** The gate reads the memory only when the person's key is close to one already
written.
- The 44 facts the base model knows stayed **100%** correct at every night with the gate. Without
  it they fell to 2–7%.
- For never-written people, the memory changed the answer 19–26% of the time with the gate, against
  96–99% without it.
- The gate itself let 20–27% of never-written people through, and passed 86% of unseen wordings of
  written people.

**Deletion.** Deleted facts fell from 53% to 7.8%; the base rate is 2.2%. Kept facts were unchanged
(49.4% → 49.5%). The remaining gap comes from a solver that stopped at its iteration cap: the
difference from a never-learned memory was 1.3, against values up to 95.

**Targets without a backward pass on the device:**
- **Per-answer codebook.** Gradient targets were computed once, offline, on the calibration people
  and averaged per answer. Their cosine with each fact's own gradient target is 0.91.
  - Injected directly: 99.95% (written wordings), 100% (unseen).
  - Through the memory: **96.0%**, against 96.2% for per-fact gradient targets; 45.8% against 43.6%
    on unseen wordings.
  - So on the device, writing a fact becomes a table lookup plus forward passes and a linear solve.
  - Caveat: this holds here because every answer is one token from a known list. Open-ended,
    multi-token answers are untested.
- **The answer token's output-embedding direction:** 22% at its best scale. It fails.

**What remains, and why:**
1. **Recall falls from 96% to 52% as facts go from 3,000 to 24,000.**
   - Both halves of the key had 256 codes, but only 8 relations exist. The facts therefore reached
     only 20,902 of the 65,536 slots.
   - That puts the effective load above one fact per reachable slot, where the simulation predicted
     degradation.
   - The solver also stopped at its 300-iteration cap every night. The certificate flagged it:
     73% of constraints were more than 20% off on night 1, and 99.9% by the end.
2. **Unseen wordings: 44% on night 1.** Injecting the target directly works on them 100% of the
   time, so the loss is in the key. A name in the middle of a new sentence has a different state at
   its last token, and 30% of its slots move.
3. **The gate lets 20–27% of never-written people through.** A key from the name's last token mostly
   reflects its last word piece, and many invented surnames share one.

**Test 3** (`reallm/memtest3.py`, running now) targets each cause:
- **Person key.** The name is encoded on its own, averaged over all its tokens. It is then snapped to
  the nearest person already written, or the memory stays silent.
- **Relation.** The relation is snapped to the nearest known relation.
- **Key split.** The key is split 1,024 person codes × 64 relation codes, so all slots are reachable.
- **Solver.** The solver is preconditioned.
- **Context comparison.** The same questions are answered with 10, 100 or 1,000 facts in the context,
  with only the best-matching fact selected into it, and from the memory alone.

### Real-model test 3: canonical, snapped keys; context vs memory

(`reallm/memtest3.py`; results in `results/reallm3/summary.md`; 62 minutes on the same laptop. The
laptop session had to chunk one step to fit in memory; that change was made on the laptop and is not
yet in this repo.)

**The keys:**
- **Person key.** The person's name, encoded on its own and averaged over its tokens, at block 7. It
  is whitened, then snapped to the nearest person already written, if the cosine is at least 0.981;
  otherwise the memory stays silent.
- **Relation key.** The question's last-position state, classified to the nearest of 8 relation
  centroids.
- **Addressing.** 1,024 person codes × 64 relation codes.
- **Solver.** Preconditioned conjugate gradient.

**First night's facts after each night:**

| Facts written | Joint LS, written | **Joint LS, unseen wordings** | Nightly batch LS, written | Delta rule, written |
|---|---|---|---|---|
| 3,000 | 100.0% | **96.5%** | 100.0% | 37.9% |
| 6,000 | 100.0% | 96.4% | 49.9% | 15.2% |
| 12,000 | 98.6% | 94.1% | 17.7% | 4.7% |
| 18,000 | 93.5% | 86.8% | 9.2% | 2.4% |
| 24,000 | **86.8%** | **78.7%** | 8.4% | 2.2% |

For comparison, the earliest facts at 24,000 recalled 13.2% in test 1 and 51.8% in test 2 on
written wordings; 5.1% and 11.1% on unseen ones.

**What the numbers say:**
1. **Wording no longer matters.**
   - Unseen wordings are recalled at 96.5%, and the relation classifier is right on 96.5% of unseen
     wordings. Every remaining unseen-wording error is a relation misread; none come from the memory
     or the person key.
   - The route to unseen wordings was: last-position key 6% → person-token key 44% → canonical,
     snapped key 96.5%.
2. **The joint solve keeps old facts.**
   - With good keys, nightly batch editing also stores each night exactly (100% on its own night).
     But it erases earlier nights: 100% → 49.9% after one more night, 8.4% by the end.
   - The joint solve keeps them: 86.8% at 24,000 facts. This is the simulation's central
     prediction, now on a real model.
3. **The gate is now clean.**
   - 0% of never-written people passed the gate, and none of their answers changed (test 2: 20–27%
     passed).
   - All 44 known facts stayed correct.
   - 5 of 2,999 people were wrongly merged with a similar name at this threshold.
4. **Deletion reaches chance level.**
   - Deleted facts fell from 86% to 6.9%. The memory still returns other facts' answers for the
     deleted pair's slots, and a random answer from a relation's list of 13–27 words is right about
     5–7% of the time. So 6.9% means the deleted information is gone; the no-memory base rate is
     2.2%.
   - Kept facts were unchanged (87.5% → 88.4%).
   - Difference from a never-learned memory: 0.21, against values up to 120.
5. **Writing without a backward pass holds.** The offline per-answer codebook gives 99.95% on
   written wordings and 96.5% on unseen, identical to per-fact gradient targets.

**Context vs memory** (100 first-night facts, each asked with an unseen wording):

| Setup | Accuracy |
|---|---|
| No context, no memory | 1% |
| All 10 facts pasted into the prompt (103 tokens) | 43% |
| All 100 facts pasted (974 tokens) | 27% |
| All 1,000 facts pasted (9,633 tokens) | **13%** |
| Only the best-matching fact selected into the prompt (any N) | **92%** |
| Memory holding all 24,000 facts, no prompt context | **83%** |

- Adding facts to the prompt cut accuracy from 43% to 13%. The model saw the answer every time.
- Selecting the one fact first held 92% regardless of pile size.
- The memory, with 24 times more facts than the largest prompt and no context at all, scored 83%.
- The absolute in-context numbers are low partly because the model is small (0.5B), and larger
  models retrieve better in context. The trend, and the gap between blending and selecting, is the
  finding.

**What remains, and why:**
1. **Recall still declines past about 9,000 facts.**
   - The facts reached only 21,462 of the 65,536 slots. With just 8 relations, each relation uses a
     few of its 64 relation codes, so each relation has about 3,000 slots for about 3,000 facts.
     That is capacity, and the certificate says so: 95% of constraints are more than 20% off by the
     end.
   - The fix is to bind the relation to the person key instead of taking a product with it. Rotate
     the person key by a relation-specific random rotation, the role-filler binding of holographic
     reduced representations (Plate, 1995), before the lookup. Every (person, relation) pair then
     spreads over the whole table.
2. **The address is closer to a structured key-value store.**
   - The system is told where the name is, and it knows the 8 relations in advance. Real use needs
     entity detection and an open, growing set of relations; growth would work as in Fix 2.
   - What the language model contributes: fuzzy matching of names, recognising the relation from
     free wording (96.5% on unseen wordings), and the value that makes the frozen model say the
     answer.
3. **Answers are one word from a known list.** Open, multi-word answers and multi-step use are
   untested.

## 5. Tests that decide it, in order

| # | Test | Pass criterion |
|---|---|---|
| 1 | Fixes 1–2 inside a real memory-layer LM (1B-class, e.g. the Memory Layers at Scale / sparse-memory-finetuning setup): stream 100,000 synthetic personal facts plus 10,000 real post-2024 facts with joint solves and certificate growth | Earliest 1,000 facts keep ≥95% of first-day no-context recall at 100K facts; unrelated QA (NaturalQuestions) within 1 point of the base |
| 2 | Canonical-key consistency on messy real questions (Fix 3) | Same key for ≥95% of paraphrases; unseen-wording recall within 3 points of stored-wording recall |
| 3 | Forward-only targets (Fix 4) vs gradient-computed targets | ≥90% of the gradient target's recall, with no backward pass |
| 4 | Search-once, 90 simulated days: re-asked questions in fresh sessions | Repeat-search rate for unchanged facts <5%; no-context accuracy within 10 points of retrieval |
| 5 | Two- to four-step questions via chained canonical recall (Fix 5) | Two-step accuracy ≥ 0.9 × (single-step)² |
| 6 | Strong-prior conflicts with margin writes (Fix 6), KID-Bench style | ≥80% on the strongest-prior quartile (published best: 71–72.5%) |
| 7 | Memory-layer solve on a phone, measured | 100K-fact nightly solve ≤30 min at ≤5 W |

Test 1 needs a GPU-class machine (the memory-layer LM does not fit this sandbox's CPU budget). The
code for Fixes 1–2 is in `memsim/`, and the maths carries over unchanged. Tests 2 and 3 can run on
any open 1–3B model.

---

## Reproduce the simulations

```bash
python3 memsim/sim.py 60000 5000 0.3   # erosion vs joint solve up to 60K facts (about 10 min)
python3 memsim/sim.py 4000 2000 0.6    # low slot overlap between wordings
python3 memsim/extras.py               # deletion, certificate + growth, more wordings per fact
```

Results: `memsim/result_60000_5000.json`, `memsim/extras_result.json`.
