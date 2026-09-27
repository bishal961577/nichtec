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
- **First real-model test (Qwen2.5-0.5B, section 4): the design failed as specified.** The storage
  maths behaved as predicted relative to the other write rules, but the lookup key did not tell
  people apart, so even the joint solve recalled only 36% of facts at 18% of capacity. The
  forward-only targets of Fix 4 also failed. Test 2 changes the key; its results are pending.

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
| No loss of stored facts below capacity | **Confirmed in simulation; not achieved on a real model (test 1)** | Simulation: stored fidelity 1.000 / 0.999 / 0.992 at 5K / 10K / 15K facts. Qwen2.5-0.5B: 36% recall at 0.18 constraints per slot, because the keys collided (section 4) |
| Exact deletion | **Confirmed in simulation; not exact on a real model (test 1)** | Simulation: differs from never-learned by ≤0.00018. Real model: the re-solve did not converge when keys collided |
| Capacity overflow detected and repaired | **Detection confirmed on a real model; repair confirmed in simulation** | Test 1: the certificate flagged 99.9% of constraints the night storage failed |
| Base model's knowledge untouched | **Guaranteed for the weights** (frozen) | Unrelated behaviour still has to be measured, because memory outputs can fire on unrelated questions |
| Found under any wording | **Failed with last-position keys (test 1); person × relation keys in test 2** | Test 1: unseen wordings 5–6%, near the 2% base rate |
| Learning without a backward pass | **In-context difference failed (test 1); two replacements in test 2** | Test 1: 7.7% of facts said the answer with the forward-only target, against 100% for the gradient target |
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

It uses the same 24,000 facts. Results are pending.

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
