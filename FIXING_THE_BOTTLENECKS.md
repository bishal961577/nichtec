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
  - **Test 4: real Wikidata edits (MQuAKE), with nothing handed to the system at question time.**
    - The system finds the subject itself and classifies the relation itself. Answers are open and
      multi-word.
    - With the fixed calibration, after all 2,764 edits: 97.3% on the written form, 91.0% on the
      question form. A third run reproduced every figure exactly.
    - Unedited facts: 98.3% unchanged (1,071 facts), 92.3% for other facts about edited subjects.
      Every change comes from the lookup matching a wrong fact, so locality depends only on the lookup.
    - The first night's edits held at 97.0%.
    - The remaining question-form misses equal lookup misses (91.0%).
    - Multi-hop questions cannot be judged on the 0.5B model: even unedited, it answers only 2.7–3.0%
      of them.
  - **Test 5: GPT-J-6B, the model of the published MQuAKE results, with pass lines fixed before the run.**
    - Single-hop passed every line: 95.9% written form, 95.7% question form, earliest edits kept at 97.0%,
      unedited facts 98.7% unchanged. Every unwanted change was a name collision in the lookup ("Francis"
      inside "Francis II").
    - Multi-hop: 9.4%, beating the published weight editors on the same model and data (MEMIT 5.4%,
      MEND 6.1%) but not MeLLo (14.2%). On the real-world changes of MQuAKE-T: 18.8% against MEMIT 0.0% and
      MeLLo 30.7%.
    - Using new facts in reasoning, relative to the model's own facts: 0.40 on MQuAKE-CF (fail), 0.82 on
      MQuAKE-T (pass). Two of my predictions (beating MeLLo; 0.6–0.9) were wrong.
  - **Test 6: the model's own states do not address the memory reliably.**
    - Every token querying the memory finds 62.8% (0.5B, closed) and 78.4% (1.5B, in between) of stored names in
      unseen sentences, against a 90% line.
    - With the name's position given, 98–99%, but only as a lexical fingerprint of its tokens.
    - The middle entity of two-hop questions is not readable from the question's last token (2.8–4.0%; line 20%),
      though the probe's own control is weak in the middle layers.
    - The pre-registered next step (a trained canonicaliser) and the lexical-table route the results point to are
      published 2026 work: Engram, NGM, TF-Engram, User as Engram, ENGRAFT.
  - **Open:**
    - Multi-hop use of new facts: the memory is missed when the model words the step itself.
    - Test 3's capacity limit (a third of the slots reachable) is gone in test 4: the relation is bound
      into the person key, so the whole table is used. It has not yet been tested past 2,764 facts.

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
| Better than pasting facts into the prompt | **Confirmed on a 0.5B model (tests 3 and 4)** | Same questions: 10 / 100 / 1,000 facts in the prompt 43% / 27% / 13%; one selected fact 92%; memory with 24K facts 83%. Real MQuAKE edits, question form: memory 80.4%, the matched fact pasted into the prompt 9.7% |
| Real-world edits, subject and relation not given | **Works on 0.5B (test 4) and 6B (test 5)** | 2,764 Wikidata edits. Qwen-0.5B: 97.3% written form, 91.0% question form, first night kept at 97.0%, unedited 98.3% unchanged. GPT-J-6B: 95.9% / 95.7%, 97.0%, 98.7%; GRACE-style 57.5% / 50.9%, unedited 72.3% |
| Model finds the memory itself | **Not from frozen states (test 6)** | Every token querying: 62.8% (0.5B) / 78.4% (1.5B) of stored names in unseen sentences, line 90%; position given: 98–99%, lexical only; longer names containing a stored one fire 30–58% |
| Multi-hop use of new facts | **Beats weight editing, not MeLLo (test 5, GPT-J)** | MQuAKE-CF, 3,000 edited cases: 9.4% vs MEMIT 5.4%, MEND 6.1%, MeLLo 14.2%; 0.40 of the model's own multi-hop rate. MQuAKE-T: 18.8% vs MEMIT 0.0%, MeLLo 30.7%; 0.82 |
| Learning without a backward pass | **Works for closed-vocabulary answers (test 2)** | An offline per-answer codebook gives 96.0% recall vs 96.2% for per-fact gradient targets; in-context differences (7.7%) and output-embedding directions (22%) fail |
| Base model's known facts untouched | **Confirmed (tests 2, 3 and 5)** | 44 of 44 known facts kept at every night; never-written people: 0% false reads in test 3. Test 5 (GPT-J): 0 of 36 known facts about never-edited subjects changed; the 5 that changed are all edits MQuAKE itself asks for (the capital of Japan becomes Bondi Junction, ...) |
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
first attempts ran out of GPU memory on the 9,633-token prompts of the context comparison. The fix,
now in the repo, feeds long prompts through the model in 1,024-token chunks with a key-value cache;
this gives the same attention, up to rounding.)

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

### Real-world test 4: MQuAKE, real Wikidata edits, nothing handed to the system

(`reallm/memtest4.py`; results in `results/reallm4/`; Qwen2.5-0.5B on the same laptop, 73 minutes.)

**Setup:**
- **Data.** MQuAKE-CF-3k-v2 (Zhong et al., EMNLP 2023; 2024 fixed version): 2,764 distinct real
  Wikidata fact edits over 37 relations, with open, multi-word answers ("Fernando Santos is a citizen
  of" → "United Kingdom"). They are written 500 per night.
- **What the system is given.** Each edit request as the dataset states it: subject, cloze sentence,
  question, new answer.
- **What it is not given at question time.**
  - Where the subject is: every word span of the question is encoded and matched against the
    subjects written so far.
  - Which relation is asked: a classifier fitted on separate calibration cases decides.
  - Whether any fact applies at all: if nothing matches, the memory stays silent.
- **Addressing.** The relation is bound into the person key by a relation-specific rotation. Test 3's
  capacity limit goes away: 44,699 distinct slots were used for 2,701 facts.
- **Answers.** Generated freely and scored by string match against the answer and its aliases, as in
  MQuAKE.

**Every edit after the last night:**

| Method (same model, data, scoring) | Written form (cloze) | Question form |
|---|---|---|
| Unedited model | 2.4% | 0.7% |
| **Joint memory (this project)** | **95.3%** | **80.4%** |
| Same slots, nightly batch least squares (MEMIT-style) | 93.2% | 78.1% |
| GRACE-style codebook (published lifelong editor) | 64.0% | 54.7% |
| The matched fact pasted into the prompt (retrieval) | 88.5% | 9.7% |

**Earliest facts:** the first night's edits after each later night:

| Edits | Joint | Batch | GRACE |
|---|---|---|---|
| 500 | 97.3% | 96.7% | 58.3% |
| 2,764 | 96.3% | 90.0% | 58.3% |

**Unedited facts:** the model's answer to 1,087 unedited facts from the same cases stayed identical
94.6% of the time with the joint memory. The other methods scored 83.9% (retrieval) and 30.8%
(GRACE).

**What the numbers say:**
1. **Real-world editing works without being told the subject.**
   - Injecting the stored value directly gives the new answer 99.7% of the time on both forms.
   - The lookup, from the question alone, found the right fact 80.0% of the time. The question-form
     score is 80.4%, so almost every question-form miss is a lookup miss.
2. **The lookup misses have a known cause, which is my error.** The calibration split excluded any
   case sharing any entity with the test, even common answers such as "United Kingdom". That left 44
   cases covering 24 of the 37 relations, so the relation classifier never saw 13 of the relations it
   was tested on. The same cause explains two locality figures:
   - 72.7% on unedited relations of edited subjects;
   - 86.3% on well-known facts, where the classifier routed some known-fact questions to an edited
     relation.
   Excluding only cases that share a subject gives 556 calibration cases covering 36 relations. The
   rerun with this fix is below.
3. **Old edits hold.** The joint memory kept the first night at 96–97% throughout, while batch editing
   slid from 96.7% to 90.0%. The gap is smaller than in test 3 because 2,701 facts use a small part of
   the capacity.
4. **Pasting the fact into the prompt fails on questions for this small model.**
   - It works on the cloze form (88.5%), which it can copy from, but not on the question form (9.7%).
   - The memory answers the question form at 80.4%.
   - Larger models read context better, so this gap is expected to shrink with model size.
5. **Multi-hop questions: no conclusion.**
   - With no edits at all, this 0.5B model answers only 2.7% of the multi-hop questions correctly
     through the sub-question chain. The MQuAKE paper reports 40.5% for GPT-J.
   - Every method scored 0.4–0.9%, which is floor level.
   - This test needs a model that can decompose questions: GPT-J, as in the published comparison
     (MeLLo 14.2%, MEMIT 5.4% at 3,000 edits). A 6 GB laptop cannot run it.

**Rerun with the fixed calibration** (`results/reallm4b/summary.md`; 37 minutes):
- **Calibration.** 556 cases, 36 relations, 2,246 training texts for the relation classifier.
- **Same edits, model, scoring and baselines** as the first run.

| After all 2,764 edits | Written form | Question form | Right fact found from the question alone |
|---|---|---|---|
| Unedited model | 2.4% | 0.7% | – |
| **Joint memory, first run** | 95.3% | 80.4% | 80.0% |
| **Joint memory, fixed calibration** | **97.3%** | **91.0%** | **91.0%** |
| Nightly batch least squares | 94.1% | 87.7% | 91.0% |
| GRACE-style codebook | 64.0% | 54.7% | (uses no lookup) |
| Matched fact pasted into the prompt | 90.2% | 10.9% | 91.0% |

- **Earliest facts.** The first night's edits went from 98.7% to 97.0% across all six nights with the
  joint memory. With batch editing they went from 98.3% to 92.7%.
- **Unedited facts.** 96.9% of the 1,087 unedited facts gave identical output (first run: 94.6%).
- **What remains.** The question-form score equals the lookup's accuracy exactly (91.0%). Every
  remaining miss is the lookup picking the wrong fact or none: subject matching, or relation
  classification. When the lookup is right, the stored value works.
- **Two measurements were wrong and are now fixed in the code.**
  1. MQuAKE applies all edits together, but the "unedited facts about edited subjects" set (55 facts)
     included 16 facts that another test case edits. One example is "The official language of
     Helsinki is". Changing those answers is correct. So the reported 65.5% (first run 72.7%) is not
     a clean measure of damage. It is at worst 51% and at best 92% on the 39 truly unedited facts;
     the per-fact outputs needed to settle it were not saved.
  2. The known-facts check compared against base outputs generated in different batches. All five
     methods scored exactly 94.1%, including ones that change nothing. That points to bf16 rounding
     from padding, not to damage. The base outputs are now regenerated in the same batches.
- **Multi-hop** (200 cases, one question each). The unedited model gives the new answer 0.5% of the
  time and the original answer 3.0%. With the joint memory the chain gives the new answer 3.0% of the
  time. That is as often as the unedited model uses its own knowledge, but both are 6 cases out of
  200. It is floor-level evidence and settles nothing; that needs a GPT-J-class model.

**Third run, with both locality measurements fixed** (`results/reallm4c/summary.md`; 27 minutes):
- **Reproducible.** Every editing number and the multi-hop numbers are identical to the second run to
  the last digit: 97.3% / 91.0%, first night 98.7% → 97.0%.
- **Unedited facts.** 1,071 facts remain after removing 43 that another case edits.

| Unedited facts, identical output | All 1,071 | Other relations of edited subjects (39) | Known facts (51) |
|---|---|---|---|
| **Joint memory** | **98.3%** | **92.3%** (36 of 39) | 96.1% (49 of 51) |
| Nightly batch least squares | 98.3% | 92.3% | 96.1% |
| GRACE-style codebook | 86.2% | 74.4% | 96.1% |
| Matched fact pasted into the prompt | 94.4% | 89.7% | 84.3% |

- **Joint and batch agree to four digits on every locality figure.** This follows from the design.
  - When the lookup matches no fact, the memory adds nothing and the output is exactly the base
    model's.
  - So every change to an unedited fact comes from the lookup matching a wrong fact. The write rule
    only decides what that wrong match adds.
  - Locality is therefore a property of the lookup (subject matching and relation classifier), not of
    the memory. Improving it means improving the lookup.
  - The 3 of 39 changes on edited subjects fit the relation classifier sending an unedited relation of
    that subject to its edited one.
- **The known-facts check is contaminated by the benchmark itself.** MQuAKE edits some of the
  well-known facts used as the damage check.
  - Word for word: the capitals of Japan (to Bondi Junction), Italy (Duluth), Egypt (Yungay), Canada
    (Königsberg) and South Korea (Chiavari).
  - Same fact, other wording: the language of Germany, Italy and Japan ("Most people in Japan
    speak"), the author of Romeo and Juliet and the creator of the Mona Lisa.
  - Changing those answers is the edit working, not damage.
- **What the two changed known facts probably are.** The evidence is consistent, but the run did not
  save which facts changed, so this is not proven.
  - Each memory method (joint, batch and GRACE) changed exactly 2 of the 51, and each turned exactly 2
    right answers wrong (the base scored 50 of 51 in the same batches, each method 48).
  - GRACE finds facts with a different mechanism, a similarity gate. That gate changes about 8 times
    as many unedited facts (13.8% against 1.7%), yet on known facts it matches the joint memory exactly.
    That is what an exact match to a stored edit produces, and not what false matches would produce.
  - Unexplained: if all five word-for-word capitals were among the 51 the base answers correctly, the
    memory should have changed five, not two. The base model's per-prompt answers were not saved
    either.
- **Fixed in the code for the next run.** Each run now records every changed fact, known or unedited,
  with the base answer, the new answer and the fact the lookup matched. It also reports:
  - how often the lookup matches any fact;
  - damage on the known facts whose subject no case edits, so that figure cannot be caused by a
    correct edit.

### Real-world test 5 (pre-registered in commit ca50c62; results below): GPT-J-6B on one H100

(`reallm/modal_gptj.py` runs `reallm/memtest4.py` on a rented H100; results will go in
`results/reallm4_gptj/` and `results/reallm4_gptj_t/`.) This section was written and committed before the run.
The pass lines and predictions below are fixed and will not be moved after the results arrive.

**Why GPT-J.** It is the model the MQuAKE authors used. The current version of their paper uses the same data
file as this project (MQuAKE-CF-3k-v2; arXiv v3, Table 5). Their GPT-J results with all 3,000 cases edited at
once are the reference; a case counts if any of its three questions gets the new answer:

| Published, GPT-J, MQuAKE-CF-3k-v2, 3,000 edited cases | Multi-hop accuracy |
|---|---|
| MeLLo (explicit fact memory with retrieval and self-check) | 14.2% |
| MEND | 6.1% |
| MEMIT | 5.4% |
| Unedited GPT-J on the original answers (Table 3) | 40.5% |

On MQuAKE-T (1,868 cases built on 96 real-world changes): MeLLo 30.7%, MEND 4.6%, MEMIT 0.0%.

**What is fixed in advance:**
- GPT-J-6B, half-precision weights run in bf16. MQuAKE-CF-3k-v2, all 2,764 edits written 500 per night. All
  3,000 cases with 3 questions each.
- If the time budget stops multi-hop early, the cases done are a random subset (they run in a fixed random
  order). Every method is scored on the same cases, and the report states how many.
- Same code and settings as the 0.5B runs, with three changes:
  - the decomposition prompt uses all four of MeLLo's published examples, not two;
  - the few-shot prompt is encoded once and reused; tested to give identical outputs;
  - if the stored values reach the target on fewer than 80% of calibration edits, the target optimiser is
    retried once with longer, larger steps. This is decided on calibration data only.
- Multi-hop methods: the joint memory, the unedited model, and the matched fact pasted into each
  sub-question's prompt.

**Q1. Does the single-hop result carry over from 0.5B to 6B?**
- **Pass (all four):** written form ≥ 90%; question form ≥ 85%; first-night edits ≥ 95% after the last
  night; unedited facts unchanged ≥ 95%.
- **Prediction:** pass. The lookup and the stored values do not depend on model size, and the target was
  reached 99.7% of the time on 0.5B.
- **Main risk:** the target optimiser's settings were tuned on 0.5B; the retry is built in for this.

**Q2. Multi-hop against the published GPT-J editors.** Joint memory, sub-question chain, 95% interval over cases:
- **Beats MeLLo:** the interval's lower end is above 14.2%.
- **Beats weight editing only:** the lower end is above 5.4%, but not above 14.2%.
- **Fails:** the lower end is at or below 5.4%.
- **Prediction:** beats MeLLo, with about 60% confidence.
  - MeLLo falls from 38.9% (one case) to 14.2% (3,000 cases) because retrieval picks wrong facts once
    thousands are stored.
  - This lookup is keyed on subject and relation. At 0.5B it found the right fact 91% of the time with
    2,764 stored.
  - The main risk is GPT-J writing wrong sub-questions; Q3 separates that from the memory.

**Q3. Are facts from the memory used in reasoning as well as the model's own facts?**
- **Measure:** the joint memory's rate of new answers divided by the unedited model's rate of original
  answers, on the same cases with the same chain.
- **Pass: ≥ 0.8.** Facts read from the memory work nearly as well in multi-step reasoning as facts in the
  weights (Fix 5).
- **Fail: < 0.5.** Using new facts in reasoning is the open bottleneck.
- **Validity check:** the unedited model's chain must reach at least 20% on the original answers. Below that,
  the decomposition is too weak for Q2 to test the memory, and Q2 is reported as not decided.
- **Prediction:** 0.6–0.9.

**Q4. Memory vs pasting the fact into the prompt, at 6B.**
- At 0.5B, pasting failed on questions (10.9% against 91.0%). A 6B model reads context better.
- **Prediction:** the gap shrinks a lot; pasting reaches 50–85% on the question form.
- **If pasting matches or beats the memory** on both the single-hop question form and multi-hop, the
  accuracy advantage seen at 0.5B was a small-model effect. The case for the memory then rests on no
  context, cost and scale, not accuracy.

**Q5 (only if the budget leaves time).** The same questions on MQuAKE-T, against MeLLo 30.7% and MEMIT 0.0%.

**A known lookup failure, recorded before the run.** A CPU rehearsal with a tiny random GPT-J matched "Which
religion is Francis II affiliated with?" to the stored subject "Francis": a stored name that is part of a longer
name matches exactly. This is a property of word-span matching, not of model size. The run's per-fact
diagnostics will show how often it happens on GPT-J; the method is not changed before the run.

**What this cannot prove, whatever the result:**
- The relation classifier knows the 36 relations of its calibration data, a closed set.
- Edits arrive in the dataset's own cloze and question form.
- A 6B model on a datacentre GPU says nothing about power or speed on a phone.
- Facts are keyed by named subjects; facts about unnamed things are out of scope.

#### Test 5 results (`results/reallm4_gptj/summary.md`, `results/reallm4_gptj_t/summary.md`)

One H100 run of about 47 minutes: 36.8 on MQuAKE-CF-3k-v2, then 8.5 on MQuAKE-T with the budget that was left.
The target optimiser worked on the first try (calibration: 93.8% written form; stored values injected
directly: 100%), so no retry and no abort.

| Pre-registered question | Pass line | Result | Verdict | My prediction |
|---|---|---|---|---|
| Q1 single-hop carries over to 6B | written ≥ 90, question ≥ 85, first night ≥ 95, unedited ≥ 95 | 95.9 / 95.7 / 97.0 / 98.7% | **Pass** | Pass: right |
| Q2 multi-hop vs published GPT-J editors | lower end > 14.2% beats MeLLo; > 5.4% beats weight editing | **9.4%** [8.4–10.5] | **Beats MEMIT (5.4%) and MEND (6.1%); not MeLLo (14.2%)** | Beats MeLLo: **wrong** |
| Q3 memory facts in reasoning vs own facts (CF) | ≥ 0.8 pass, < 0.5 fail; valid if unedited chain ≥ 20% | 9.4% / 23.4% = **0.40** (validity 23.4%, met) | **Fail** | 0.6–0.9: **wrong** |
| Q4 memory vs pasting the fact | pasting must match on single-hop question form and multi-hop | Question form 95.7 vs 84.0%; multi-hop 9.4 vs 8.4% (overlapping) | Pasting caught up but not on both; the memory keeps a smaller single-hop lead | Gap shrinks, pasting 50–85%: right |
| Q5 MQuAKE-T (750 random cases of 1,868; budget) | same lines; MeLLo 30.7%, MEND 4.6%, MEMIT 0.0% | **18.8%** [16.2–21.8]; ratio **0.82** | Beats MEND and MEMIT, not MeLLo; **Q3 passes on T** | – |

**Single-hop, all 2,764 edits, GPT-J:**

| Method | Written form | Question form | First night after last night | Unedited facts unchanged |
|---|---|---|---|---|
| Unedited model | 1.0% | 0.7% | – | – |
| **Joint memory** | **95.9%** | **95.7%** | **97.0%** (from 97.7%) | **98.7%** |
| Nightly batch least squares | 82.3% | 77.5% | 77.7% (from 79.3%) | 98.8% |
| GRACE-style codebook | 57.5% | 50.9% | 55.0% (from 55.0%) | 72.3% |
| Matched fact pasted into the prompt | 91.0% | 84.0% | – | 98.3% |

The lookup found the right fact from the question alone 96.4% of the time (0.5B: 91.0%). On MQuAKE-T (96 real
changes) the joint memory scored 95.8% on both forms, and all 1,813 unedited facts were unchanged.

**What the numbers say:**
1. **Single-hop real-world editing holds at 6B.** Every pre-registered line passed, with the subject and relation
   found by the system.
   - Caveat: the question form is the dataset's own question, which the memory saw when writing the fact. New
     wordings are tested only by the multi-hop sub-questions.
   - Caveat: batch editing was already weak on night 1 (79.3%) and lost only 1.6 points after that. Its gap to the
     joint memory is mostly a worse fit, not forgetting; its anchor setting came from the 0.5B runs and was not
     re-tuned for GPT-J.
2. **The known-facts question is settled.** GPT-J answers 54 of the well-known facts correctly (it continues "The
   capital of France is" with "a city of contrasts", so several capitals were left out). The memory changed 5 of
   the 54:
   - the capitals of Japan, Egypt and South Korea to Bondi Junction, Yungay and Chiavari;
   - "Most people in Italy / Japan speak" to Walloon and Swedish.
   Every one is an edit MQuAKE itself asks for; the last two are the same fact in other words (official language).
   On the 36 known facts whose subject no case edits, **0 changed**.
3. **Every locality failure of the memory is a name collision in the lookup, as recorded before the run.** All 14
   changed unedited facts (1.3%) matched a stored name that is part of, or one character away from, the name in
   the question:
   - part of a longer name (10): Francis in "Francis II", Portal in "Portal 2", iPod in "iPod Classic", Xbox in
     "Xbox Live Indie Games", Ford Sierra in "Ford Sierra RS Cosworth", World Judo Championships in "2011 World Judo
     Championships" (three years), and others;
   - one character away (3): ARM Cortex-A8 / A9, Gears of War 3 / 2, Power Mac G5 / G4.
   The stored values and the solver caused none. Pasting the matched fact into the prompt suffers the same false
   matches, but GPT-J often ignores an irrelevant pasted fact ("Who is the developer of Gears of War 2?" stays Epic
   Games), which the memory cannot do.
4. **Multi-hop is the open bottleneck, as Q3 defines it.** On MQuAKE-CF the memory's chain gave the new answer
   9.4% of the time and the old, pre-edit answer 19.5%.
   - An old final answer needs every other hop right and the edited hop answered from the model's own knowledge.
     So in about two thirds of the chains that otherwise work, the memory did not override the edited step.
   - Pasting the matched fact fails the same way (8.4% new, 19.3% old) and uses the same lookup. This points to the
     lookup missing the edited fact when GPT-J words the sub-question itself, not to the stored values. This is an
     inference: per-step lookups were not logged.
   - The chain itself is weak. GPT-J often restates the whole question instead of splitting it, and answers steps
     in full sentences ("The country is the United States of America"). The unedited model's chain reaches 23.4% on
     the original answers, against 40.5% for GPT-J with direct few-shot prompting in the MQuAKE paper. MeLLo shows
     the model examples at every step; this chain answers each step from a bare question, where GPT-J gets 55% of
     unedited facts right.
5. **On MQuAKE-T the memory's facts are used almost as well as the model's own (0.82).** The contrast with CF (0.40)
   fits the lookup explanation:
   - most T cases are two hops ending in an edited "head of state / government" fact, which GPT-J asks in close to
     the template wording ("What is the name of the current head of state in South Korea?");
   - CF edits sit at any of 2–4 hops across 37 relations, often about an intermediate entity named in the model's
     own words.

**What this run establishes and what it does not.**
- **Established:** at 6B, on 2,764 real edits with nothing given at question time, the joint memory:
  - answers 95.9% (written) and 95.7% (dataset question);
  - keeps its earliest edits at 97.0%;
  - leaves 98.7% of unedited facts and all 36 never-edited known facts untouched.
  In multi-hop it beats the published weight editors on the same model and data (9.4% vs 5.4–6.1%; T: 18.8% vs
  0.0–4.6%).
- **Not established:** multi-hop at MeLLo's level (14.2% CF, 30.7% T). Using new facts in reasoning does not work
  as well as using the model's own facts on CF (0.40).

**Next, in order of cost:**
1. **Lookup name collisions:** only accept a match when the span cannot be extended into a longer name, and treat
   names differing in a digit or letter as different. This can be checked on the laptop.
2. **Log every chain step:** was the lookup hit, was it the case's edited fact, and did the value take effect.
   This decides between the lookup and value-transfer explanations. It needs another GPT-J run.
3. **Few-shot prompts for each chain step**, as MeLLo uses, so the chain itself is not the ceiling.

### Real-world test 6 (pre-registered; results below): can the model's own states address the memory?

(`reallm/keytest.py`; `reallm/remastered_check.py`; results will go in `results/keytest_*` and
`results/remastered_check/`.) This section was written and committed before the run.

**Why this test, before anything else.** Tests 1–5 trace the known triangle of lifelong editing: reliability,
generalisation and locality cannot all be had at once. They also add a fourth column, whether the model finds the
memory itself:

| Test | Reliable | New wordings | Local | Model finds it itself |
|---|---|---|---|---|
| 1: key from the question's last position | 36% | – | – | yes |
| 2: key from the person's words in the sentence | 96% | 44% | yes | yes |
| 3–5: name encoded alone, snapped, external lookup | 96% | 96.5% | 98.7% | no (two-hop, asked directly: 2.5%) |

Any design that makes new knowledge part of the model's own reasoning (consolidation, "one brain", continual
learning in weights) needs the last column and the middle ones together. The consolidation papers of 2026 (SCoL,
"Language Models Need Sleep", Cartridges) do not test this. So this is measured first, with forward passes only.

**Data.** The corrected MQuAKE-Remastered CF-3k, because the MQuAKE-Remastered audit (ICLR 2025) found 33–76% of
MQuAKE's questions and labels corrupted. `remastered_check.py` reports exactly what differs from the v2 file that
tests 4 and 5 used.

**What is measured** (Qwen2.5-0.5B, then Qwen2.5-1.5B, on the laptop):
- Each stored name gets a key; the queries are sentences about the same names that were never used for the keys.
- **Keys come in two kinds:** the name encoded on its own, or the name's state inside its write-time sentence.
- **The name's position:** first given, then not given. The native scan: every token queries the memory, and it fires
  where the best match passes a threshold.
- **Negatives:** names never stored, and longer names that contain a stored one ("Francis II" for "Francis").
- **Two-hop questions:** can a linear map fitted on calibration questions read the hidden middle entity (the country
  in "the capital of the country of citizenship of X") from the question's last position?
- **No leakage from the test set:** thresholds, layers and map settings are chosen on calibration names that share no
  subject with the test set. The calibration memory is padded with distractor names to the test memory's size.

**Pass lines:**
- **Native addressing viable:** native-scan hit ≥ 90% on unseen sentences, with ≤ 2% false fires on never-stored
  names.
- **Closed:** below 70%.
- **In between:** not viable as is; a learned canonicaliser is the next experiment.
- **Multi-hop hook exists:** middle-entity top-1 ≥ 50% on two-hop questions whose first hop the model answers
  correctly. None if below 20%.

**My predictions, stated before the run:**
- With the position given, a name inside a sentence stays close to the name encoded alone in the early layers. Hit
  80–95% at some early layer.
- The native scan will fall below 90%. The difficulty is not finding the name but staying silent on the other tokens,
  where words like "capital" may land near stored keys. Hit 60–85%, meaning "not viable as is".
- The middle entity will be weakly readable: 20–50% at 0.5B, higher at 1.5B.

**What each outcome means:**
- **Viable:** build consolidation on native keys, the SQuAD-stream experiment.
- **In between:** train a small canonicaliser on calibration names, then retest.
- **Closed:** the in-weights route to continual learning is closed for this design. The project then either becomes
  a retrieval-style memory or moves to the retrofitted memory-layer route, where a model is trained to address its
  own memory.


### Real-world test 6 results (Qwen2.5-0.5B and 1.5B, laptop RTX 3050, 30 Sep 2026)

(`results/keytest_Qwen2.5-0.5B_v2/`, `results/keytest_Qwen2.5-1.5B_v2/`; 29 and 31 minutes.)

**One deviation from the plan: the data.** The runs used MQuAKE-CF-3k-v2 (with MQuAKE-CF for calibration), not
Remastered. The Remastered loader failed on the dataset's real layout: each edit stores `target_new_str` flat, not
the nested `target_new` its README shows. The pre-agreed fallback to v2 was used. This does not change the
conclusions below:
- The name measurements use only subject names and sentences that contain them; no answer label is involved.
- The two-hop measurement uses the middle entity of each fact chain. Even if half of those labels were wrong, a
  true 50% would show as 25% or more, not 2–4%.

The loader is fixed (checked on a parquet built in the real layout); the Remastered comparison itself is still to run.

| Pre-registered measure (line) | 0.5B | 1.5B | Verdict |
|---|---|---|---|
| Native scan hit, best variant, layer chosen on calibration (viable ≥ 90% with ≤ 2% false fires; closed < 70%) | 62.8% (false fires 2.9%) | 78.4% (false fires 3.6%) | 0.5B **closed**; 1.5B **in between** |
| Longer names containing a stored one ("Francis II"), fired | 35–54% | 30–58% | – |
| Name position given, name's tokens averaged at layer 1 | 99.3% (false fires 3.1%) | 98.2% (2.3%) | – |
| Name position given, last token only | 59–78% | 68–86% | – |
| Middle entity of two-hop questions, top-1 where the first hop is known (hook ≥ 50%; none < 20%) | 2.8% (chance 0.06%) | 4.0% | **none** |

**Predictions against outcomes:**
- **Position given, 80–95% at an early layer:** 98–99%. Right direction, too low, and the high number is lexical
  (below).
- **Native scan 60–85%, "not viable as is":** 62.8% and 78.4%. Right.
- **Middle entity 20–50% at 0.5B, higher at 1.5B:** 2.8% and 4.0%. Wrong.

**What the numbers say:**
1. **A name's identity survives only where the representation is still lexical.**
   - Compare the name inside a sentence with the same name encoded alone. The two are 0.97–0.98 similar at
     layers 2–4, falling to 0.41 at the last layer.
   - From layer 22 (0.5B) and layer 26 (1.5B) on, a name inside a sentence is closer to some *other* name than to
     itself.
   - The deep layers, where the model's meaning is, carry the sentence's next-word prediction, not the name.
2. **The 99% with the position given is a lexical fingerprint.**
   - Averaging the layer-0/1 states over a name's tokens gives nearly the sum of its token embeddings.
   - It works because the query sentences contain the name verbatim. It would not find "the French president" for
     "Emmanuel Macron".
   - It needs the name's position marked, which means an external name finder: what tests 3–5 already use.
3. **Most of the loss is one token not identifying a multi-token name.**
   - Without a marked position, the scan must use single-token states. With the position given but only the
     last token, the hit rate is already down to 59–86%; scanning loses another 2–15 points.
   - The right name is usually the nearest one (top-1 90–97%). But with 2,580 stored names, a never-stored name
     lands close enough to some stored key that a 2%-false-fire threshold rejects many true matches (hit 59–86%).
   - Longer names containing a stored name fire 30–58% of the time. This is the name-collision failure behind
     every unwanted change in test 5, now seen in the representations.
   - Scale helped (62.8% → 78.4%). But two sizes make no trend, and collisions and false fires come from the
     keys being lexical, not from model size.
4. **Multi-hop: the probe found nothing, and the probe is partly to blame.**
   - Its control is reading the first subject, which is written in the question. It reaches 49–85% at the
     first or last layers but only 4–16% in the middle layers, which is where a hidden middle entity would be.
   - So "none" means that a linear map from the question's last token onto name keys does not find the middle
     entity. It does not show that the model never computes it.
   - Work with better probes already covers this: "Do Large Language Models Latently Perform Multi-Hop
     Reasoning?" (Yang et al., 2024) and "Hopping Too Late" (Biran et al., 2024) find partial hidden first hops
     and a late or failed second hop (B5).
   - My design error: a "none" verdict should have required the control to pass at the same layer.

**Decision.** By the pre-registered lines, 0.5B is closed and 1.5B is in between, whose next step was "train a
canonicaliser, retest". A prior-art check (30 Sep 2026) came first. That step, and the route these results point
to (the only reliable native key is lexical and close to the input), are already published:
- **Engram** (DeepSeek, arXiv 2601.07372, Jan 2026): hashed n-gram lookup tables inside the model, learned in
  pretraining.
- **NGM** (2605.16893, May 2026): training-free n-gram memory from averaged token embeddings. This is the same
  construction as the 99% key above.
- **TF-Engram** (2607.07388, Jul 2026): phrase memory built offline without training, to add knowledge.
- **Lngram** (2605.24869, May 2026): n-gram keys learned from hidden states; also injects knowledge after
  pretraining.
- **User as Engram** (2606.19172, Jun 2026): each user's facts written as edits to Engram rows. It reports about
  5.6× the indirect-reasoning accuracy of per-user LoRA, with about 33,000× less disruption to unrelated text.
- **ENGRAFT** (github.com/fulvian/engraft-ngram, Sep 2026): gradient writes to the n-gram table of
  Qwen3.8-Flash-Next.
  - 84.1% exact recall on 100 facts.
  - Paraphrases in another template fail.
  - Two-fact composition: both answers right in 4 of 83 probes.
  - 32 of 37 failures come from facts that share a subject.
- **Trained entity finder plus fact memory:** Entities as Experts (2020), Facts as Experts (2021) and KBLaM
  (ICLR 2025).

So the canonicaliser would reproduce published work, not produce a result. The problems that these systems and
tests 3–6 share, still unsolved, are:
- **other names for the same thing:** aliases and descriptions, because the keys are lexical;
- **name collisions:** ENGRAFT 32 of 37 failures; test 5's every unwanted change; 30–58% here;
- **using written facts together in reasoning:** ENGRAFT 4 of 83; test 5 9.4%; no readable middle entity here.

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
