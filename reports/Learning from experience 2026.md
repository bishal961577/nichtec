# Learning From Experience: Is There an Opening for a Small Team? (September 2026)

The goal's third part is "learns on its own and gets better at its work every time". The earlier report ranked this
the first bottleneck between current AI and AGI. This report checks the 2025–2026 work on it before anything is
proposed, and asks whether a team with a 6 GB laptop GPU and a small cloud budget could win anything there.

**Bottom line:**
- **The problem is real and open.** On CL-Bench, the 2026 benchmark built to measure exactly this, the best
  system captures 25.4% of what could be learned. Dedicated memory systems do worse than keeping the whole history
  in the prompt.
- **Every mechanism a solution needs already has several 2025–2026 papers:** experience memory, belief revision,
  learning into weights from feedback, self-modifying agents, and statistically gated self-updates.
- **The measured bottleneck is not storage.** It is the model's ability to work out the reusable lesson from
  experience. Even with the full history in its context, a frontier model gets a quarter of the way. That ability is
  set by the base model, which a small team cannot change without large compute.
- **Small models are specifically weak at this.** Dynamic Cheatsheet reports "more limited and inconsistent gains"
  for smaller models. Self-distillation from feedback reportedly does worse than plain RL at 1.5B.

This likely explains the poor results of the owner's earlier experience-card system. I found no opening of
breakthrough size that is both unclaimed and within this team's means.

## 1. What CL-Bench measures and finds

CL-Bench (UC Berkeley, Snorkel, UW-Madison; June 2026):
- **Tasks:** six expert-validated domains (software engineering, signal processing, outbreak forecasting, database
  querying, strategic games, demand forecasting). Each task is a sequence of tens of instances that share a hidden
  structure, such as a codebase's layout, a database's conventions or an opponent's strategy. The structure is not
  learnable from pretraining, and some tasks change halfway ("concept drift").
- **Metric:** gain, the reward with experience minus the same system's reward without it, as a fraction of the
  available headroom.

Results:

| System | Normalized gain | Cost per run |
|---|---|---|
| Full history in context, Claude Sonnet 4.6 | 25.4% (best) | $30.4 |
| Claude Code, Sonnet 4.6 | 23.9% (65.1% on Sales Prediction, 43.6% on Database Exploration) | $38.6 |
| Full history in context, Gemini 3 Flash | 16.4% | $7.6 |
| ACE (evolving playbook) | 8.6% | $62.8 |

The failures the authors name:
- Agents are "rigid in early, incorrect beliefs, and struggle to update despite feedback".
- They "overfit to immediate observations or fail to reuse knowledge across instances".
- "Memory modules introduce spurious generalizations and stale beliefs."
- On the epidemiology task no system extracts the structure; all are "misled by spurious correlations".

Two limits matter here:
- Methods that learn into the weights were not evaluated. The authors invite them.
- The tasks "require frontier-level capability", so small models' failure modes are not visible.

## 2. Every needed mechanism is already being worked on

| Mechanism | 2025–2026 work |
|---|---|
| Lessons kept as text and reused ("experience cards") | Dynamic Cheatsheet: AIME doubled for Claude 3.5 Sonnet, Game of 24 from 10% to 99% for GPT-4o, limited gains for small models. ReasoningBank (Google): up to 34.2% relative on web and coding agents. ACE. Evo-Memory (DeepMind). Test-Time Learning with an Evolving Library (May 2026). Procedural Memory Distillation (Jul 2026) |
| Belief revision: stale or wrong lessons | Belief Memory (May 2026; memories stored as single deterministic conclusions let errors persist). TOKI (Jun). TEPA, revoking stale memories (Aug). "Can Agent Memory Systems Track Evolving State?" (Aug). STALE benchmark. A survey, "From Memory to Belief" |
| Learning into the weights from feedback | SDFT (Jan): a model teaches itself from demonstrations; skills accumulate without regression at 7B. SDPO (Jan): the model conditioned on error feedback is its own teacher. SIEVE (Apr): parametric learning from three examples. SEAL (2025): the model writes its own training data but forgets over sequential edits. A predictive law for self-distillation from world feedback (May). Rethinking continual experience internalization (Jun). Denser ≠ Better: limits of self-distillation for continual post-training (Jul). From Self-Distillation to Self-Practice (Sep). Meta's Early Experience: +13 to +18 points over imitation |
| Agents that improve their own code | Darwin Gödel Machine (ICLR 2026): SWE-bench 20% → 50%. Follow-ups in Aug–Sep 2026 (Hierarchical Self-Improvement, Ouroboros, fast tree search). Live-SWE-agent |
| Accountability: accept a self-update only if it really helps | PACE (Jun 2026): anytime-valid acceptance tests, 0% false commits in its sweeps. Self-Evolving Agents with Anytime-Valid Certificates (Jul 2026) |

## 3. Why the owner's experience-card system gave poor results

The system stored a card each time the AI solved a problem and retrieved it for similar problems. The literature
predicts three failure causes, in order of likely weight:
1. **The model was probably too small to extract and use lessons.** Dynamic Cheatsheet's small models showed
   limited, inconsistent gains. SDPO reportedly falls behind plain RL at 1.5B.
2. **Cards store conclusions without their evidence,** so a wrong or outdated card is never revised. This is the
   failure CL-Bench names ("spurious generalizations and stale beliefs") and Belief Memory analyses.
3. **Retrieval by surface similarity** brings up cards for problems that look alike but need different
   strategies. ReasoningBank and ACE counter this by distilling strategies and tracking which bullets helped or
   hurt.

All three have published remedies (sections 2 and 4). The design itself was sound.

## 4. First principles: where the bottleneck is

Learning from experience needs five things:
1. a feedback signal;
2. a learner that extracts the reusable lesson;
3. a store (prompt, memory or weights);
4. a gate against learning the wrong thing;
5. reuse on new instances.

The table in section 2 covers 3, 4 and 5 many times over. CL-Bench isolates 2 as the weak link. With the entire
history in its context, where no information is lost, a frontier model still captures only 25.4%. Memory systems do
worse because they compress experience through the same weak extractor, which throws away the evidence needed to
revise a wrong lesson later.

Whatever form the lesson is stored in, the extractor is the base model's own inductive ability. That ability grows
with model scale and training, the resources the labs have. Self-teaching methods are bounded by what the model can
do with the privileged context. At 1.5B that is little.

The one route that sidesteps the model's own ability is to let it induce with tools, meaning code and statistics
over the stored data. That is where Claude Code beats plain in-context learning on CL-Bench (Sales Prediction,
Database Exploration). Program-synthesis world models and hypothesis search are also published (WorldCoder,
Hypothesis Search, Darwin Gödel Machine).

## 5. Conclusion and options

Both lines checked in depth, facts learned after deployment and skills learned from experience, lead to the same
answer:
- the problems are real;
- every mechanism is being worked on by several well-resourced groups in 2026;
- the binding constraint is base-model capability and compute, which this team does not have.

I found no unclaimed, breakthrough-sized opening within the team's means.

What remains realistically open to a small team:
1. **Build, not discover.** Use frontier models through their APIs, plus the published pieces (full history,
   tool-based induction, evidence-keeping memory, a statistical acceptance gate), in one domain where the owner has
   access to real work. Measure it with CL-Bench's gain metric. This yields a useful system and honest evidence, not
   a new method. A CL-Bench run costs about $8–$60 per system.
2. **Publish the accountable fact memory** (see "Using learned knowledge in reasoning 2026").
3. **Obtain compute or join a group** already working on this, if the aim remains a method-level breakthrough.

## Sources

- CL-Bench: [arXiv 2606.05661](https://arxiv.org/abs/2606.05661), [leaderboard](https://snorkel.ai/leaderboard/continual-learning-bench/)
- Dynamic Cheatsheet: [arXiv 2504.07952](https://arxiv.org/abs/2504.07952); Test-Time Learning with an Evolving Library: [arXiv 2605.14477](https://arxiv.org/pdf/2605.14477)
- ReasoningBank: [arXiv 2509.25140](https://arxiv.org/abs/2509.25140); Evo-Memory: [arXiv 2511.20857](https://arxiv.org/abs/2511.20857); Early Experience: [arXiv 2510.08558](https://arxiv.org/abs/2510.08558)
- Belief Memory: [arXiv 2605.05583](https://arxiv.org/pdf/2605.05583); TOKI: [arXiv 2606.06240](https://arxiv.org/pdf/2606.06240); TEPA: [arXiv 2608.07429](https://arxiv.org/pdf/2608.07429); Can Agent Memory Systems Track Evolving State?: [alphaXiv 2608.19652](https://www.alphaxiv.org/abs/2608.19652); belief-revision survey: [paper list](https://github.com/jiminHuang/belief-state-survey)
- SDFT: [arXiv 2601.19897](https://arxiv.org/abs/2601.19897); SDPO: [arXiv 2601.20802](https://arxiv.org/abs/2601.20802), [code](https://github.com/lasgroup/SDPO); SIEVE: [arXiv 2604.02339](https://arxiv.org/abs/2604.02339); SEAL: [arXiv 2506.10943](https://arxiv.org/abs/2506.10943)
- Predictive law for self-distillation: [arXiv 2605.30070](https://arxiv.org/pdf/2605.30070); Rethinking continual experience internalization: [arXiv 2606.04703](https://arxiv.org/pdf/2606.04703); Denser ≠ Better: [arXiv 2607.01763](https://arxiv.org/pdf/2607.01763); Procedural Memory Distillation: [arXiv 2607.01480](https://arxiv.org/pdf/2607.01480); Self-Practice: [arXiv 2609.29051](https://arxiv.org/pdf/2609.29051); SOD for small agents: [arXiv 2605.07725](https://arxiv.org/pdf/2605.07725)
- Darwin Gödel Machine: [arXiv 2505.22954](https://arxiv.org/pdf/2505.22954); Hierarchical Self-Improvement: [arXiv 2608.08466](https://arxiv.org/pdf/2608.08466); Ouroboros: [arXiv 2608.08311](https://arxiv.org/pdf/2608.08311); Live-SWE-agent: [arXiv 2511.13646](https://arxiv.org/pdf/2511.13646); fast tree search: [arXiv 2609.19526](https://arxiv.org/pdf/2609.19526)
- PACE: [arXiv 2606.08106](https://arxiv.org/pdf/2606.08106); Anytime-valid certificates: [arXiv 2607.00871](https://arxiv.org/pdf/2607.00871)
- Earlier reports: [AGI bottlenecks and research gaps 2026](AGI%20bottlenecks%20and%20research%20gaps%202026.md), [Using learned knowledge in reasoning 2026](Using%20learned%20knowledge%20in%20reasoning%202026.md)
