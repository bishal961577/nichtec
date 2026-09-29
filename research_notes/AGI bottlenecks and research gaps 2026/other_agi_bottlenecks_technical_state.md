# Technical state of AGI bottlenecks other than memory/continual learning (as of 29 Sep 2026)

_Status: IN PROGRESS — notes written incrementally. Sections below are filled as research proceeds._

## 1. Reasoning reliability, hallucination and calibration
(pending)

## 2. Long-horizon autonomous agents
(pending)

## 3. Sample efficiency and learning from experience
(pending)

## 4. Out-of-distribution generalization and abstraction (ARC)

### Takeaway
Static ARC puzzles (ARC-AGI-1/2) are now effectively saturated by frontier reasoning models (ARC-AGI-2 went from ~0-4% in early 2025 to 90% verified for Claude Fable 5.1 in Sept 2026), but ARC Prize itself argues this is partly "knowledge overfitting" via dense synthetic coverage of the task space, not human-like fluid intelligence. The interactive ARC-AGI-3 (launched 25 Mar 2026; humans 100%, frontier AI <1% at launch) is the live test of OOD adaptation: scores jumped to 30.2% (Claude Opus 5, Jul 2026) and 62.7% (GPT-6 Astra, standard harness, Sep 2026) -- and to 99.9% when the model is allowed to preserve its own reasoning state across turns, which makes ARC-AGI-3 performance largely a function of within-episode memory/adaptation.

### Cited Findings
**ARC-AGI-1/2 (static)**
- ARC Prize 2025 Kaggle competition (26 Mar-3 Nov 2025): 1,455 teams, 15,154 entries; top private-set ARC-AGI-2 score 24.03% (NVARC, NVIDIA; fine-tuned ~4B model + heavy synthetic data + test-time training) at ~$0.20/task; 2nd the ARChitects 16.53% (2D-aware masked-diffusion LM with recursive self-refinement); 3rd MindsAI 12.64% (test-time fine-tuning pipeline). Paper submissions rose from 47 (2024) to 90 (2025). — [ARC Prize 2025 Technical Report, arXiv 2601.10904 (Jan 2026)](https://arxiv.org/abs/2601.10904)
- ARC Prize names the "refinement loop" (per-task iterative program optimisation guided by a feedback signal) the defining theme of 2025: test-time training (weights as program), zero-pretraining tiny nets (TRM: 7M params, 45% ARC-AGI-1 / 8% ARC-AGI-2; CompressARC: 76K params, 20% ARC-AGI-1, no pretraining, ~20 min/task on an RTX 4070, MDL objective), evolutionary program synthesis (Berman: natural-language programs; Pang: Python + learned abstraction library), and CoT refinement. — [ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904)
- Application-layer "refinement harnesses": Poetiq's Gemini 3 Pro harness raised ARC-AGI-2 from 31% at $0.81/task to 54% at $31/task (verified by ARC Prize, Q4 2025); similar gains on Claude Opus 4.5 at ~$60/task (Poetiq-reported). Gemini 3 Pro used 96 reasoning tokens on one ARC-AGI-1 task where Gemini 3 Deep Think used 138,000. — [ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904)
- ARC Prize's own diagnosis (Jan & Mar 2026): LRMs automate a domain only when (1) the base model has sufficient knowledge coverage and (2) the domain provides verifiable feedback; "AI reasoning capability is tied to LRM knowledge ... human reasoning capability is not bound by domain knowledge." They assert ARC-AGI-1/2 have been "overfit" via dense (possibly synthetic) sampling of the task space -- evidence: Gemini 3 Deep Think used the correct ARC integer-to-colour mapping in its reasoning without being told it was an ARC task. For the ARC-AGI-1/2 format "the Grand Prize accuracy gap is now primarily bottlenecked by engineering, while the efficiency gap remains bottlenecked by fundamental science." — [ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904); [ARC-AGI-3 paper, arXiv 2603.24621](https://arxiv.org/abs/2603.24621)
- Sept 2026 verified: Claude Fable 5.1 (Anthropic, released ~1 Sep 2026) ARC-AGI-2 90.0% at $3.12/task, ARC-AGI-1 97.5% at $1.40/task; Fable 5 was 89.2%. — [ARC Prize on X](https://x.com/arcprize/status/2094894027451539744); [ARC Prize results page](https://arcprize.org/results/anthropic-claude-fable-5-1). (A secondary aggregator lists $4.49/task and Gemini 3.7 Flash at 84.6% for $0.25/task and GPT-5.5 at 85% (Apr 2026) — [bracai.eu](https://www.bracai.eu/post/arc-agi-2-benchmark); these secondary numbers were not verified against arcprize.org, which is blocked from this environment.)
- Cost collapse: an aggregator reports the same 87.5% ARC-AGI-1 score that cost ~$4,560/task (o3-preview high, Dec 2024) now costs ~$0.30/task in 2026, and that ARC-AGI-4 is planned for 2027 (unverified secondary source). — [R40](https://r40.io/blog/arc-agi-verified-cost-per-task-price-step/)

**ARC-AGI-3 (interactive, launched 25 Mar 2026)**
- Design: hundreds of hand-made turn-based game environments with no instructions, no stated rules, no stated goals; the agent must explore, infer goals, build a model of dynamics and plan. Score = RHAE (Relative Human Action Efficiency): actions needed to beat each level on first contact vs. a human baseline; "beating" requires matching human action efficiency averaged over private environments. — [ARC-AGI-3 paper, arXiv 2603.24621](https://arxiv.org/abs/2603.24621)
- At launch (developer preview, Mar 2026): humans solved 100% of environments; frontier models scored 0-0.37% (Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Claude Opus 4.6 0.25%). — [ARC-AGI-3 paper](https://arxiv.org/abs/2603.24621); [ARC Prize launch blog via search snippet](https://arcprize.org/blog/arc-agi-3-launch)
- Progression (per ARC Prize results pages/search snippets): GPT-5.6 Sol 7.8% (prior record) -> Claude Opus 5 30.2% (~24 Jul 2026; solved 5 environments no AI had beaten, 4 at or above human efficiency; ARC Prize noted it wrote "reflection equations") -> GPT-6 Astra 62.7% on the Standard harness (~Sep 2026, ~$26K for the semi-private set) and 99.9% (~$19K) with OpenAI's "Provider Adapter" harness. Leaderboard as of 24 Sep 2026 per an aggregator: GPT-6 Astra 62.7%, Claude Opus 5 30.2%, Gemini 3.8 Flash 10.4%. Leaderboard computed on ~50% of test data. — [ARC Prize: GPT-6 Astra blog](https://arcprize.org/blog/astra); [Renascence](https://www.renascence.io/news/7984/claude-opus-5-scores-30-2-on-arc-agi-3-nearly-4-prior-record); [BenchLM](https://benchlm.ai/benchmarks/arcagi3). (arcprize.org itself could not be fetched; figures come from search-result extracts of arcprize.org and news coverage -- treat as likely but not directly verified.)
- The 62.7% vs 99.9% gap for the same model is attributed by ARC Prize to memory handling: the Standard harness lets the model carry only visible notes it chooses to keep; the Provider Adapter preserves opaque reasoning state between requests and compacts long conversations, letting the model reuse prior work. — [ARC Prize: GPT-6 Astra on ARC-AGI-3](https://arcprize.org/blog/astra); [andrew.ooo explainer](https://andrew.ooo/answers/arc-agi-3-standard-harness-vs-provider-adapter-2026/)
- ARC Prize 2026 ARC-AGI-3 Kaggle track: $700K for the first agent to reach 100%; milestone checkpoints 30 Jun and 30 Sep 2026; submissions close 2 Nov 2026. Milestone #1 top entries were REPL-based agents, the winner running Qwen 3.6 27B locally and writing/running Python in a live REPL (hand-crafted tools reportedly hurt performance). — [ARC Prize 2026 milestone #1 (search snippet)](https://arcprize.org/blog/arc-prize-2026-milestone-1); [ARC-AGI-3 launch coverage](https://mlq.ai/news/arc-agi-3-benchmark-reveals-major-gap-between-frontier-models-and-human-level-reasoning/)

### Inferences
- The mechanism of the remaining OOD gap, per ARC Prize, is that LRM reasoning is "knowledge-bound": performance comes from dense coverage of a task distribution (including synthetic data + RL on verifiable ARC-like tasks), so when a benchmark is iid with public data it saturates, and a genuinely new format (ARC-AGI-3) resets scores to ~0. The 2026 ARC-AGI-3 jump happened within ~6 months, which is consistent either with real progress in exploration/in-context learning or with labs training on interactive-game distributions -- the data cannot yet separate these.
- ARC-AGI-3 is the benchmark most directly tied to continual/in-episode learning: the scoring is literally sample (action) efficiency of learning a new environment, and the single largest measured lever (62.7% -> 99.9%) was preserving the model's state across turns. This bottleneck is therefore strongly (not fully) downstream of the learn-after-deployment problem, but at the within-episode (context/state) timescale rather than weight updates.
- Top open-source methods (TTT, TRM, CompressARC) are themselves forms of test-time weight learning, i.e., they are small-scale continual learning; this is evidence that "learning at test time" is the working ingredient for OOD abstraction.

### Gaps
- Could not access arcprize.org directly to confirm exact current ARC-AGI-2/3 leaderboard numbers, costs and dates; figures above come from search-engine extracts and secondary coverage.
- Results of the 30 Sep 2026 ARC-AGI-3 milestone were not yet available (today is 29 Sep 2026).
- Whether ARC-AGI-3 gains transfer to other interactive OOD settings is unmeasured.

## 5. World models and physical/embodied grounding
(pending)

## 6. Data, compute and energy limits of scaling
(pending)

## 7. Which bottlenecks are downstream of missing continual learning vs independent
(pending)
