# What stands between frontier AI and AGI: expert claims and benchmark evidence (2024 to 29 Sep 2026)

(Work in progress; notes being appended incrementally.)

## Q1. What do key figures say are the main missing pieces?

## Q2. What does benchmark evidence show as of 2026?

## Q3. Which bottlenecks recur most, and where do experts disagree?

## Q4. Published AGI definitions / level frameworks and what they say is weakest

---
### RAW NOTES BATCH 1 (to be restructured)

**Access note:** metr.org, arcprize.org and dwarkesh.com are blocked by this environment's egress proxy. Primary numbers from those sites are taken from arXiv mirrors on Hugging Face (hf://papers/...) where possible, otherwise from search snippets / secondary reporting (flagged).

**"A Definition of AGI" (Hendrycks, Song, Szegedy, ... Marcus, Tallinn, Schmidt, Bengio; arXiv 2510.18212, Oct 2025)** — read in full via HF mirror of https://arxiv.org/abs/2510.18212
- Definition: AGI = "matching the cognitive versatility and proficiency of a well-educated adult"; grounded in Cattell-Horn-Carroll (CHC) theory; 10 equally-weighted domains (10% each): General Knowledge (K), Reading & Writing (RW), Math (M), On-the-Spot Reasoning (R), Working Memory (WM), Long-Term Memory Storage (MS), Long-Term Memory Retrieval (MR), Visual (V), Auditory (A), Speed (S).
- Table 1 (MEASUREMENT, by authors' battery): GPT-4 (2023) = 27% total: K 8, RW 6, M 4, R 0, WM 2, MS 0, MR 4, V 0, A 0, S 3. GPT-5 (2025, "Auto" mode) = 57% total: K 9, RW 10, M 10, R 7, WM 4, MS 0, MR 4, V 4, A 6, S 3 (each out of 10).
- Sub-scores: MS (associative 4%, meaningful 3%, verbatim 3%) = 0/0/0 for both GPT-4 and GPT-5. MR: Fluency 4/6, Hallucinations 0/4 for BOTH models ("they both frequently hallucinate"). R: GPT-5 deduction 2/2, induction 2/4, ToM 2/2, planning 1/1, adaptation 0/1. V: GPT-5 perception 2/4, generation 2/3, visual reasoning 0/2, spatial scanning 0/1. Speed: 3/10 for both.
- Authors' claim: "Long-term memory storage is perhaps the most significant bottleneck, scoring near 0% for current models. Without the ability to continually learn, AI systems suffer from 'amnesia'... forcing the AI to re-learn context in every interaction. Similarly, deficits in visual reasoning limit the ability of AI agents to interact with complex digital environments."
- "Capability contortions": long context windows (WM) used to compensate for missing MS ("fails to scale for tasks requiring days or weeks of accumulated context"); RAG used to paper over hallucination and the absence of "a dynamic, experiential memory".
- Engine analogy: "An artificial mind, much like an engine, is ultimately constrained by its weakest components."

**ARC-AGI-3 technical report (ARC Prize Foundation; arXiv 2603.24621; launched 25 Mar 2026)** — read via HF mirror of https://arxiv.org/abs/2603.24621 and https://arcprize.org/media/ARC_AGI_3_Technical_Report.pdf
- Interactive, turn-based novel environments; agent must explore, infer goals, build a model of dynamics, and plan, with no instructions. Scoring = action efficiency vs. the 2nd-best first-run human (RHAE). "humans can solve 100% of the environments, in contrast to frontier AI systems which, as of March 2026, score below 1%."
- Human calibration: 486 participants, 414 candidate environments, 2,893 attempts; each environment attempted by 10 people and included only if ≥2 solved it independently on first contact; median attempt 7.4 min.
- Launch scores (Table 2 not rendered in mirror; figures from secondary reports): Gemini 3.1 Pro 0.37%, GPT-5.4 0.26%, Claude Opus 4.6 0.25%, Grok 4.2 ~0% — [MLQ](https://mlq.ai/news/arc-agi-3-benchmark-reveals-major-gap-between-frontier-models-and-human-level-reasoning/), [Medium summary](https://medium.com/@AdithyaGiridharan/arc-agi-3-dropped-and-frontier-ai-scored-less-than-1-90cd70e65a61), [DataCamp](https://www.datacamp.com/blog/arc-agi-3).
- Harness evidence: "in a variant of environment TR87, Opus 4.6 scores 0.0% with no harness and 97.1% with the Duke harness, yet in environment BP35, Opus 4.6 scores 0.0% under both configurations" -> hand-crafted harnesses don't transfer to unseen environments.
- Preview competition (18 Jul–19 Aug 2025): winner StochasticGoose (Tufa Labs, CNN+RL) 12.58%; 2nd Blind Squirrel 6.71% — both essentially informed search/brute exploration.
- ARC Prize claims (opinion): LRM reasoning "is tied to LRM knowledge"; automation works where (a) base model has knowledge coverage and (b) domain is verifiable. "machines that can perform highly efficient adaptation to produce paradigm-shifting innovation are still well outside our reach." Believe ARC-AGI-1/2 were partly "attacked via higher-level shortcuts" (dense synthetic task training); evidence: Gemini 3 Deep Think used the correct ARC integer-to-colour mapping unprompted.
- History (measurement): ARC Prize 2024 top private ARC-AGI-1 score 53.5% (test-time training); ARC Prize 2025 (ARC-AGI-2) winner NVARC (NVIDIA) 24% with a 4B model + synthetic data + TTT; 85% grand prize unclaimed both years. ARC-AGI-1 task ~30 s for humans; ARC-AGI-2 ~300 s.

**Sutskever (Dwarkesh Podcast, 25 Nov 2025, "We're moving from the age of scaling to the age of research")** — [Apple Podcasts listing](https://podcasts.apple.com/uy/podcast/ilya-sutskever-were-moving-from-the-age-of-scaling/id1516093381?i=1000738363711); [Calcalist](https://www.calcalistech.com/ctechnews/article/h1fudk7z11x); [MindStudio summary](https://www.mindstudio.ai/blog/ilya-sutskever-age-of-research-scaling)
- Eras: "age of research" ~2012–2020, "age of scaling" 2020–2025, now back to research with big compute. Core technical gap = generalization: models "generalize dramatically worse than people"; bottleneck is ideas, not compute (per Calcalist headline). Pre-training data finite.

**Karpathy (Dwarkesh Podcast, released 17 Oct 2025, "AGI is still a decade away")** — [Simon Willison notes, 18 Oct 2025](https://simonwillison.net/2025/Oct/18/agi-is-still-a-decade-away/); [Fortune, 20 Oct 2025](https://fortune.com/2025/10/20/workers-fear-ai-job-cuts-open-ai-co-founder-says-ai-agents-will-take-a-decade-before-they-even-work-they-dont-have-enough-intelligence-unemployment-automation-2035/)
- Quote: "They just don't work. They don't have enough intelligence, they're not multimodal enough, they can't do computer use and all this stuff. They don't have continual learning. You can't just tell them something and they'll remember it. They're cognitively lacking and it's just not working." Said it will take "about a decade" to work through these; "decade of agents" vs. "year of agents".

**Dwarkesh Patel essay "Why I don't think AGI is right around the corner" (2 Jun 2025 per X post; some aggregators say 3 Jul 2025)** — [X post](https://x.com/dwarkesh_sp/status/1929598755508310090); [Zvi response, 9 Jun 2025](https://thezvi.wordpress.com/2025/06/09/dwarkesh-patel-on-continual-learning/); [Nathan Lambert rebuttal](https://www.interconnects.ai/p/contra-dwarkesh-on-continual-learning)
- "I think continual learning is a huge bottleneck to the usefulness of these models, and extended computer use may take years to sort out." Argues you can't automate a job by baking in a fixed skill set; humans improve on the job via feedback; LLMs don't.

---
### RAW NOTES BATCH 2

**Access note (cont.):** Also blocked: en.wikipedia.org, lesswrong.com, business-standard.com, thezvi.substack.com, nextbigfuture.com, officechai.com, r40.io, llm-stats.com, artificialanalysis.ai, the-decoder.com, digitalapplied.com, techmeme.com, thenewstack.io. For these, only search-engine snippets were available; figures below that rely on snippets are marked [snippet].

**Silver & Sutton, "Welcome to the Era of Experience" (preprint of a chapter for MIT Press book *Designing an Intelligence*; circulated April 2025)** — read in full: [DeepMind PDF](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
- "while imitating humans is enough to reproduce many human capabilities to a competent level, this approach in isolation has not and likely cannot achieve superhuman intelligence across many important topics and tasks. In key domains such as mathematics, coding, and science, the knowledge extracted from human data is rapidly approaching a limit... The pace of progress driven solely by supervised learning from human data is demonstrably slowing."
- Four needed shifts: agents "will inhabit streams of experience, rather than short snippets of interaction"; actions/observations "richly grounded in the environment"; rewards "grounded in their experience of the environment, rather than coming from human prejudgement"; planning/reasoning grounded in world, not imitating human thought.
- Human-prejudged rewards lead to "an impenetrable ceiling on the agent's performance"; current RL is "designed for short episodes of ungrounded, human interaction, and are not suitable for long streams of grounded, autonomous interaction."

**Sutton on Dwarkesh (late Sep 2025, "Father of RL thinks LLMs are a dead end")** — [Dwarkesh (blocked, title only)](https://www.dwarkesh.com/p/richard-sutton); [Apple Podcasts](https://podcasts.apple.com/us/podcast/richard-sutton-declares-llms-a-dead-end/id1802074035?i=1000732717643); [The Neuron summary](https://www.theneuron.ai/explainer-articles/the-great-ai-debate-are-llms-a-brilliant-leap-or-a-sophisticated-dead-end/) [snippet]
- LLMs imitate what a person would say rather than predict what will happen in the world; no goal / no ground truth; cannot learn on the job -> a new architecture enabling continual learning is needed "no matter how much we scale".

**Hassabis, India AI Impact Summit, New Delhi, 18 Feb 2026** — [Business Standard](https://www.business-standard.com/technology/tech-news/google-deepmind-ceo-demis-hassabis-ai-impact-summit-delhi-systems-agi-126021800278_1.html) [snippet]; [Storyboard18](https://www.storyboard18.com/amp/brand-makers/google-deepmind-ceo-says-agi-not-here-yet-calls-current-ai-jagged-intelligence-90028.htm) [snippet]
- AGI not here yet; current AI is "jagged intelligence" — IMO-gold-level yet elementary mistakes. Missing: continual learning ("we train them and then they're kind of frozen and then put out into the world"; need to "continually learn online from experience and the context they're in"), long-term planning, consistency across tasks. AGI possible in "five to eight years".
- A later X clip (id 2060949392526352472, ~mid-2026) paraphrases him: still jagged; fail on "reliability, consistency, memory, and understanding how the world works"; "only a few problems remain, like continual learning, long-term..." — [X/Haider](https://x.com/haider1/status/2060949392526352472) [snippet; secondary poster, treat as paraphrase].

**LeCun / AMI Labs (2026)** — [MIT Technology Review, 22 Jan 2026](https://www.technologyreview.com/2026/01/22/1131661/yann-lecuns-new-venture-ami-labs/) [snippet]
- Left Meta (Nov 2025) to found Advanced Machine Intelligence Labs; March 2026 seed round reported at $1.03B (~$3.5B pre-money) [snippet via secondary sites]. Position: human-level AI "won't be built on LLMs" and needs conceptual breakthroughs: world models that understand physical reality, JEPA (joint-embedding predictive architecture), persistent memory, reasoning and planning. Clashed publicly with Amodei and Hassabis at Davos (Jan 2026).

**Amodei on Dwarkesh (Feb 2026, "We are near the end of the exponential")** — [YouTube](https://www.youtube.com/watch?v=n1E9IZfvGMA); [Zvi analysis](https://thezvi.substack.com/p/on-dwarkesh-patels-2026-podcast-with) [snippet]
- Says we are not at AGI ("if we did have a country of geniuses in a datacenter... everyone would know"). On continual learning: Anthropic is working on it and there is "a good chance" it is solved in "the next year or two", but most of the economic value (trillions/yr) can arrive without it. Stands by "country of geniuses" arriving within a few years. [snippet-derived paraphrase; exact wording not verified]

**Shane Legg (DeepMind co-founder / Chief AGI Scientist), Dec 2025 (Google DeepMind podcast per NextBigFuture)** — [NextBigFuture, Dec 2025](https://www.nextbigfuture.com/2025/12/google-ai-lead-shane-legg-defines-levels-of-agi-and-superintelligence-and-how-to-test-for-it.html) [snippet]
- "Minimal AGI" expected ~2027-2028 (50% by 2028 is his long-standing forecast). Current weaknesses: continual learning, visual/spatial reasoning (perspective in scenes, diagram reasoning); lack of episodic memory (hippocampus-like rapid learning) compensated by long context windows. Sees "relatively clear paths" and "no big blockers" for delusions/factuality/memory.
- Legg on X (Jan 2025): AGI = can do the cognitive problems regular people can do; "By this criteria we're not there yet, but I think we might get there in the coming years." — [X](https://x.com/ShaneLegg/status/1877674960770007042)

**METR time horizons (software tasks; human-expert-time at which agent succeeds 50%)**
- Claude Opus 4.5: ~4h49m 50%-horizon (late 2025) — [LessWrong post title](https://www.lesswrong.com/posts/q5ejXr4CRuPxkgzJD/claude-opus-4-5-achieves-50-time-horizon-of-around-4-hrs-49) [snippet]
- Time Horizon 1.1 (29 Jan 2026): 228 tasks (up from 170); post-2023 doubling ~130.8 days (4.3 months); from 2024 onward ~89 days (~3 months) — [METR TH1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) [snippet]. Original (Mar 2025) estimate: doubling ~7 months (2019–2025). Another snippet cites ~105-day doubling Jan 2024–Feb 2026.
- Claude Opus 4.6 (added 20 Feb 2026): 50%-horizon ~14.5 h, 95% CI 6 h to 98 h; METR: "this measurement is extremely noisy because our current task suite is nearly saturated." — [METR on X](https://x.com/METR_Evals/status/2024923422867030027)
- Claude Mythos Preview (added 8 May 2026): ≥16 h (95% CI 8.5–55 h); METR: "Measurements above 16 hrs are unreliable with our current task suite." — [OfficeChai](https://officechai.com/ai/claude-mythos-shows-50-time-horizon-of-16-hours-on-metr-benchmark/) [snippet]; [Digg](https://digg.com/tech/9x3pz8ed) [snippet]
- METR limitations note (22 Jan 2026) — [METR](https://metr.org/notes/2026-01-22-time-horizon-limitations/) (blocked; title only).

**ARC-AGI-3 progress in 2026 (major change since launch)**
- Launch (25 Mar 2026): best frontier <1% (Gemini 3.1 Pro 0.37%).
- ARC Prize analysis of GPT-5.5 & Opus 4.7 on ARC-AGI-3 — [ARC Prize blog](https://arcprize.org/blog/arc-agi-3-gpt-5-5-opus-4-7-analysis) (blocked; title only).
- GPT-5.6 Sol (Max): 7.8% (previous record before Opus 5) — [the-decoder](https://the-decoder.com/anthropics-opus-5-blows-past-fable-5-and-gpt-5-6-sol-on-the-benchmark-designed-to-measure-real-intelligence/) [snippet]
- Claude Opus 5 (High): 30.16% verified by ARC Prize, 24 Jul 2026 — [ARC Prize results page](https://arcprize.org/results/anthropic-claude-opus-5) [snippet]; [FourWeekMBA](https://fourweekmba.com/ai-claude-opus-5-arc-agi-benchmark-harness-scaffolding/) says 97.5% with a harness vs 30.16% official [snippet].
- GPT-6 Astra (ARC Prize, ~3 Sep 2026): 62.7% on Semi-Private with Standard harness at max reasoning for $26,098; 99.9% for $18,817 with OpenAI's "Provider Adapter" harness (can preserve opaque reasoning state and compact long conversations); ARC Prize: "surpasses human performance on 96% of ARC-AGI-3 levels" and "builds the most precise symbolic model of novel environments we've seen" — [ARC Prize on X](https://x.com/arcprize/status/2095597602545025138); [ARC Prize blog "astra"](https://arcprize.org/blog/astra) (blocked); [Techmeme 3 Sep 2026](https://www.techmeme.com/260903/p40) [snippet]; [The New Stack](https://thenewstack.io/astra-arc-agi-benchmark/) [snippet]. NOTE: the "96% of levels" figure is vs the median tested human in action efficiency per snippet; exact definition unverified.
- Claude Opus 5.5 (ARC Prize Verified): ARC-AGI-2 93.3% at $0.41/task; ARC-AGI-1 98.5% at $0.16/task; ARC-AGI-3 testing not completed because API requests were misclassified as reverse-engineering attempts — [ARC Prize on X](https://x.com/arcprize/status/2102512140405866568) [snippet].

**ARC-AGI-2 progression** [snippets, verify]: GPT-5.2 Pro 54.2% vs Gemini 3 Deep Think preview 45.1% (Dec 2025) — [OfficeChai](https://officechai.com/ai/gpt-5-2-pro-creates-new-record-of-54-2-on-arc-agi-2-beats-gemini-3-deep-think-preview-at-45-1/); Gemini 3 Deep Think 84.6% at $13.62/task (early 2026) — [Implicator](https://www.implicator.ai/google-gemini-3-deep-think-hits-84-6-on-arc-agi-2-beating-gpt-5-and-claude-2/); Opus 5.5 93.3% at $0.41/task (Sep 2026, ARC Prize verified). Aggregator claims (GPT-6 Astra 95.0%, GPT-5.6 Sol 92.5%, Opus 5 90.4%) come from low-reliability aggregators (benchlm.ai) — UNVERIFIED. ARC-AGI-1 87.5% cost ~$4,560/task (o3, Dec 2024) vs ~$0.30 in 2026 per [R40](https://r40.io/blog/arc-agi-verified-cost-per-task-price-step/) [snippet]; same source says ARC-AGI-4 planned for 2027.
