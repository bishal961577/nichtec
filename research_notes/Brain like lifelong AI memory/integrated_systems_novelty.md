# Integrated "learn-into-weights" personal assistant: novelty check and prior-art map (as of 2026-09-27)

Scope: is there prior work on an AI assistant, ideally on-device or personal, whose own parameters are permanently updated from (1) things the user teaches it in conversation and (2) information it looks up on the web once? The goal is an assistant that later answers from its parameters with no context window, no re-retrieval and no re-search, keeps learning for life and does not forget.

Source-reliability labels used below:
- **[Primary]**: I read the paper, doc or README myself, either on the Hugging Face paper mirror, raw GitHub, or the official docs page.
- **[Primary via snippet]**: the official page was blocked by the egress proxy, so the content comes from the search engine's summary of that page.
- **[Secondary]**: press or third-party guide.
- **[Rumour/unverified]**: an unsourced or speculative claim.

These hosts were blocked for direct fetch: openai.com, help.openai.com, support.google.com, research.google, meta.com, thinkingmachines.ai, hyper.ai and arxiv.org. Papers were read through the Hugging Face paper mirror (hf://papers/<id>).

---

## Q1. How do deployed assistants implement "memory" today? Is any of them parametric (weights change per user)?

### Takeaway
Every deployed assistant-memory product I found keeps memory outside the weights. That covers ChatGPT, Claude, Gemini, Apple Intelligence, Meta AI, Limitless/Rewind, Mem0, Letta/MemGPT, Zep and A-MEM. Memory is extracted text, a profile, files, knowledge-graph entries or a semantic index, and it is retrieved or injected into the context window at answer time. No product documents per-user weight updates. The only production system I found that continuously updates weights from usage is Cursor's "real-time RL" for Composer. That is population-level and skill/behaviour-oriented, not a per-user store of facts.

### Cited Findings
**ChatGPT (OpenAI)**
- [Primary via snippet] ChatGPT memory has two parts. "Saved memories" are explicit facts the user asked it to remember. "Reference chat history" draws on past conversations. OpenAI says saved memories "work similarly to custom instructions, except our models update them automatically". — [OpenAI Memory FAQ](https://help.openai.com/en/articles/8983136-what-is-memory); [Memory and new controls for ChatGPT](https://openai.com/index/memory-and-new-controls-for-chatgpt/)
- [Primary via snippet] "ChatGPT can now reference all past conversations" launched on April 10, 2025. — [OpenAI Developer Community announcement thread](https://community.openai.com/t/chatgpt-can-now-reference-all-past-conversations-april-10-2025/1229453)
- [Secondary] Reference chat history is not exposed as an editable list; the user can only turn it on or off. — [chatgptmemory.com](https://chatgptmemory.com/saved-memories-vs-reference-chat-history/)
- [Secondary] Relevant memory is selected and injected into the model's context "like a hidden system note". Memories are stored server-side, tied to the account. — [gizmotimes guide](https://www.gizmotimes.com/guides/chatgpt-memory-explained/50958); [Embrace The Red reverse-engineering post](https://embracethered.com/blog/posts/2025/chatgpt-how-does-chat-history-memory-preferences-work/)
- [Secondary] Since April 2025, ChatGPT also uses memory to personalize web searches. — [TechCrunch, 2025-04-18](https://techcrunch.com/2025/04/18/chatgpt-will-now-use-its-memory-to-personalize-web-searches)

**Claude (Anthropic)**
- [Primary] Claude app memory, blog post dated 2026-08-25: memory is stored as topic-based files the user can read, edit or delete. "Memory updates as you chat": topics are added during the conversation instead of being summarized afterwards. The same memory is shared between Claude chat and Claude Cowork. Sensitive subjects such as health and beliefs are excluded by default. The post describes no weight changes. — [Claude blog: "Claude's memory works everywhere, and you decide what's in it"](https://claude.com/blog/claudes-memory-works-everywhere-and-you-decide-whats-in-it)
- [Secondary] Persistent memory rolled out to all Claude users, free and paid, in March 2026. — [LumiChats guide](https://lumichats.com/blog/claude-memory-2026-complete-guide-how-to-use); [Tom's Guide](https://www.tomsguide.com/ai/claude-just-unlocked-memory-that-syncs-with-chatgpt-heres-how-it-works?web=1)
- [Primary] The API memory tool (`memory_20250818`) "lets Claude store and retrieve information across conversations in a directory of memory files". It supports "just-in-time context retrieval". It "operates client-side", so the application executes file operations under `/memories`. It works alongside compaction and context editing. — [Claude Platform docs: Memory tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool)

**Gemini (Google)**
- [Primary via snippet] Announced 2025-08-13: when "personal context" is on, Gemini "learns from your past conversations over time" by remembering key details and preferences. The setting is on by default. Temporary Chats "aren't used to personalize or train models" and are kept for up to 72 hours. — [blog.google: Gemini app personalizes responses based on past chats](https://blog.google/products-and-platforms/products/gemini/temporary-chats-privacy-controls/); [9to5Google, 2025-08-13](https://9to5google.com/2025/08/13/gemini-personal-context/)
- [Primary via snippet] "Personal Intelligence" is Google's name for the whole personalization layer: chat memory, instructions and connected Google apps. — [Gemini Apps Help](https://support.google.com/gemini/answer/16598469?hl=en&co=GENIE.Platform%3DAndroid); [datastudios summary (secondary)](https://www.datastudios.org/post/does-gemini-remember-past-conversations-context-retention-and-session-behavior)
- Note: Google's wording "learns from your past conversations" is not evidence of weight updates. None of the documents I found describe a per-user model update. This is my reading, and it is flagged as such.

**Apple Intelligence**
- [Secondary] Apple Intelligence uses an on-device "semantic index" of the user's files, photos, emails and similar content. Relevant personal context is retrieved and fed to the generative model. This is retrieval, not training. — [TechTarget](https://www.techtarget.com/whatis/definition/Apple-Intelligence)
- [Primary via snippet] WWDC26: app "entity schemas contribute your content to the Spotlight semantic index for personal context understanding". — [WWDC26 session 240, Apple Developer](https://developer.apple.com/videos/play/wwdc2026/240/)
- [Primary via snippet] Apple's position: "We do not use our users' private personal data or user interactions when training our foundation models." — [Apple ML Research, third-generation foundation models](https://machinelearning.apple.com/research/introducing-third-generation-of-apple-foundation-models); [Apple Support: Applebot model training](https://support.apple.com/en-ie/120320)
- [Primary via snippet] Apple's 2025 foundation-model tech report covers a roughly 3B on-device model. The Foundation Models framework exposes "LoRA adapter fine-tuning" for developers. LoRA adapters are also used to recover quality lost to 2-bit quantization. — [Apple Foundation Models Tech Report 2025](https://machinelearning.apple.com/research/apple-foundation-models-tech-report-2025) (arXiv 2507.13575)
- [Primary] Apple research, not a product: PLUM (Apple + Cambridge, 2024) injects prior user conversations into a LoRA adapter. See Q3. — [HF 2411.13405](https://huggingface.co/papers/2411.13405)

**Meta AI**
- [Primary via snippet] Meta AI memory, January 2025: Meta AI remembers details the user shares in 1:1 chats on WhatsApp and Messenger. Users can say "remember X", or Meta AI picks details up from context (e.g., "you're vegan" informs later recipe suggestions). It does not remember group chats, memories can be deleted, and it launched in the US and Canada. Meta AI can also use Facebook and Instagram profile and activity data. — [Meta Newsroom, "Building Toward a Smarter, More Personalized Assistant"](https://about.fb.com/news/2025/01/building-toward-a-smarter-more-personalized-assistant/); [Meta Help Center](https://www.meta.com/help/artificial-intelligence/1887269842194694/); [Silicon Republic (secondary)](https://www.siliconrepublic.com/machines/meta-ai-memory)

**Rewind / Limitless**
- [Secondary] Limitless (formerly Rewind) was acquired by Meta in December 2025. Pendant sales ended, and the Rewind app disabled screen and audio capture from 2025-12-19. The product captured and transcribed conversations into "searchable transcripts and summaries", which is a retrieval store. — [TechCrunch, 2025-12-05](https://www.techcrunch.com/2025/12/05/meta-acquires-ai-device-startup-limitless/); [WinBuzzer](https://winbuzzer.com/2025/12/05/meta-acquires-ai-wearables-startup-limitless-kills-pendant-sales-and-sunsets-rewind-app-xcxwbn/); [Rewind "what happened" page](https://rewind.ai/what-happened-to-rewind/)

**Mem0**
- [Primary] The Mem0 paper (April 2025) "dynamically extract[s], consolidat[es], and retriev[es] salient information", with an optional graph memory variant. It reports 26% relative LLM-as-judge improvement over OpenAI memory on LoCoMo, 91% lower p95 latency, and more than 90% token savings versus full context. — [HF 2504.19413](https://huggingface.co/papers/2504.19413)
- [Primary] README "New Memory Algorithm (April 2026)": "Single-pass ADD-only extraction… Memories accumulate; nothing is overwritten". It uses multi-signal retrieval (semantic, BM25 and entity) and time-aware retrieval. Scores: LoCoMo 92.5 and LongMemEval 94.4 at about 7K retrieved tokens, "Single-pass retrieval… top_200 retrieval budget". — [mem0 GitHub README](https://github.com/mem0ai/mem0)

**Letta / MemGPT and sleep-time compute**
- [Primary via snippet] Letta memory blocks: "A memory block represents a labeled section of the context window with an associated character limit". Agents rewrite blocks with tools such as `memory_replace` and `memory_rethink`. — [Letta blog: Memory Blocks](https://www.letta.com/blog/memory-blocks/)
- [Primary via snippet] Sleep-time agents "share the memory of your primary agents, but run in the background and can modify the memory asynchronously". Memory is reorganized during idle periods. — [Letta docs: Sleep-time agents](https://docs.letta.com/guides/agents/architectures/sleeptime/)
- [Primary] Sleep-time compute paper (Letta + UC Berkeley, April 2025): the model "think[s] offline about contexts before queries are presented". Test-time compute for the same accuracy falls by about 5× on Stateful GSM-Symbolic and Stateful AIME. Accuracy rises by up to 13% and 18% respectively. Amortizing across related queries cuts average cost per query by 2.5×. What gets pre-computed is stored as context, not weights. — [HF 2504.13171](https://huggingface.co/papers/2504.13171)
- [Primary] "Letta (f.k.a. MemGPT)": active development has moved to `letta-ai/letta-code`, and the V1 server is archived. — [letta GitHub README](https://github.com/letta-ai/letta)

**Zep / Graphiti**
- [Primary] Zep is a memory layer built on Graphiti, a temporally-aware knowledge graph. It scores 94.8% vs MemGPT's 93.4% on DMR. On LongMemEval it reports up to 18.5% accuracy improvement and 90% lower latency. — [HF 2501.13956](https://huggingface.co/papers/2501.13956); [Graphiti README](https://github.com/getzep/graphiti)

**A-MEM**
- [Primary] A-MEM builds Zettelkasten-style notes, with "Intelligent indexing and linking of memories via ChromaDB" and "memory evolution". It is an external store, not weights. — [A-mem README](https://github.com/agiresearch/A-mem); [HF 2502.12110](https://huggingface.co/papers/2502.12110)

**Production parametric continual updates, population-level only**
- [Primary via snippet] Cursor "real-time RL" for Composer serves checkpoints and turns billions of tokens of user interactions into reward. It updates the weights, gates the result on evals such as CursorBench, and redeploys, about every five hours. In an A/B test of Composer 1.5, edits persisting in the codebase rose 2.28%, dissatisfied follow-ups fell 3.13%, and latency fell 10.3%. — [Cursor blog: Improving Composer through real-time RL](https://cursor.com/blog/real-time-rl-for-composer); [Cursor on X](https://x.com/cursor_ai/status/2037205514975629493)
- [Secondary] Bessemer describes "Level 2" continual learning as weights updated post-deployment from production data. Its examples are Cursor and Prime Intellect's platform. — [Bessemer, AI Infrastructure Roadmap 2026](https://www.bvp.com/atlas/ai-infrastructure-roadmap-five-frontiers-for-2026)
- [Primary] A counterpoint from 2026: "Learning on the Job" (The Memory Company) argues that deployment feedback is enough for continual learning with frozen weights plus an external store of natural-language rules. On tau-bench banking it reports 1.6× success from outcome verdicts and 2.6× from corrections, and it replicated this on Claude Sonnet 5. — [HF 2607.22157](https://huggingface.co/papers/2607.22157)

### Inferences
- The industry's deployed notion of "memory" is context engineering: extraction, storage, retrieval and injection. What the user teaches is re-read into the context window on every relevant query. That is the opposite of the target design's "no context, no re-retrieval".
- The vendors frame privacy as not training on personal data (Apple explicitly; Gemini Temporary Chats "aren't used to... train models"). So per-user weight learning would be a product departure for them, not only a technical one. On-device per-user weights may be the only privacy story compatible with that framing.
- The only parametric learning happening in production (Cursor) is global, fleet-wide and RL-based. It targets behaviour and skills, not per-user episodic or semantic facts.

### Gaps
- I could not read the OpenAI help pages, Google support pages, the Meta help page or Apple's ML pages directly (egress blocked). Their details come from search snippets.
- I found no official statement from any vendor on how memory is retrieved (embedding vs full injection), except Anthropic's docs.
- The Apple Foundation Models framework mentions LoRA adapter training for developers. I could not confirm whether any on-device per-user training runs in shipping iOS. I found no evidence that it does.

---

## Q2. Research systems that write new or retrieved knowledge into weights, and the 2025–2026 continual-learning claims from labs

### Takeaway
There is substantial prior art for each mechanism in the target design:
- **Self-generated augmentation, then weight update:** SEAL, Self-Tuning, EntiGraph, prompt distillation, SDFT.
- **Web retrieval, then QA, then fine-tuning:** ALAS.
- **Sparse, low-forgetting parametric memory written after deployment:** sparse memory finetuning, WISE, MemoryLLM/M+.
- **Sleep / replay / dreaming consolidation into weights:** Behrouz & Mirrokni 2026, and Nested Learning/Hope for online consolidation.
- **Online fast-weight LoRA memory for agents:** TMEM, Doc-to-LoRA.

All of these are research prototypes, and none is an on-device, lifelong personal assistant. Lab continual-learning claims from 2025–2026 are mostly blog-level research directions, plus Cursor's production RL. I found no verified public release of a model that learns per-user facts into its weights after deployment.

### Cited Findings
**Self-generated augmentation leading to weight updates**
- [Primary] **SEAL** (Zweiger, Pari et al., MIT, June 2025): given new input, the model writes a "self-edit" (restructured data, optionally with hyperparameters). SFT on the self-edit gives "persistent weight updates". RL (ReST-EM) trains the self-edit policy, with the downstream accuracy of the updated model as reward.
  - No-context SQuAD knowledge incorporation (Qwen2.5-7B, single-passage LoRA):
    - base model: 32.7%
    - training on the passage only: 33.5%
    - base model's own synthetic data: 39.7%
    - GPT-4.1 synthetic data: 46.3%
    - SEAL: 47.0%
  - Continued-pretraining setting with n=200 passages: 58.2%.
  - Limitations the authors state:
    - Catastrophic forgetting: "performance on earlier tasks gradually declines as the number of edits increases".
    - Each self-edit evaluation takes about 30–45 s.
    - RL requires a paired QA set for each context.
  - The authors envision agents that "synthesize a self-edit which triggers a weight update" after an interaction.
  - Sources: [HF 2506.10943](https://huggingface.co/papers/2506.10943); [SEAL repo](https://github.com/Continual-Intelligence/SEAL)
- [Primary] "Search over Self-Edit Strategies" (Jan 2026): a SEAL variant (Qwen3-8B) in which the model also generates its own self-edit templates. With an archive of past templates it beats the "Implications" baseline but not the human "Rewrite" baseline. — [HF 2601.14532](https://huggingface.co/papers/2601.14532)
- [Primary] **Self-Tuning** (Zhang et al., CUHK/Tencent, June 2024; v5 May 2025): self-teaching augments raw new documents with knowledge-intensive tasks covering memorization, comprehension and self-reflection. It uses the Wiki-Newpages-2023-QA datasets and aims to acquire new knowledge while preserving prior knowledge. — [HF 2406.06326](https://huggingface.co/papers/2406.06326)
- [Primary] **Synthetic continued pretraining / EntiGraph** (2024): the model learns from small domain corpora through synthetic data generated over entity graphs. — [HF 2409.07431](https://huggingface.co/papers/2409.07431)
- [Primary] **Prompt distillation / "Efficient Knowledge Injection in LLMs via Self-Distillation"** (Aalto/System 2 AI, v2 Aug 2025): internalizes facts from free-form documents without a larger teacher. It "outperforms standard supervised fine-tuning and can even surpass RAG", and combining it with RAG beats RAG alone. — [HF 2412.14964](https://huggingface.co/papers/2412.14964)
- [Primary] **SDFT, "Self-Distillation Enables Continual Learning"** (MIT/ETH, Jan 2026): a demonstration-conditioned copy of the model acts as its own teacher, giving on-policy learning. On skill and knowledge-acquisition tasks it "substantially reduc[es] catastrophic forgetting" compared with SFT, and it accumulates skills sequentially without regression. — [HF 2601.19897](https://huggingface.co/papers/2601.19897)
- [Primary] Knowledge-Instruct (2025): injects knowledge with synthetic instructions and reduces forgetting. — [HF 2504.05571](https://huggingface.co/papers/2504.05571)

**Web retrieval leading to weights (closest to "search once, then learn")**
- [Primary] **ALAS: Autonomous Learning Agent for Self-Updating Language Models** (Aug 2025).
  - The loop: it "autonomously generates a learning curriculum", "retrieves up-to-date information from the web (with citations), distills this into question-answer training data, and fine-tunes the model through SFT and DPO". It then evaluates and revises the curriculum iteratively.
  - Post-cutoff QA accuracy rose "from 15% to 90% on average" on new Python releases, CVEs and academic trends.
  - It is built on "OpenAI's Deep Research and Fine-Tuning" APIs, so it is cloud-based, and learning is driven by a curriculum, not by user queries.
  - Stated limitations: cost and dependence on source quality.
  - Source: [HF 2508.15805](https://huggingface.co/papers/2508.15805)

**Sparse or dedicated parametric memory written after deployment (anti-forgetting)**
- [Primary] **Sparse memory finetuning** (FAIR at Meta + UC Berkeley, Oct 2025): uses memory-layer models and updates only the top-t memory slots, ranked by TF-IDF of access relative to pretraining. After learning a stream of TriviaQA facts, NaturalQuestions F1 drops 89% with full finetuning, 71% with LoRA, and 11% with sparse memory finetuning, "with the same level of new knowledge acquisition". — [HF 2510.15103](https://huggingface.co/papers/2510.15103)
- [Primary] **MemoryLLM** (2024): a transformer plus a roughly 1B-parameter latent memory pool that "can self-update with text knowledge". It shows no degradation "even after nearly a million memory updates". **M+** (2025) adds a co-trained retriever and long-term memory, extending retention from under 20k to over 160k tokens. — [HF 2402.04624](https://huggingface.co/papers/2402.04624); [HF 2502.00592](https://huggingface.co/papers/2502.00592)
- [Primary] **WISE** (2024): lifelong model editing with a dual (side) parametric memory and knowledge sharding. — [HF 2405.14768](https://huggingface.co/papers/2405.14768)
- [Primary] **MAC** (2024): online adaptation to unseen documents through a memory of amortized contexts. — [HF 2403.04317](https://huggingface.co/papers/2403.04317)

**Sleep / replay / consolidation into weights**
- [Primary] **"Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories"** (Behrouz & Mirrokni, Google; arXiv June 2026; on OpenReview since Sept 2025).
  - Proposes a "Sleep" paradigm that distils "short-term fragile memories into stable long-term knowledge with replay".
  - Sleep has two stages:
    1. Memory Consolidation, or "Knowledge Seeding": upward distillation of a smaller-self into a larger network, using on-policy distillation plus RL-based imitation.
    2. Dreaming: RL generates a synthetic curriculum "to rehearse new knowledge and refine existing capabilities".
  - Evaluated on long-horizon, continual-learning, knowledge-incorporation and few-shot tasks.
  - Source: [HF 2606.03979](https://huggingface.co/papers/2606.03979)
- [Primary] **Nested Learning / Hope** (Behrouz et al., Google; arXiv Dec 2025): a "Continuum Memory System" of modules updated at different frequencies, plus a self-modifying sequence model. Hope is the resulting continual-learning module, showing "promising results in language modeling, knowledge incorporation… continual learning". The Sleep paper describes Nested Learning as addressing online consolidation only, and adds that it "still uses the same amount of model's capacity". Google's blog post on Nested Learning could not be fetched (research.google blocked). — [HF 2512.24695](https://huggingface.co/papers/2512.24695); [HF 2606.03979](https://huggingface.co/papers/2606.03979)
- [Primary] **"Language Models Need Sleep"** (Lee, McLeish, Goldstein, Fanti; CMU/UMD, May 2026): the model periodically converts recent context into persistent fast weights in SSM blocks before clearing the KV cache, running N offline recurrent passes. This is a long-context mechanism, not lifelong knowledge storage. — [HF 2605.26099](https://huggingface.co/papers/2605.26099)
- [Primary] **Auto-Dreamer** (UIUC/UCSD, May 2026): a learned offline consolidator for a textual memory bank, trained with GRPO. It gains 7 points on ScienceWorld with a 12× smaller memory bank. This is non-parametric. — [HF 2605.20616](https://huggingface.co/papers/2605.20616)
- [Primary] Counter-evidence on text consolidation, "Useful Memories Become Faulty When Continuously Updated by LLMs" (May 2026): LLM-rewritten memory banks first help, then degrade, and can fall below the no-memory baseline. "Even when consolidating from ground-truth solutions, GPT-5.4 fails on 54% of a set of ARC-AGI problems it had previously solved without memory". — [HF 2605.12978](https://huggingface.co/papers/2605.12978)

**Agent fast-weight memory and context-to-weights**
- [Primary] **TMEM, "Scaling Self-Evolving Agents via Parametric Memory"** (Alibaba Qwen-Character + PKU, June 2026): when the context budget is reached, the agent "distills the current session into grounded QA-style supervision" and applies an online SFT update to fast LoRA weights Δt. The base θ0 stays fixed within the rollout, and RL trains the extraction policy. Evaluated on LoCoMo, LongMemEval-S, multi-objective search and CL-Bench, it beats summary and retrieval baselines. The scope is within a single episode, not lifelong. — [HF 2606.04536](https://huggingface.co/papers/2606.04536)
- [Primary] **Doc-to-LoRA** (Sakana AI, Feb 2026): a hypernetwork generates a LoRA from a document in one forward pass, so "subsequent queries [are] answered without re-consuming the original context". It achieves near-perfect needle-in-a-haystack accuracy at more than 4× the native context window. The authors envision "frequent knowledge updates and personalized chat behavior". — [HF 2602.15902](https://huggingface.co/papers/2602.15902)
- [Primary] **PERK** (2025): long-context reasoning as test-time learning in a low-rank adapter. **In-Place TTT** (Apr 2026): updates the MLP final projection during inference. — [HF 2507.06415](https://huggingface.co/papers/2507.06415); [HF 2604.06169](https://huggingface.co/papers/2604.06169)
- [Primary] "Learning, Fast and Slow" (May 2026): the fast weights are optimized context (prompt evolution) and the slow weights are RL-updated. The paper reports reduced forgetting. — [HF 2605.12484](https://huggingface.co/papers/2605.12484)

**Lab and industry continual-learning claims, 2025–2026 (by reliability)**
- [Primary via snippet] Thinking Machines Lab, "On-Policy Distillation" blog: an assistant fine-tuned on internal company documents loses instruction-following. On-policy distillation, with an earlier checkpoint of the model as teacher, restores it "without regressing" the new knowledge. The post pitches alternating fine-tuning and distillation as a path to continual learning, and says LoRA personalization "learns less knowledge and still forgets". — [Thinking Machines blog](https://thinkingmachines.ai/blog/on-policy-distillation/) (fetch blocked)
- [Secondary, treat with caution] A headline reads "Thinking Machines Labs Claims Solution to Catastrophic Forgetting". It is unverified and could not be fetched. — [HyperAI](https://hyper.ai/en/headlines/da0745de589cc1668c6b7effe87260a5)
- [Rumour/unverified] A prediction newsletter lists OpenAI, SSI and Thinking Machines as focused on continual learning, and says "full, general continual learning won't arrive in 2026". I found no primary SSI statement or release. — [nextsignalprediction Substack](https://nextsignalprediction.substack.com/p/agi-2026-are-we-the-final-white-collar)
- [Primary, opinion] Dwarkesh Patel called continual learning a "huge bottleneck" in June 2025. His 2026-08-07 essay "8 Predictions for the Era of Continual Learning" argues that agents writing Markdown notes between sessions cannot replace experience accumulated into weights, and raises regulatory issues if base models update daily. — [Dwarkesh, June 2025](https://www.dwarkesh.com/p/timelines-june-2025); [Dwarkesh, "8 Predictions…"](https://www.dwarkesh.com/p/era-of-continual-learning); [X post](https://x.com/dwarkesh_sp/status/2085781456375218232)
- [Primary via snippet] Cursor's real-time RL (see Q1) is the one verified production example of frequent post-deployment weight updates.

### Inferences
- The mechanisms in the target design all exist separately:
  - self-generated augmentation: SEAL, Self-Tuning, EntiGraph, prompt distillation
  - web-to-weights: ALAS
  - sparse low-interference slots: sparse memory finetuning
  - sleep consolidation with replay or dreaming: Behrouz & Mirrokni
  - forgetting control: SDFT and on-policy distillation
- SEAL and the sparse-memory paper both frame their work as a step toward continual learning. They measure forgetting in isolation; neither is deployed.
- The Google "Sleep" paper is the closest published match to a "sleep consolidation into weights" component. It distils into a larger network, which grows capacity, and that conflicts with a fixed on-device budget.
- ALAS is the closest published match to "look it up on the web once, then write it into weights". It is curriculum-driven, not triggered by user queries, and it runs on cloud APIs.

### Gaps
- Sparse memory finetuning: I did not verify whether the paper uses synthetic paraphrase augmentation for its fact stream.
- ALAS: I did not verify whether the paper measures forgetting on general benchmarks after repeated updates.
- I could not read Google's Nested Learning blog or the Thinking Machines blog directly.
- I found no primary evidence of any 2026 lab product (OpenAI, Google, Anthropic, SSI, Thinking Machines) that learns per-user knowledge into weights after deployment. Rumours exist, but no releases.

---

## Q3. Personalization through per-user weights (per-user LoRA/PEFT) and on-device continual personalization

### Takeaway
Per-user parametric personalization is well established in research. Examples are OPPU ("One PEFT Per User", EMNLP 2024), PLUM (Apple, 2024), which puts user conversations into a LoRA through QA augmentation, and MinT (2026), which treats PEFT adapters as persistent personal state at million-user scale. On-device work exists (MemLoRA from Samsung, the MobileMem benchmark), but its memory content stays non-parametric. Evidence from 2026 shows that sequential LoRA personalization on the edge is unstable.

### Cited Findings
- [Primary] **OPPU**, "Democratizing Large Language Models via Personalized Parameter-Efficient Fine-tuning" (Tan et al., EMNLP 2024): "employing personalized parameter-efficient fine-tuning (PEFT) modules to store user-specific behavior patterns and preferences. By plugging in personal PEFT parameters, users can own and use their LLMs individually". It integrates parametric personal PEFT with non-parametric retrieval and profiles. — [OPPU GitHub README](https://github.com/TamSiuhin/OPPU) (arXiv 2402.04401)
- [Primary] **PLUM**, "On the Way to LLM Personalization: Learning to Remember User Conversations" (Magister et al., Cambridge + Apple, Nov 2024):
  - Up-samples conversations into positive and negative QA pairs.
  - Trains a LoRA with weighted cross-entropy, one conversation at a time and in time order, because "per-user personalization is only viable in parameter-efficient settings".
  - Achieves 81.5% accuracy across 100 conversations, "competitive with baselines such as RAG".
  - Source: [HF 2411.13405](https://huggingface.co/papers/2411.13405)
- [Primary] **"On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters"** (Mind Lab, June 2026):
  - Treats adapters as "persistent local state on top of strong shared foundation models" that carry "preferences, skills, tool habits, and memory-like updates".
  - Organizes the problem along three axes: Scale Up, Scale Down and Scale Out.
  - MinT is its infrastructure for adapter identity, revision, provenance, evaluation and serving residency.
  - Source: [HF 2606.02437](https://huggingface.co/papers/2606.02437)
- [Primary] Hypernetwork personalization, "Instant Personalized LLM Adaptation via Hypernetwork" (2025/2026): maps user profiles to adapter parameters without per-user training. — [HF 2510.16282](https://huggingface.co/papers/2510.16282)
- [Primary] **MemLoRA** (Samsung R&D UK + TUM, Dec 2025):
  - An on-device memory system in which small models carry specialized adapters for knowledge extraction, memory update and memory-augmented generation, "without cloud dependency".
  - It outperforms Gemma2-27B and is comparable to GPT-OSS-120B on LoCoMo. MemLoRA-V scores 81.3 vs 23.7 on visual QA.
  - The memories themselves are still stored and used "as context". The adapters are memory-operation skills, not the user's facts.
  - Source: [HF 2512.04763](https://huggingface.co/papers/2512.04763)
- [Primary, metadata] **MobileMem** (Aug 2026): a benchmark and framework for "on-device long-term memory through year-scale, multimodal mobile experience trajectories", covering temporal reasoning, knowledge updating and preference inference. Its error analysis compares RAG and long-context memory systems. — [HF 2608.13606](https://huggingface.co/papers/2608.13606)
- [Primary, metadata] "Continual Learning for Sequential Personalization of Small Language Models: A Stability Monitoring Analysis" (June 2026): "Sequential LoRA personalization of small language models reveals hidden instability patterns… highlighting risks of catastrophic forgetting during continual edge deployment." — [HF 2606.27634](https://huggingface.co/papers/2606.27634)
- [Primary, metadata] Other on-device and portable personalization work:
  - Crayon (2024): on-device adapter blending plus edge-server hybrid inference.
  - PortLLM (2024): training-free portable model patches.
  - Federated fine-tuning survey (2025).
  - Sources: [HF 2406.07007](https://huggingface.co/papers/2406.07007); [HF 2410.10870](https://huggingface.co/papers/2410.10870); [HF 2503.12016](https://huggingface.co/papers/2503.12016)
- [Primary, metadata] Persona-Plug (2024): personalizes through user-specific embeddings, not fine-tuning. — [HF 2409.11901](https://huggingface.co/papers/2409.11901)
- [Primary, metadata] Risk: "When Personalization Misleads" (Jan 2026) finds that personalized LLMs can hallucinate content aligned with the user's history instead of factual truth. — [HF 2601.11000](https://huggingface.co/papers/2601.11000)

### Inferences
- "Per-user adapter holds the user's facts, frozen shared base" is prior art: OPPU, PLUM and MinT. A design cannot claim novelty for per-user LoRA or PEFT memory as such.
- PLUM is the closest prior work for "things the user teaches in conversation become weights". It already uses QA augmentation and time-ordered training. It lacks web learning, sleep/replay consolidation, an on-device demonstration and lifelong-scale forgetting evaluation.
- On-device personal memory research (MemLoRA, MobileMem) still stores content outside the weights. On-device parametric fact-learning appears unexplored in deployment, and 2026 evidence (2606.27634) warns of instability.

### Gaps
- I did not find Apple, Google or Samsung shipping docs for on-device per-user fine-tuning of LLMs. I did not search patents, and patent prior art (e.g., on-device personalization patents) is a likely blind spot.
- I could not get PocketLLM-style on-device fine-tuning papers (derivative-free, 2024) from the HF index. Their status is unverified.

---

## Q4. Is there any published system that combines everything? Component matrix and closest prior work

### Takeaway
I found no published system or product that combines all of the target components:
- (A) frozen base
- (B) sparse or parametric memory written after deployment
- (C) self-generated augmentation of new facts
- (D) periodic replay/"sleep" consolidation
- (E) web-search-triggered learning
- (F) learning facts the user teaches in conversation
- (G) on-device execution
- (H) lifelong retention / anti-forgetting evaluation
- (I) answering from weights with no retrieval or context

The closest candidates each cover three to five components: ALAS, SEAL, TMEM, PLUM, sparse memory finetuning, and Google's "Language Models Need Sleep". A novelty claim is defensible only for the integrated combination, especially G, E and F together with measured "search once". It is not defensible for any single mechanism.

### Cited Findings
Component matrix. "Y" means present, "N" absent, "~" partial, "?" not verified. Sources are in the last column and in Q2/Q3.

| Candidate | A frozen base | B post-deploy parametric memory | C self-generated augmentation | D replay / sleep consolidation | E web-search → weights | F user-taught facts → weights | G on-device | H lifelong / forgetting eval | I answers without retrieval | Source |
|---|---|---|---|---|---|---|---|---|---|---|
| **ALAS** (2025) | ? (API fine-tuning) | Y (SFT+DPO) | Y (agent-distilled QA) | ~ (iterative re-eval / curriculum, no replay stated) | **Y** (curriculum-driven, not query-triggered) | N | N (OpenAI APIs) | ? | Y (post-cutoff QA 15%→90%) | [HF 2508.15805](https://huggingface.co/papers/2508.15805) |
| **SEAL** (2025) | ~ (LoRA in single-passage) | Y | **Y** (RL-trained self-edits) | N | N | N | N | ~ (shows forgetting under sequential edits) | Y (no-context SQuAD 47.0%) | [HF 2506.10943](https://huggingface.co/papers/2506.10943) |
| **Sparse memory finetuning** (Meta FAIR, 2025) | Y (only memory slots updated) | **Y (sparse slots)** | ? | N | N | N | N | **Y** (NQ drop 11% vs 71% LoRA / 89% full FT) | Y | [HF 2510.15103](https://huggingface.co/papers/2510.15103) |
| **LMs Need Sleep** (Behrouz & Mirrokni, Google, 2026) | N (distils into larger net) | Y | Y ("Dreaming" synthetic curriculum) | **Y** (consolidation + replay) | N | N | N | Y (continual-learning tasks) | Y | [HF 2606.03979](https://huggingface.co/papers/2606.03979) |
| **Nested Learning / Hope** (Google, 2025) | N (architecture) | Y (continuum memory) | N | ~ (online consolidation only) | N | N | N | ~ | Y | [HF 2512.24695](https://huggingface.co/papers/2512.24695) |
| **TMEM** (Alibaba, 2026) | Y (within rollout) | Y (fast LoRA Δt) | Y (self-distilled QA) | N | ~ (search-agent trajectories) | ~ (conversational memory benchmarks) | N | N (episode-scoped) | ~ | [HF 2606.04536](https://huggingface.co/papers/2606.04536) |
| **PLUM** (Apple/Cambridge, 2024) | Y | Y (per-user LoRA) | Y (QA up-sampling) | N | N | **Y** | N (not shown) | ~ (sequential conversations) | Y (81.5%, ≈RAG) | [HF 2411.13405](https://huggingface.co/papers/2411.13405) |
| **OPPU** (EMNLP 2024) | Y | Y (per-user PEFT) | N | N | N | ~ (user history, not taught facts) | N (ownership framing) | N | N (combined with retrieval) | [OPPU README](https://github.com/TamSiuhin/OPPU) |
| **MinT / PEFT scaling** (2026) | Y | Y (persistent adapters) | ? | ? | N | ~ | N | ? | ? | [HF 2606.02437](https://huggingface.co/papers/2606.02437) |
| **MemoryLLM / M+** (2024–25) | ~ | Y (latent memory pool, not gradients) | N | N | N | N | N | Y (~1M updates, 160k-token retention) | Y / ~ (M+ adds retriever) | [HF 2402.04624](https://huggingface.co/papers/2402.04624), [HF 2502.00592](https://huggingface.co/papers/2502.00592) |
| **Doc-to-LoRA** (Sakana, 2026) | Y | Y (generated LoRA) | N (hypernetwork) | N | N (could be) | ~ (envisioned) | N | N | Y | [HF 2602.15902](https://huggingface.co/papers/2602.15902) |
| **Self-Tuning** (2024) | N | Y | Y | N | N | N | N | ~ (prior-knowledge retention) | Y | [HF 2406.06326](https://huggingface.co/papers/2406.06326) |
| **MemLoRA** (Samsung, 2025) | Y | N (facts stay in text memory) | N | N | N | Y (as text memory) | **Y** | N | N | [HF 2512.04763](https://huggingface.co/papers/2512.04763) |
| **Letta sleep-time agents** | Y | N (text memory blocks) | ~ | **Y** (text) | ~ (via tools, not to weights) | Y (text) | N | N | N | [Letta docs](https://docs.letta.com/guides/agents/architectures/sleeptime/), [HF 2504.13171](https://huggingface.co/papers/2504.13171) |
| **Cursor real-time RL** (prod.) | N | Y (global weights) | N | N | N | N (population RL) | N | ~ (eval gating) | n/a | [Cursor blog](https://cursor.com/blog/real-time-rl-for-composer) |
| **Thinking Machines OPD personalization** (blog) | N | Y | N | ~ (alternating FT + distill-to-recover) | N | N (company docs) | N | ~ | Y | [TML blog](https://thinkingmachines.ai/blog/on-policy-distillation/) |

- [Primary] TMEM explicitly argues that prompt-space memory agents "can *look up* what they have seen but cannot *learn from* it". This is the same motivation as the target design, but its answer is episode-level fast weights. — [HF 2606.04536](https://huggingface.co/papers/2606.04536)
- [Primary] The ALAS authors describe RAG as something that "effectively outsources memory to an external database and does not teach the model new facts". This is the same framing as the target design. — [HF 2508.15805](https://huggingface.co/papers/2508.15805)
- [Primary] The SEAL authors envision that "after an interaction, the agent could synthesize a self-edit which triggers a weight update". This is a stated future direction, not an implemented system. — [HF 2506.10943](https://huggingface.co/papers/2506.10943)

### Inferences
- Closest prior work for each sub-claim:
  - Web lookup to weights: **ALAS**.
  - Self-generated augmentation that is learned: **SEAL**; that is prompted: Self-Tuning, EntiGraph, prompt distillation.
  - Sparse, low-forgetting post-deployment memory: **Sparse memory finetuning**, with WISE and MemoryLLM as alternatives.
  - Sleep/replay consolidation into weights: **Behrouz & Mirrokni 2026**.
  - Conversation-taught facts into per-user LoRA: **PLUM**, then OPPU and MinT.
  - Agent fast-weight memory including search trajectories: **TMEM**.
  - On-device personal memory: **MemLoRA**, which is non-parametric for content.
- No candidate has G (on-device) together with E (query-triggered web learning into weights). No candidate has E together with F (user-taught facts) in one parametric store.
- No candidate reports lifelong-scale retention across both web-learned and user-taught facts.
- Safe novelty wording: "To our knowledge, no prior system integrates query-triggered web-to-weights learning and conversation-to-weights learning in one on-device parametric memory with sleep-style replay consolidation." Each component should cite the prior work above.
- Unsafe wording: claiming novelty for "self-updating LLM", "sleep consolidation for LLMs", "per-user LoRA memory", "sparse memory updates to avoid forgetting", or "learning from web search into weights". Each of these has direct prior art.

### Gaps
- The search covered the Hugging Face paper index, web search and GitHub READMEs. It did not cover patents, closed industry work, or non-English venues.
- Several matrix cells are "?" because I read only abstracts or introductions: ALAS forgetting evaluation and base freezing, sparse-memory augmentation, MinT details.
- I may have missed very recent (Aug–Sept 2026) arXiv papers not yet indexed on HF.

---

## Q5. Is "search once, remember forever" measured anywhere (repeated retrieval falling over time, retrieval-to-parametric distillation)?

### Takeaway
I found no paper or product that measures, over a lifelong or longitudinal deployment, how the rate of repeated web searches or retrievals falls because retrieved knowledge was internalized into weights. Related evidence comes in three kinds:
- One-shot "retrieve or read once, then answer without context" results: ALAS, SEAL, PLUM, prompt distillation, Doc-to-LoRA, PRAG/DyPRAG.
- Amortization results that stay non-parametric: sleep-time compute, Mem0 token savings.
- RL search agents that reduce unnecessary searches by using existing parametric knowledge, without learning new knowledge: IKEA, AdaSearch, SAAS.

This leaves the measurement itself open, and it is a plausible evaluation contribution.

### Cited Findings
**Answering from weights after a one-time read**
- [Primary] ALAS: after web-sourced SFT and DPO, post-cutoff QA accuracy goes "from 15% to 90% on average", with 85–90% on knowledge-updated queries, compared against RAG and fine-tuning baselines. — [HF 2508.15805](https://huggingface.co/papers/2508.15805)
- [Primary] SEAL: 47.0% no-context SQuAD after one self-edit update, vs 33.5% for training on the passage alone. — [HF 2506.10943](https://huggingface.co/papers/2506.10943)
- [Primary] PLUM: 81.5% accuracy on questions about 100 past conversations from a LoRA, "competitive with baselines such as RAG". — [HF 2411.13405](https://huggingface.co/papers/2411.13405)
- [Primary] Prompt distillation: beats SFT, "can even surpass RAG", and PD+RAG > RAG. — [HF 2412.14964](https://huggingface.co/papers/2412.14964)
- [Primary] Doc-to-LoRA: queries are answered "without re-consuming the original context", reducing latency and KV-cache memory. — [HF 2602.15902](https://huggingface.co/papers/2602.15902)
- [Primary] Parametric RAG (PRAG, Tsinghua, Jan 2025) integrates documents "directly into the parameters of feed-forward networks… through document parameterization", saving online cost from in-context injection. DyPRAG (Mar 2025) uses a "parameter translator" to convert documents to parametric knowledge at test time. — [HF 2501.15915](https://huggingface.co/papers/2501.15915); [HF 2503.23895](https://huggingface.co/papers/2503.23895)
- [Primary, metadata] DRAG (2025) distils RAG knowledge from LLMs into SLMs. — [HF 2506.01954](https://huggingface.co/papers/2506.01954)

**Non-parametric amortization**
- [Primary] Sleep-time compute: 2.5× lower average cost per query when offline computation is shared across related queries, and about 5× less test-time compute for equal accuracy. — [HF 2504.13171](https://huggingface.co/papers/2504.13171)
- [Primary] Mem0: more than 90% token savings and 91% lower p95 latency vs full context. — [HF 2504.19413](https://huggingface.co/papers/2504.19413)

**Fewer unnecessary searches, using existing knowledge**
- [Primary, metadata] IKEA (May 2025) uses internal and external knowledge together, "reducing hallucinations and unnecessary retrievals". AdaSearch (Dec 2025) "reduces unnecessary search calls". SAAS (May 2026) handles "over-search mitigation". — [HF 2505.07596](https://huggingface.co/papers/2505.07596); [HF 2512.16883](https://huggingface.co/papers/2512.16883); [HF 2605.29796](https://huggingface.co/papers/2605.29796)

**Benchmarks for streaming or lifelong knowledge**
- [Primary] OAKS (KAIST/UNC/Google et al., Mar 2026): across 14 models, "both state-of-the-art models and agentic memory systems fail to adapt robustly" to streaming, changing facts. — [HF 2603.07392](https://huggingface.co/papers/2603.07392)
- [Primary, metadata] Memora (Apr 2026) is a benchmark for consolidation and knowledge updates in personalized agents. MobileMem (Aug 2026) covers year-scale on-device memory. — [HF 2604.20006](https://huggingface.co/papers/2604.20006); [HF 2608.13606](https://huggingface.co/papers/2608.13606)

**Evidence that internalized knowledge may not be used**
- [Primary, metadata] The "Knowing–Using Gap" (July 2026): "LLMs can quickly memorize new facts, yet fail to use them for downstream reasoning tasks". The paper explains this as knowledge-circuit misalignment, and a heuristic recovers 58–75% of oracle headroom. — [HF 2607.08393](https://huggingface.co/papers/2607.08393)
- [Primary, metadata] "How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM?" (2025): LoRA knowledge updates risk degrading external benchmarks. — [HF 2502.14502](https://huggingface.co/papers/2502.14502)
- [Primary] SEAL: gradual forgetting under sequential self-edits (see Q2). — [HF 2506.10943](https://huggingface.co/papers/2506.10943)

### Inferences
- A "search once, remember forever" evaluation would be new. For example: a longitudinal stream of user queries in which a fact is looked up once, then later queries are asked with retrieval disabled. Metrics would be recall-from-weights over time, the fraction of repeated searches avoided, and forgetting of older facts and general skills.
- The closest existing measurements are ALAS (post-update QA without retrieval) and PLUM (parametric vs RAG recall). Neither tracks how retrieval frequency falls over time.
- The Knowing–Using Gap and LoRA-capacity results suggest the evaluation should test more than verbatim recall. It should also test downstream use of internalized facts in reasoning, and multi-hop use.

### Gaps
- I found no production telemetry from any vendor on repeated searches for the same fact per user, and no study quantifying how often assistants re-search facts they have looked up before.
- I found no 2024–2026 paper titled or framed as "RAG-to-weights over time", "knowledge cache distillation into weights" or "retrieval-to-parametric distillation" with a longitudinal retrieval-reduction metric. This may be a search limitation, since HF-index coverage of Aug–Sept 2026 arXiv may be incomplete.
