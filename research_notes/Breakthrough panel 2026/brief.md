# Brief for all five panel members (breakthrough panel, November 2026)

## The owner's goal
A "small AGI": an AI that learns continually after deployment.
- **Accountable:** exact memory, receipts, real deletion.
- **One per person, on a phone.**
- **Non-profit, free of big-company control.**
- **Memory is the first step.**

## The owner's question to this panel, in his words (lightly edited)
> "Before 2017, everyone thought today's kind of AI was impossible; one new idea changed everything. If people had kept
> using the old architectures and frameworks, it might still look impossible today. I want that for the brain: not my
> own Qwen, but a new way for memory and learning to work, thought from the beginning. Humans connect things across
> time: I read one paper today and another ten weeks later, and in the shower I realise they fit together. Today's AI
> can't. Maybe we have to build the whole architecture from first principles, not rely on current frameworks."

He asked the moderator to:
- list the assumptions today's AI memory rests on, find which one is blocking, and propose replacements with tests
  that prove or kill them (`assumptions_v1.md`, in this folder: read it first);
- have the panel judge honestly whether anything here can be a breakthrough.

## The project's record (all in this repository: `/home/user/nichtec`; read what you need)

| Work | Where | Result |
|---|---|---|
| Founding thesis | `AGI_BOTTLENECK.md` | Gradient updates don't commute, so AI can't learn in the order life happens. SPARC (sparse, local, commutative memory) works on small image streams, not on language. |
| Counting-only language model | `COUNTING_AI.md` | Lost to a small neural network. |
| Additive feature learning | `reports/Additive feature learning 2026.md` | Route ended at its stop line. |
| Keyed memory around frozen models (0.5B, GPT-J) | `FIXING_THE_BOTTLENECKS.md`, `reallm/` | GPT-J: 96% single-fact recall after 2,764 edits and exact deletion, but combining stored facts 9.4%. The model's own hidden states can't find stored names (63–78%). |
| Order gap | `reports/Order gap 2026.md` | 27B full-history agent learned rules; the ledger did not help. Route ended. |
| Real data (MovieLens) | `reports/Real data test 2026.md` | Learning a person fast needs a representation learned in advance. |
| Expert panel 1 | `reports/Expert panel 2026.md` | Nobody can turn a few experiences into knowledge used as if always known. A new fact joins the model's thinking only if it was active together with what it must combine with, or the network was trained to read its memory. |
| Test 1, notebook reading | `reports/Notebook reading at shipping size 2026.md` | Qwen3.5-4B (phone size) reads the right pages at 99.9% but fails to read a whole 5,000-note notebook (20%). Cloud model: whole notebook 94%. |
| Memory road panel | `reports/Memory road panel 2026.md` | Long-term memory = raw notes + an add-only index + a small recent window; test the owner's association idea. |
| Test 2, messy-chat index, development phase | `reports/Messy chat index test 2026.md`, sections 14–16; `brain/messy_world.py`, `brain/messy_index.py`; `results/messy_index_dev_dev1500/` | See below. |

**Test 2 world.** A generated chat history (6,292 messages, 260 people) with:
- pronouns, nicknames, "my mum";
- "Kofi's flatmate", second-hand news;
- role changes, corrections, retractions, deletions;
- near-name lures.

**Test 2, first 1,500 dev messages, real Qwen3.5-4B, hard set (288 questions):**

| Arm | Thinking on | Thinking off |
|---|---|---|
| Perfect pages | 95.5% | 66.7% |
| Truth index | 86.1% | – |
| **The 4B's own index (the owner's association idea)** | **70.1%** | 50.3% |
| Same index without links | 67.4% | – |
| Search agent | 62.8% | 39.2% |
| Equal-cost search | 50.0% | 29.9% |

What else test 2 showed:
- **Wrong person:** 6.3% of facts filed under the wrong person; "Kofi's flatmate" filed right 11–22%.
- **Forgetting:** "forget X" removes 89% of their messages on average, 62% for the worst person.
- **Guessing:** the index never guessed a never-stated fact; search did 35–38% of the time.
- **Cost of thinking:** answers with thinking use a median of about 4,200 tokens. Thinking-off fails because the 4B
  doesn't connect identities across messages.
- **The thinking writer** ran past 8,192 tokens and wrote nothing, even on 5-message batches.
- **Verdict:** WRITER-LIMITED.

## Resources
- Modal credit for one L40S (48 GB) at a time; the Qwen3.5-4B server exists.
- The owner's laptop (RTX 3050, 6 GB).
- No paid API key.
- A small team, no deadline.
- Anything you propose to run must fit this.

## Rules for every panel member
1. **Be brutally honest.** No flattery of the owner, the moderator or each other. If the honest answer is "this is not a
   breakthrough" or "this is known", say so and say why.
2. **Label every claim:**
   - **[checked]**: you verified it today, in the repository or by web search; give the source.
   - **[recall]**: from your own knowledge.
   - **[speculation]**: your idea, not yet supported.
3. **Check prior art with web searches before calling anything new.** Name the closest existing work.
4. **Every proposal ends in an experiment:**
   - what is built;
   - what is measured;
   - the pass line and the stop line;
   - rough cost in GPU hours and days of work, within the resources above.
5. **Don't modify repository files.** Scratch work goes under the session scratchpad, in your own subfolder
   `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/<your role>/`.
   Don't call the Modal endpoint.
6. **Don't open `research_notes/Messy chat test review 2026/sealed/`.** It holds sealed test material.
7. **Plain language.** The owner is not a specialist. Explain any technical term the first time you use it.
