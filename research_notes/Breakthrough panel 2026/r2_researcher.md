# Round 2: AI research scientist

Labels: **[checked]** verified today, source given; **[recall]** my own knowledge; **[speculation]** my idea. I changed no repository files and did not open `sealed/`. My scratch work is in `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/researcher/`; the script is `rt_relations.py`.

## 1. What I got wrong in round 1

- **I read "Kofi's flatmate, filed right 11%" as evidence for deferring identity decisions.** Test 2 never uses a relation before stating it (moderator-confirmed). So that 11% measures failed lookups against a noisy index, not decisions made too early.
- **I put the main fix on the reading side ("bind first, read fast").** The founder's error count (moderator-confirmed) puts 42 of 49 wrong rows in tracking people across messages, while writing. My fix addresses answer cost only; I underweighted writing.
- **"Unbinding when the evidence is deleted" is not open research.** Once the memory is recomputed from the raw messages, it is a test, not a contribution.
- **I drop my runner-up, the shower test on science papers.** The neuroscientist's late-connection test fits the owner's case better.

## 2. What moved me, and the novelty check

**"Describe, don't decide" has four parts:**
1. the model writes with no memory in view;
2. it writes each mention as a description ("Kofi's flatmate");
3. code works out who everyone is from all descriptions together, in any order;
4. deletion is done by recomputing, checked by an equality test.

**No system I found does all four [checked: six searches].** The closest:

| System | Has | Lacks |
|---|---|---|
| **DeepDive** (VLDB 2015) [checked] | The closest structure: each document read alone, inference over all candidates together, kept up to date incrementally | Exact resolution; a language model; a deletion check |
| **Lemmalog** (2026 GitHub project) [checked: README] | The closest on the code side: the model only reads in; a rules engine tracks sources, retracts and rebuilds derived facts from a log | The model still proposes which names are the same person; pronouns and roles are dropped |
| **DISCERN** (Nov 2026) [checked] | Never merges confusable people when writing | Identity is settled by an agent loop at question time, not by code |
| **GraphRAG-style extraction** [checked: third-party docs] | Each chunk read with no memory in view | People merged by string name only |
| **Selective replay** (arXiv 2609.04875) [checked: abstract] | Deletion judged against a never-saw-it baseline | Judged by behaviour, not exact equality |

Old roots: Swoosh, order-free entity resolution (2009) [checked]; discourse referents (Kamp 1981) [recall].

**Verdict: a disciplined combination, not a new idea.**

**A catch for the engineer.** Each description also reads the previous message. So deleting a message changes its neighbour's description, and the equality check holds exactly only if the describer gives identical output on every run.

**Use-before-definition: little support.** I found no published measurement for chat. The nearest:
- About half of "the X" phrases in news introduce something new (Poesio & Vieira 1998) [checked].
- A pronoun before its name is about 0.5% of pronouns in written English (Trnavac 2016) [checked: summary].

**So I measured REALTALK** [checked: my count, hand-reviewed]: 10 real conversations, 8,944 messages over 21 days.
- **None of 14 named third parties (9 pets, 5 people) was named before being introduced.** Each name came with its introduction ("my stupid inconsiderate roommate Julio") or one message later.
- **The reverse happens: role first, name later** (4 of 14). "My cat" came about 250 messages before "my cat's name is soufflé".
- **The dominant pattern is roles never named.** There were 200 "my <relation>" mentions. One girlfriend came up about 20 times in three weeks, never named, as "my gf", "your girlfriend" and "her".
- **Caveat:** the participants were strangers. People use bare names when they assume the listener knows the person (Sacks & Schegloff 1979) [checked]. Among friends, though, the definition often never arrives, which deferring cannot fix.

**For the design:** drop use-before-definition as a selling point. The real case is reproducibility, no compounding errors, batched writing, re-reading the past with a better model, and exact deletion.

## 3. Is the convergence real?

**The diagnosis is real:** errors are made while writing. It rests on re-checked numbers (the 42-of-49 breakdown; 95.1% without thinking on full names, against 66.7% on messy chat).

**The agreement on the remedy is close to one vote.** One model read one brief that bolded "WRITER-LIMITED". And "read each document alone, link globally" is the field's textbook pipeline.

**The strongest argument against.** The only evidence for the remedy is a simulation built from the generator's own templates. In that world a mention's kind can be read off its spelling: the resolver's `classify()` splits strings on "'s", "my " and " from ". Real chat's hard cases are not like that. Is "her new friend" (message 377) the same as "the friend she's talking to" (message 936)? That needs judgement over context. Memory-blindness hides the context from the model, and string rules cannot judge.

So the design may only turn wrong answers into "unknown": fewer wrong merges, no better answers. Agreement inside the panel cannot test this; real data can.

**A rival explanation, not yet ruled out:** the writer was an untrained, prompted 4B looking at an index whose links were 57% right. Test: write with the truth index in view. If wrong merges fall to about 1%, the cause is compounding noise, not seeing memory as such.

## 4. Vote

**Run first: "describe, don't decide".** The founder's order-free reader with code binder and my stage 1 are the same experiment. The architect's quick "pre-joined answer, thinking off" check is my stage 0, and becomes its gate 0, because the pass line needs answers without thinking.

| Step | What | Cost | Line |
|---|---|---|---|
| Gate 0 | Truth-built answer cards, 4B thinking off | 1 L40S-hour, 2 days | ≥ 90%, else the speed half is dead |
| Gate 1 | Describer fields, 300 dev messages | ½ hour | ≥ 95% (engineer's line) |
| Main run | Engineer's design and lines, fresh test seed | ~3 hours, 3 weeks | as registered |

**Why first:** it attacks the one blocker the data locates. It is cheap, and its stop line is clear. The from-scratch model and the C5 tests do better once identity works.

**Odds [speculation]:**
- gate 0 passes: 60%; gate 1 passes: 45%;
- the engineer's full pass line, overall: **about 15%**;
- D without thinking beats B with thinking by ≥ 5 points, with an interval above 0: 35%;
- stop line: 30%.

**Guards against the designer's advantage:**
1. Freeze the resolver (commit its hash) and the lines before generating the test world.
2. The moderator, who wrote no resolver code, builds a stress world:
   - unseen forms ("the guy from Kofi's building");
   - roles never named;
   - perspective flips ("your crush");
   - name clashes.
3. Real data:
   - two people label about 400 person mentions in two REALTALK conversations, agreement reported (about 6 hours, under 1 GPU hour);
   - the owner labels 300 of his own messages.
4. Report the share left "unknown". Facts filed right must be at least 85% (B: 78.2%), so the design cannot win by refusing to decide.
5. Any rule edited after the test is seen counts as development and is re-tested on a fresh seed.

## 5. To the owner

**Is anything here 2017-class? No.**
- Every proposal re-arranges known parts, and close relatives exist (DeepDive, Lemmalog, DISCERN).
- The only shift-shaped idea is the architect's "write locally, read globally". Block-Attention already shows it on text that names its subjects; on messy chat it is a long shot.
- 2017 came from removing a bottleneck that real use exposed, proven on a shared test.

**Honest best case for the next month:**
- a 1-hour check showing whether your phone model answers in seconds once the bookkeeping is done for it;
- a writer that confuses people in about 1–2% of facts instead of 6.3%, answers in about 5 seconds instead of 3.5 minutes, and deletes exactly;
- the first real-chat number.

That is a solid systems result, not new science. **The likely case:** filing improves, real-chat references ("the friend she's talking to") stay unresolved, and answers without thinking stall near 75–80%.

**Do now:**
1. Run the 1-hour check this week.
2. Build the describer and binder, with frozen lines and an outside stress world.
3. Label 300 of your own messages.
4. Keep one day in five for long shots.

What I found in real chat: people rarely use a name before saying who it is. They say "my girlfriend" for weeks without naming her. Your memory must handle that, and sometimes only asking you will.

Sources: [DeepDive](https://arxiv.org/abs/1502.00731) · [Lemmalog](https://github.com/JordyZomer/lemmalog) · [DISCERN](https://neurips.cc/virtual/2026/170468) · [Selective replay](https://arxiv.org/abs/2609.04875) · [Swoosh](https://www.microsoft.com/en-us/research/?p=184027) · [GraphRAG-style extraction](https://docs.falkordb.com/graphrag/extraction) · [Poesio & Vieira](https://arxiv.org/abs/cmp-lg/9710007) · [Trnavac 2016](https://scholarlypublications.universiteitleiden.nl/access/item%3A3213756/download) · [Sacks & Schegloff](https://conversationanalysis.org/mp-files/two-preferences-in-the-organization-of-reference-to-persons-in-conversation-and-their-interaction.pdf) · [REALTALK](https://github.com/danny911kr/REALTALK)
