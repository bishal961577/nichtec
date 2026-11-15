# Round 1: AI research scientist

**Labels.** [checked]: I verified it today, in the repository or by web search, and the source is given. [recall]: from my own knowledge. [speculation]: my idea, not yet supported. Most 2026 sources below are arXiv preprints, not peer-reviewed. I read their abstracts or documentation, not the full papers. I wrote no repository files and did not open `sealed/`.

**The prior-art finding comes first, because it changes the whole discussion.** Every part of the "living belief system" has been published, mostly in the last six months. Several have been published together. The newest is DISCERN, submitted four days ago (details in section 4). The field now has thousands of agent-memory papers. A novelty claim from this team has to be very narrow.

## 1. The assumptions list

**What is wrong in it:**
- **A1 is already false in the field.** Knowledge has a third home now.
  - Cartridges (ICLR 2026) trains a small saved attention cache per document collection. It matches reading the text with 38.6× less memory [checked].
  - User as Engram (June 2026) writes each user's facts as local edits to a lookup table inside the model [checked].
  - None of these shows what matters here: two facts stored separately being combined without written-out reasoning. The "two-hop curse" paper (Balesni et al.) found chance-level accuracy on two-step questions when the two facts were learned from different documents, unless the model writes its reasoning [checked]. This project's GPT-J reached 9.4% [checked, brief].
  - So the true form of A1 is: "stored facts combine only if they were active together." That is the wall.
- **A2's "facts in weights can't be deleted" is too strong.** Weights rebuilt from the message store are as deletable as an index. This is exact unlearning by retraining (SISA, 2021) [recall]. The real problems are two: rebuild cost on a phone, and leakage into unrelated answers. Per-user fine-tuning leaked in User as Engram [checked].
- **A3 holds for this project, not for the field.** Offline "sleep" processing is crowded:
  - Sleep-time compute (2025): about 5× fewer tokens at answer time for the same accuracy, but only when questions are predictable [checked].
  - MemDream (Sept 2026): offline self-testing of memory, +4.5 F1 on LoCoMo [checked].
  - Auto-Dreamer (May 2026) [checked].
- **A4's evidence does not support its fix.** A typed language catches form errors ("NEW: He", the fact put in the label). It does not catch filing under the wrong person, which is the real damage: 6.27% wrong merges, and only 11.1% of "Kofi's flatmate"-style facts filed right [checked, `results/messy_index_dev_dev1500/summary.md`]. Typing is hygiene, not the unblocker.
- **A6's evidence is misread.** The oracle index (86.1%) is still retrieval plus reading, and perfect pages (95.5%) beat it. The data say that correct filing matters. They say nothing about "belief state versus storage".
- **A7 is a cost problem more than an accuracy problem.** With thinking on, retractions reached 100% from a prompt change alone. With thinking off on perfect pages, accuracy is 26.7% on retractions, 33.3% on roles and 33.3% on last meetings [checked, same summary].

**What is missing:**
- **M1. "Every interpretation must be final when written."** A5 is one case of this. The writer also had to decide at write time whether a fact lasts or is a one-off, and which relation and which date apply. The thinking writer could not finish even 5-message batches [checked, report §16.9]. That fits a model trying to settle everything in one pass [speculation on the cause].
- **M2. "The perceiver is a frozen, prompted general model."** Every fix in §16 was a prompt or parser patch. The field now trains 4B memory writers:
  - MemReader-4B (April 2026): built on Qwen3-4B and trained with reinforcement learning. It can defer messages whose references are ambiguous [checked].
  - DuoMem (June 2026): a 4B student rose from 4.3% to 77.9% on ALFWorld [checked].
  - This project's generator records the truth for every mention: surface text, person and mention style [checked, `brain/messy_world.py`]. Those are free training labels.
- **M3. "The team writes its own test world, and it is synthetic."** REALTALK (2025) is 21 days of real messaging between real people. Even with the whole history, models scored only 0.19–0.36 F1 on its memory questions [checked]. A result that holds only on `messy_world` will not convince reviewers.
- **M4 (for C5). "Relevance comes from a question."** Noticing without being asked needs a model of what matters to the owner. TriggerBench found that models given more reasoning drift towards "always remind", which means more false alarms [checked].

**The single assumption whose removal would unblock the most:** *the reader works out identities, times and links itself, in its thinking, at question time.* A5, A7 and A8 share this root. The evidence [checked]:
- Test 1's notebook used full names everywhere. There the 4B with thinking off scored 95.1% on the hard set, using about 3 tokens per answer.
- Test 2's messy chat, with perfect pages and thinking off: 66.7%.
- The failures fall exactly on bookkeeping: two-hop 23.5%, retractions 26.7%, roles 33.3%, last meeting 33.3%, nicknames 58.8%.

Removing this assumption would unblock C7: answers in seconds instead of about 3.5 minutes. If the identity decisions stay revisable, it would also help C1 (no false memories) and C3 (association).

## 2. My candidate

**Honest headline: I have no candidate I would bet is a 2017-class shift.** The best I can offer is the one whose failure would teach the most and whose success would change what fits on a phone.

**Candidate: "bind first, read fast".** In plain words:
- Before the phone model reads anything, an engine rewrites memory. Every "she", "my mum" or "Kofi's flatmate" gets the person's full name attached. Each person's page opens with what is true now and since when. Links are already followed.
- This binding (deciding which person or time each mention refers to) is done at night. Every binding keeps its receipts, and it is redone when new evidence arrives or a message is deleted.
- The phone model then answers with thinking off.

**Why it could be more than a re-arrangement.** It changes the cost curve, not a constant: from a median of about 4,200 thinking tokens per memory answer to tens. That is the difference between a memory you talk to and one you wait minutes for. The scientific claim is narrow and easy to kill: *in the 4B's memory answers, the thinking is mostly bookkeeping.* If that is false, the moderator's engine loses its phone-cost argument.

To be clear: the parts are old, and the combination is the moderator's A5+A7+A8. What I add is the cheapest test that could kill it.

**Closest prior art:**
- **DISCERN (11 Nov 2026).** "Preserve at write, resolve at read": it never merges confusable people at write time and resolves who is meant when a question comes. It scored 0.80 on entity resolution where no baseline exceeded 0.40 [checked]. It resolves through an agent loop at question time; my proposal resolves at night and hands the reader finished text.
- **Hindsight (2026 vendor blog).** It resolves identities at write time and creates a new person rather than risk a wrong merge [checked via search snippet].
- **Sleep-time compute** supplies the principle: work done in advance pays off over many later questions [checked].
- **Coreference and entity resolution** (working out which mentions refer to the same person): decades of work [recall].

**Runner-up: a shower-moment test built on the owner's own example, papers.** The literature splits in two:
- Prospective memory is now measured: acting later on an intention or constraint stated earlier. TriggerBench, PM-Bench and ATRBench all do this [checked]. User as Code fires rules such as "new drug conflicts with an allergy noted months ago" [checked].
- Scientific connection-finding is measured on science corpora: Swanson's literature-based discovery, which links two literatures through shared terms and tests on time-cut data [checked]; Science4Cast, which predicts which unconnected concepts will be studied together later [checked]; and MOOSE-Chem, which recovers published hypotheses from a background question plus inspiration papers [checked].
- **What I did not find, in three searches:** a test of unprompted pairing over one person's stream, where neither item is a stated goal, with a false-alarm rate. "Not found" is not "does not exist".

## 3. The smallest decisive experiment ("bind first, read fast")

**Stage 0: can binding alone buy answers without thinking? This stage can only kill the idea.**
- **Built:** code uses the dev world's truth to rewrite the perfect pages:
  - every mention gets the person's full name in brackets;
  - each person gets a "now" header: the current value per slot, its start date, earlier values, and retractions marked;
  - one-hop links are written out ("Kofi's flatmate = Ana Varga, message #812").
  - Three single-ingredient versions (names only, header only, links only) show which part helps.
- **Measured:** the 288 dev1500 hard-set questions; 4B with thinking off, 256-token cap; answer prompt 4 and the existing scorer; results per question kind and median answer tokens.
- **Pass:** ≥ 88% overall; two-hop, retractions, roles, last meetings and nicknames each ≥ 75%; median ≤ 50 tokens.
- **Stop:** below 80%. Thinking is then doing more than bookkeeping, so binding first is not the route to phone speed.
- **Caveat:** the header nearly contains the answers, so a pass shows the ceiling, not feasibility. My forecast: 55% pass, 20% stop.
- **Cost:** under 1 L40S-hour; 2–3 days of work.

**Stage 1: can the 4B's own perception plus code do the binding?**
- **Built:**
  - With thinking off, the 4B writes only local, typed claims about each mention. Examples: "mention 'my mum': relation to me = mother"; "mention 'Kofi's flatmate': flatmate of Kofi".
  - Code normalises dates.
  - A code solver groups mentions into people. It uses hard rules (one mother; declared nicknames) and soft scores (name similarity, shared context, how recent a pronoun's candidate is).
  - A mention whose best match is not clearly better than the next stays unbound, and pages show it as "Kofi's flatmate (not yet known)". This is deferred reference.
  - Everything is re-solved from scratch every night, so deleting a message also removes any binding that rested on it.
- **Measured:** on the sealed test world, under the existing freeze rules:
  - everything test 2 registered (hard set, association kinds, wrong merges, filing by mention style, forgetting, deletion leaks, guessing on never-stated facts);
  - median answer tokens and the share of mentions left unbound;
  - one new deletion check: delete the message that bound a "via" mention and confirm its facts return to "not yet known".
- **Arms:**
  - bind-first with thinking off, and with thinking on;
  - B with thinking on (dev 70.1%) and C with thinking on (62.8%), the strongest internal baselines;
  - perfect pages with thinking on (95.5%) as the ceiling;
  - for publication only: Mem0 run with the same 4B.
- **Pass (all must hold):**
  - hard set ≥ 80% with thinking off, and the association kinds ≥ 70%;
  - wrong merges ≤ 2%; "via" facts filed right ≥ 60%;
  - worst-person forgetting ≥ 95%; 0 deleted values shown; guessing on never-stated facts ≤ 2%;
  - median answer ≤ 100 tokens;
  - bind-first (thinking off) beats B (thinking on), with a people-resampled 95% interval that excludes 0. This interval is the range of the difference when the people in the test are resampled at random.
  - Only after a pass: 300 REALTALK questions.
- **Stop (any one):** hard set below 70% with thinking off; wrong merges above 4%; any deletion leak caused by the design.
- **Cost:** 4–8 L40S-hours (about $10–25 [recall on the price]); about 2 weeks of code; no paid API, because scoring is by code.

**Runner-up experiment (shower test on papers):**
- **Built:** a dated stream from MOOSE-Chem's pool of up to 3,000 papers [checked]. The night shift flags "these two fit" pairs without being given the research question.
- **Measured:**
  - how many of the benchmark's annotated inspiration pairs appear among the 50 pairs flagged each night;
  - false alarms, judged blind by the 27B and on a human-checked sample;
  - baselines: pairs ranked by meaning similarity, pairs sharing a Swanson-style link term, and the 4B reading in chunks at equal compute.
- **Pass:** at least 20% of planted pairs found in the top 50, and at least 2× the best cheap baseline.
- **Stop:** inside the confidence interval of meaning similarity.
- **Cost:** 2–5 L40S-hours; 1–2 weeks. The first day goes to checking that enough benchmark items have two or more inspirations [speculation that the benchmark can be reused this way].

## 4. The moderator's candidate: breakthrough or engineering?

**Verdict: good engineering. In 2026 it is not even a new combination.** Component by component:

| Component | Closest work [checked] | What is left |
|---|---|---|
| Typed memory language; code does the bookkeeping | User as Code (June 2026): typed Python objects plus a log that is only added to. ~99% on counting questions where retrieval gets 6–43%; also gives proactive alerts. NeuSymMS (May 2026): an LLM extracts facts, a rules engine reconciles them; no results reported. ChatDB (2023): SQL database as memory. | All rely on large cloud models as writers. At 4B, MemReader exists. |
| Deferred reference | DISCERN (Nov 2026). BeliefMem (May 2026): several candidate conclusions per observation, each with a probability. MemReader-4B's defer action. TANGLE: keeps unresolved conflicts open. Classic roots [recall]: discourse referents (Kamp 1981), assumption-based truth maintenance (de Kleer 1986), databases with uncertainty and lineage (Trio, 2006). | Not found: unbinding when the evidence for a binding is deleted, with facts flowing back, measured at phone size. |
| Truth maintenance for deletion | Graphiti deletes an edge or node only if no other episode supports it. Its own documentation says node summaries are not regenerated and invalidations survive deletion. AfterDeleteBench (Oct 2026): Qwen2.5-7B recovered "deleted" secrets about two-thirds of the time through derived leftovers, and provenance-aware deletion removed this in the tested settings. A 2026 survey calls this "deletion propagation" an open invariant. | Mostly covered. Deletion through binding is still open. |
| Valid-time facts | Zep/Graphiti (2025) track two timelines per fact. TOKI (June 2026): an algebra of time operators for resolving contradictions. Event calculus (1986) [recall]. | Covered. |
| Night shift | Generative Agents' reflection [recall], sleep-time compute, A-MEM, MemDream, Auto-Dreamer. | Covered for consolidation. Not for unanticipated connections. |
| Shower-moment test | TriggerBench, Pro-Memory, ATRBench (prospective memory); Swanson, Science4Cast, MOOSE-Chem (science corpora). | Unprompted, personal-stream connections with a false-alarm rate: not found. |

The other candidates in the list:
- **Persisted fast-weight state:** LiveMem keeps a fast-weight (DeltaNet) state within one long run [checked]. I found nothing on saving it per person [checked, one search]. Linear-attention layers like these recall exact items poorly [recall].
- **Test-time training and complementary learning systems:** well trodden [recall]. A June 2026 probe found that retrieval wins on facts (0.96–0.97) and slow consolidation into weights wins only on persistent goals [checked].

**What would have to be true for it to be more:**
1. **A notable result, still not a shift:** a phone-size model with thinking off, plus the engine, matches a large model reading everything, on *real* chat such as REALTALK, at ≤ 1/20 of the tokens.
2. **Real deletion through inference:** an exposed-versus-never-exposed twin audit in AfterDeleteBench's style shows zero recovery even when the deleted message had supported a binding or a connection.
3. **A shift, which means C4:** engine-held facts would have to enter the model's computation so that two facts stored apart combine without written reasoning, well above chance (≥ 70%, against 9.4% here), and stay deletable. I found no result showing this. Compiling the belief state each night into something the model reads natively (a cartridge, or lookup-table edits) is the only runnable test I can see. It is a long shot [speculation].

The 2017 analogy also cuts against the owner. The transformer was a learned, general building block that paid off only when trained at scale. One L40S cannot produce that. This team can produce a decisive measurement [speculation/opinion].

## 5. Odds

**"Bind first, read fast":**
- **(a) Measured win over the strongest baseline within these resources: about 30%.**
  - P(stage 0 passes) ≈ 55%.
  - P(stage 1 passes, given stage 0 passed) ≈ 40%: filing "via" mentions is at 11% today, and ≤ 2% wrong merges on messy chat is hard.
  - Against an external system run with the same 4B (DISCERN or Mem0 style): about 20%.
- **(b) Something the field would call a breakthrough: about 1%.** The parts are old. DISCERN and User as Code already show the advantage of structured memory with large models. The ceiling is "the first phone-size system that does this, measured honestly", which is a solid empirical paper if it transfers to real chat.

**Runner-up (shower test):** (a) about 25%; (b) about 2%. A good benchmark can matter, but Science4Cast and MOOSE-Chem cover the science side.

**Sources:** [DISCERN](https://arxiv.org/abs/2611.15729), [User as Code](https://arxiv.org/abs/2606.16707), [NeuSymMS](https://arxiv.org/abs/2605.17596), [BeliefMem](https://arxiv.org/abs/2605.05583), [TOKI](https://arxiv.org/abs/2606.06240), [TANGLE](https://arxiv.org/abs/2608.13921), [AfterDeleteBench](https://arxiv.org/abs/2610.31366), [Graphiti deletion docs](https://help.getzep.com/deleting-data-from-the-graph.mdx), [Zep](https://arxiv.org/abs/2501.13956), [Always-On Agents survey](https://arxiv.org/abs/2606.30306), [Sleep-time compute](https://arxiv.org/abs/2504.13171), [MemDream](https://arxiv.org/abs/2609.34545), [Auto-Dreamer](https://arxiv.org/abs/2605.20616), [TriggerBench](https://arxiv.org/abs/2606.23459), [Pro-Memory](https://arxiv.org/abs/2611.05731), [ATRBench](https://arxiv.org/abs/2605.28108), [MemReader](https://arxiv.org/abs/2604.07877), [DuoMem](https://arxiv.org/abs/2606.29961), [User as Engram](https://arxiv.org/abs/2606.19172), [Selective parametric consolidation](https://arxiv.org/abs/2606.26806), [Cartridges](https://arxiv.org/abs/2506.06266), [Two-hop curse](https://arxiv.org/abs/2411.16353), [REALTALK](https://arxiv.org/abs/2502.13270), [MOOSE-Chem](https://arxiv.org/abs/2410.07076), [Science4Cast](https://www.nature.com/articles/s42256-023-00735-0), [ChatDB](https://arxiv.org/abs/2306.03901), [Hindsight blog](https://hindsight.vectorize.io/blog/2026/06/29/entity-resolution-agent-memory), [LiveMem](https://arxiv.org/html/2608.02515).
