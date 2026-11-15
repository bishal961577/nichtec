# Round 1: the first-principles engineer

I read the record, ran three CPU scripts (no model, no Modal) and searched for prior art. The scripts and their outputs are in `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/engineer/` (`resolver_sim.py`, `harsher_noise.py`, `phone_budget.py`, each with an `_out.txt` file, plus this text as `round1.md`).

## 1. The assumptions list

### Start from the physics

- A decade of this user's chat at test-2 density is about 34,000 messages and about 2 MB of text [checked: arithmetic from the dev world].
- Each token the 4-bit 4B generates streams about 2.5 GB of weights from memory and costs 0.26–0.4 J [checked: PowerInfer-2 measured 0.257 J/token on a phone, arXiv 2406.06282].
- A full scan of a decade's search codes moves about 1.6 MB. So one generated token moves as many bytes as about 1,500 whole-memory scans [checked: arithmetic, `phone_budget.py`].
- Today's answer (B, thinking on) generates a median of 4,172 tokens [checked: dev summary]. That is about 220 s and 2% of a battery per question [speculation: assumes 20 tokens/s and 0.3 J/token].

**On a phone, memory is almost free. Generated tokens are the cost.** Spend model tokens only where a model is needed: understanding a message and phrasing an answer. Everything with an exact specification goes to code, and is computed once rather than at every question.

### What is wrong

- **A1 overstates weights.** Facts learned in separate documents do not combine without written steps. Facts in the same prompt do [checked: Balesni et al., "two-hop curse", arXiv 2411.16353; a revision is announced]. The context window is what integrates best. The real issue is what gets put in front of the model, and at what cost.
- **A1's fast-weight state is not a memory.** The DeltaNet state is the small working memory that 24 of the 4B's layers update as they read. It is overwritten during reading and depends on reading order. You can only delete from it by recomputing it. It can serve as a cache rebuilt from the memory, never as the memory [recall].
- **A4 blames the wrong thing.**
  - Grammar-constrained decoding fixes syntax for free: the server only lets the model produce text that fits a fixed format [checked: vLLM structured outputs].
  - The writer failed because it made **identity decisions with the index in view**. It picked an id such as P17 from about 66 people, using link rows that were right 57.1% of the time [checked: dev summary].
  - A typed language in which the model still chooses person ids would fail the same way.
- **A5 misreads the evidence.**
  - Test 2's world uses "Kofi's flatmate" only after the relation has been stated. It uses a nickname only after it was introduced, and a bare first name only when it is clear at that moment [checked: `messy_world.py` lines 603–617].
  - So "Kofi's flatmate" at 11.1% was a failed lookup, not a binding decided too early. With a perfect writer's index, the right person would have been on the list 95% of the time [checked: report §15.1].
  - Deferred reference matters in real chat, but test 2 cannot measure it.
- **A6 does not need truth maintenance at this size.** A full re-solve of the 6,292-message dev world took 0.04 s in Python on one core [checked: `resolver_sim.py`]. Deletion becomes recomputation, which is exact by construction and can be checked by an equality test.
- **A10 speeds up the part that already costs microseconds.** The bottleneck is streaming the weights for every token.
- **C4 is defined against an ideal that weights do not meet** (see A1). A measurable version has three parts:
  - reaching the fact costs nothing;
  - it shapes answers that do not mention it;
  - it combines with other facts in a few reasoning tokens, not thousands.

### What is missing

- **M1. The writer is sequential and reads the memory it writes.** Each batch's prompt depends on everything filed before it. The report records the consequences [checked: §16.3–16.4]:
  - two runs on the same 400 messages differ by 10 points;
  - saved replies cannot be re-parsed;
  - identical code gave 5.5% and 6.3% wrong merges.

  This rebuilds the founding problem inside the writer: updates do not commute, errors compound, nothing is revised, and nothing runs in parallel.
- **M2. Every question re-derives the world.** Identity, current values and dates are worked out again in about 4,200 thinking tokens per answer.
  - With full names written out, the 4B answered at 95.1% without thinking (test 1).
  - With messy names and perfect pages, it answered at 66.7% [checked: report §16.7].
- **M3. The test world is kind.** Definitions always come before use, and questions announce "use memory now" (A9).

**The single assumption whose removal unblocks most is M1.** The replacement rule: the model's output is a pure function of the message and a fixed small window before it. "Pure function" means the same input always gives the same output, whatever is stored. Everything that spans messages then becomes a deterministic function of the whole set of outputs. That one rule unblocks:
- identity (C1, C3);
- exact deletion (C2);
- deferred reference, at no extra cost;
- batching (C7);
- reproducibility;
- re-reading the whole life when a better model ships.

## 2. Design: "describe, don't decide"

**The model never sees the memory while writing.** For each message it writes a description:
- who is mentioned, *as written*: "Kofi's flatmate", "my old boss", "Mira from work", or "she" meaning the person in the previous message;
- what is claimed about them.

Code decides who everyone is, from all descriptions at once, and recomputes whenever the set changes.

### Data structures

| Store | Holds | Decade size [speculation] |
|---|---|---|
| Journal | raw messages, append-only | ~2 MB |
| Descriptions | one per message, keyed by describer version + hash of the message and its predecessor | ~4 MB |
| View (never edited, only recomputed) | people; each mention → person, "ambiguous" or "unknown"; facts with the span of time they hold; relations; roles with dates; events. Every row lists the message ids it rests on (receipts) | ~5–10 MB |
| Rulings | the model's answers to leftover "which Mira?" questions, with the message ids it read | small; void when any of those messages is deleted |
| Standing items | wishes, plans, open questions (from the same descriptions) | small |

### The description language (what the model emits)

```
#N is  REF slot = value [(plan|wish|tried|temp|past|other)] [via REF]    tagged claims never become facts
#N not REF slot     #N fix REF slot = value     #N rel REF's noun = REF [both]
#N aka REF = name   #N from REF = circle        #N role REF = noun [new]
#N did activity with REF on "<message's own words>"     #N men REF
REF := @Full Name | @Name | @Name/circle | @my.noun[~old] | REF.noun | ^k  (k-th person of the previous message)
```

A `slot` is a kind of fact (job, pet, city). A `circle` is how I know someone (work, choir).

Examples from the dev world:
- "Shirin Teixeira rents from Tobias Greco." → `rel @Shirin Teixeira's landlord = @Tobias Greco`
- "Shirin's colleague is allergic to mustard." → `is @Shirin.colleague allergy = mustard`
- "Lotta test-drove a Subaru." → `is @Lotta car = Subaru (tried)`
- "Heard from Ottilie Kaya today." / "He adopted a tabby cat." → `men @Ottilie Kaya` / `is ^1 pet = tabby cat`

A description is about 15–30 tokens per message, decoded under the grammar. Its closest relatives are discourse representation theory, where a phrase introduces a referent with conditions on it, and semantic parsing [recall].

### Algorithms

**Write.** One describer call per message, given the message, its date and the previous message. Thinking is off, decoding is greedy, and the result is cached by input hash. Then re-solve.

**Resolve.** This is code. It runs over the messages sorted by date, so arrival order never matters, and repeats until nothing changes (4 rounds were enough).
1. A full name is a person.
2. Mentions already resolved supply the definitions:
   - nicknames;
   - circles;
   - roles, where a `new` holder ends the old holder's span;
   - relations, with two-way ones recorded both ways.
3. A bare name resolves in this order:
   - the only person with that name or nickname known by that date;
   - else the most recently mentioned of them;
   - else it stays "ambiguous", never guessed.
4. `@my.boss` on date d is whoever held the role then; `~old` is the previous holder. `REF.noun` follows the relation. `^k` copies the antecedent.
5. Anything unresolved stays a placeholder ("Kofi's flatmate, unknown"). Its facts attach when a definition arrives, even a later one.

Facts:
- The latest dated statement wins.
- `not` ends the span; `fix` replaces the value and marks the old one as a mistake.
- Tagged claims never become facts.
- Code turns "on Thursday" into a date.

**Recall.**
1. The model (thinking off) writes the question in the same language, for example `? @Omar Lombardi.neighbour allergy` or `? who pet=hedgehog hobby~sailing`.
2. The engine evaluates it and returns an evidence card: the answer rows, their receipts, and the source messages with names filled in ("She [Mira Okafor] adopted a ferret").
3. The model (thinking off), or a template, phrases the answer.

**Delete.**
- Remove the message and its description, recompute the view, and void any rulings that read the message.
- Check: the view must equal the view computed from a journal that never held the message.
- "Forget Kofi" removes every message that has any mention bound to him (by name, role, pronoun, as an anchor or as a source) or contains his names.

**Night, on the charger.**
- If the describer changed, re-describe the journal, decoding several messages together.
- Ask the model, with thinking on, small focused questions about the remaining ambiguities, and store the rulings with their receipts.
- Run the runner-up's join.

### Phone budget [speculation: 20 tokens/s generating, 200 tokens/s reading, 0.3 J per generated token]

| Step | Tokens (in / out) | Time | Energy |
|---|---|---|---|
| Answer today (B, thinking) | ~2,000 / 4,172 | ~220 s | ~1,300 J (2% battery) |
| Answer, proposed (evidence card, thinking off) | ~600 / ~40 | ~5 s | ~30 J (0.05%) |
| Describe one message live | ~70 / ~30 | ~1.9 s | ~11 J |
| Describe at night, 8 messages decoded together | same | ~0.7 s each | ~4 J (50 messages a day: 0.3% battery) |
| Re-solve a decade | none | ~1 s on a phone core | negligible |

Decoding several messages together shares one pass over the weights. Only an order-free writer can do that.

### CPU check: does code lose anything if the model only describes? [checked: `resolver_sim.py`, `harsher_noise.py`]

The descriptions were simulated from the generator's truth. The resolver saw only each mention's words, never person ids. I injected errors:
- facts dropped;
- claims attached to the wrong mention;
- relations reversed;
- wrong antecedents;
- surnames dropped.

| Describer error rate (simulated) | Wrong person among filed facts | "Kofi's flatmate" facts right | Forgetting, worst person |
|---|---|---|---|
| 0% | 0.00% | 100% | 100% |
| 5% | 0.62% | 92.9% | 100% |
| 10% | 1.03% | 85.9% | 92.0% |
| 20%, plus 20% of surnames dropped | 2.02% | 80.6% | – |
| **4B's own index, test 2 dev (measured)** | **6.27%** | **11.1%** | **62.2%** |

Further checks:
- Shuffling the arrival order gives an identical view.
- Deleting every message that defines a nickname unbinds all 228 nickname mentions.
- I also built a world where 40% of nickname and relation messages arrive after the names are used:
  - binding at write time leaves 39% of those nickname facts and 37% of those relation facts unfiled;
  - re-solving the whole set leaves 6% and 2%.

**Limits.**
- I wrote the resolver after reading the generator, so it has the designer's advantage.
- The simulated errors are milder than a real model's.
- This shows the split loses nothing on this world. It does not show that the 4B can write the descriptions.

### Shift or re-arrangement? Prior art

**The case for a shift.** Attention existed before 2017. The Transformer removed the sequential state from the reader, and parallelism and scale followed. This design removes the sequential state from the writer. These properties then hold by construction:
- the order of writes does not matter, which meets the founding thesis at the system level;
- writing runs in parallel;
- the whole life can be re-read overnight by a better model;
- deletion is provably exact;
- every belief has receipts.

**My verdict: a re-arrangement with one sharp rule.** Information extraction has always extracted per document and then linked entities across documents.

Closest prior art I found:
- **DISCERN** (NeurIPS 2026 workshop): "preserve at write, resolve at read". It keeps an add-only graph, with receipts, that records both when a fact held and when it was recorded. It never merges confusable entities and resolves them at query time with an agentic loop [checked: neurips.cc/virtual/2026/170468].
- **KGGen** (NeurIPS 2025): extracts triples, then a model clusters the entity strings [checked: arXiv 2502.09956].
- **Zep/Graphiti:** facts stamped with when they held and when they were recorded, invalidation of outdated facts, and model-based deduplication when writing [checked: arXiv 2501.13956].
- **ESAA and ActiveGraph:** the model emits intentions, and a deterministic projection is built from an event log [checked: arXiv 2602.23193, 2605.21997].
- **Older work:** collective entity resolution (Bhattacharya and Getoor 2007) and DBSP, where deletions are negative weights [checked].

What I did not find:
- a describer that is memory-blind by rule;
- mentions emitted as descriptions to be resolved later;
- derived bindings deleted by recomputation and checked by equality;
- a phone energy budget.

### Runner-up (C5): noticing as a nightly join over resolved people

Once identity is exact, a late connection becomes a join: matching rows from two lists on a shared key, here the same person or people linked to them.
1. **Pair.** Each new fact is paired with facts and standing items learned at least two weeks earlier, about the same person, people linked to them, or me.
2. **Rank.** A table of which kinds of fact interact (allergy × pet, city × trip, job × need), written once by a model, ranks the pairs.
3. **Judge.** The 4B (thinking off) judges the top pairs.
4. **Tell.** At most one item a day, with receipts.

Cost: about 300 pairs a night, about 1.5 kJ on the charger [speculation]. The plans and wishes the fact layer must ignore are exactly the standing items this needs.

Prior art:
- Swanson's ABC literature-based discovery (two literatures linked through a shared middle term) and the Rete algorithm (standing patterns checked against each new fact) [recall];
- ProactiveMemBench and TriggerBench, 2026 [checked: arXiv 2605.28046, 2606.23459, abstracts only];
- sleep-time compute [checked: arXiv 2504.13171].

**Test.**
- **World:** plant 100 pairs that matter together, 2–10 weeks apart, half of which need identity resolved. Add 100 decoys: namesakes, and the same person with unrelated facts.
- **Arms:** the join, against "top-20 search for each new message, same judge".
- **Pass:** ≥ 60% noticed, ≤ 1 false alarm a week, and ≥ 20 points over search on the identity half.
- **Stop:** within 5 points of search.
- **Cost:** 3 days of work, under 1 L40S hour.

## 3. The smallest decisive experiment: the memory-blind writer

**What is built.**
- **The describer:** a prompt plus grammar for Qwen3.5-4B, thinking off, greedy decoding. It is served batch-invariant, meaning the same reply however many requests share the GPU. vLLM reportedly has this mode [checked weakly: one secondary source]. It matters because test 2 saw identical prompts get different replies.
- **The resolver:** generalised to open vocabulary, so the held-out relations and slots need no code change.
- **The question parser and the evidence card.**
- **An adapter**, so that test 2's scorer, `write_quality`, `forget_test`, deletion audit and people-resampled bootstrap apply unchanged.

**Arms.**
- D, thinking off: the decision arm.
- D with thinking, and D answered by template: reported only.
- Test 2's B, C, oracle and pages arms, measured on the same questions.

**Where.**
- Development on test 2's dev world: the first 1,500 messages and the same 288 hard questions.
- The scored run on test 2's beacon-seeded test world with the sealed wordings. It should be registered as an amendment before test 2's FREEZE. That gives one shared run and the strongest guard against my designer's advantage.

**Gate 1 (dev, about 15 GPU minutes).** Score the describer's fields against the truth on 300 messages:
- claim attached to the right mention ≥ 95%;
- relation direction ≥ 95%;
- lures tagged ≥ 90%.

If any field is still below 85% after a week of prompt work, stop before the scored run.

**What is measured.**
- Accuracy on the hard set and the association questions, thinking off and on.
- Filing: wrong merges, hard-case merges, filing by mention style.
- Deletion: per-person forgetting, leaks, guessing on facts never stated.
- Order invariance: describe in reverse order; the view must be identical.
- Derived deletion: delete every nickname, relation and role message; the view must equal a from-scratch recompute.
- Cost: tokens and phone-seconds per message and per answer.

**PASS (test world), all of:**
- D thinking-off ≥ 85% on the hard set.
- D thinking-off ≥ the stronger of B and C *with thinking* + 5 points, with the people-resampled 95% interval above 0.
- desc, who and two each ≥ 75%.
- Wrong merges ≤ 1%; hard-case merges ≤ 5%.
- Every forgetting target ≥ 95%.
- No leaks.
- Both exactness checks hold exactly.
- Median answer ≤ 64 generated tokens.

**STOP:** D thinking-off below 70%, which is no better than B spending 4,200 tokens, or wrong merges above 3%.
- This would mean the 4B cannot describe messages well enough even one at a time.
- One fallback: a LoRA describer (a small trained add-on to the 4B's weights), trained on descriptions from other seeds with paraphrased templates, about 3 L40S hours.
- If that fails too, the describer must be the 27B, run at night.

**BETWEEN:** reported, naming the missed lines.

**Reported only:** a use-before-definition variant, built from a copy of the generator (the frozen file is untouched). I predict binding at write time leaves 35–40% of nickname and "Kofi's flatmate" facts unfiled, against 2–6% for the re-solve.

**Cost.**
- About 3.5 L40S hours (about $7): gate 0.5 h, development 1 h, scored run 2 h. Describing 6,300 messages takes minutes when batched.
- About 3 weeks of one engineer's work.

## 4. The moderator's "living belief system"

It is good engineering, as the moderator says. Three corrections:
1. **Memory-blindness is the cure, not the language.** A typed language in which the model still chooses person ids repeats B's failure.
2. **Recomputation can replace truth maintenance.**
3. **Two parts are not new or not shown.** The night shift is sleep-time compute. "Skills into weights" is unspecified and untested.

What would make it more:
- **The describer learns its own referring language** through the exact engine, from question-answer rewards. The neuro-symbolic concept learner learned visual concepts through a program executor in the same way [recall]. That is still research-grade, not 2017-class.
- **Strong C4: beliefs that combine inside the model's forward pass without written steps.** That needs a reader trained on the memory (the first panel's moonshot). No arrangement of engines gets there, and the two-hop result says weights do not either.

On strong C4, my honest finding is that **no shift is available within this team's resources**. The achievable C4 is a nightly-rebuilt belief summary kept in the model's context:
- reaching it costs nothing;
- it shapes answers that never mention it;
- it is deleted exactly by rebuilding.

That is known practice [recall].

## 5. Odds

**(a) A measured win over the strongest baseline: about 40%.** A win here means D without thinking beats B with thinking by ≥ 5 points on the hard set (interval above 0), at ≤ 1/20 of the answer tokens. The full PASS line is about 20%.
- For:
  - the split loses nothing on this world;
  - in simulation, wrong filing grew at about a tenth of the error rate;
  - the 4B reads full-name notes at 95% without thinking.
- Against:
  - B's writer put lure values into 51.9% of lure messages;
  - second-hand and direction errors;
  - paraphrase matching (the truth index scored 65.6% on who-questions);
  - the sealed wordings;
  - my designer's advantage.

**(b) A breakthrough: about 1%.**
- Every part is known, and DISCERN, KGGen and event-sourced agents are close.
- If it works, it becomes the sensible default for personal memory: good engineering, not a new science.
- Strong C4 is untouched.

**Runner-up (C5):** about 35% for a measured win over search; under 1% for a breakthrough.
