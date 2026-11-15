# Founder, round 1

Labels: **[checked]** verified today, source given; **[recall]** my memory; **[speculation]** my idea.

## The direct question: chase the shift, or ship?

**Don't make a 2017-class shift the team's goal.**
- **The owner's picture of 2017 is off.** The Transformer paper's stated motive was engineering: recurrence "precludes parallelization within training examples". The authors deleted it and won on a public translation test (WMT 2014, 28.4 BLEU) at a fraction of the training cost [checked: arXiv 1706.03762, via search]. Attention was already three years old [recall: Bahdanau 2014]. One deletion, forced by a real job's bottleneck, proven on a shared test.
- **This team's base rate:** four from-scratch rebuilds (SPARC on language, the counting model, additive features, the ledger) each ended at their stop line [checked: brief, README].

**"Ship the memory that already works" is also wrong: nothing at phone size works yet on messy chat.** The 4B's own index:
- scored 70.1% on the hard set;
- filed 6.3% of facts under the wrong person;
- answered with a median of about 4,200 thinking tokens, roughly 3.5 minutes on a phone.

[checked: `results/messy_index_dev_dev1500/summary.md`]. Within a week a user would notice the wrong-person memories and the slowness.

**What does work** [checked: same file]:
- the raw journal with receipts;
- exact message deletion (0 rows left, 0 leaks);
- a reader that never invented a never-stated fact (0%, against search's 35–38%). This is the most valuable trust property the project owns.

**My recommendation:**
- Spend about 4 days in 5 making that memory good enough for one real person, the owner, on his laptop, within weeks. His shower moment (C5) cannot be tested honestly on a generated world: use is the experiment.
- Spend 1 day in 5 on bets with stop lines.
- Run the sealed test 2 as registered first. It is cheap and it is the honest baseline.

## 1. The assumptions list

**What is wrong**
- **A1** says the 4B's fast-weight state "integrates by construction". Test 1 measured this already: the whole notebook in the 4B's context (that state plus 8 attention layers) scored 20.4% with thinking on and 24.7% with it off. Name lookups were 92%, links and descriptions under 5% [checked: README].
- **A3** cites guessing as evidence, which says nothing about activity between questions. A3 is unmeasured.
- **A6 overclaims.** The truth-built index trails perfect pages by 9.4 points, and on who-questions it scores 65.6% against 96.9% [checked: summary.md]. Matching loose descriptions stays a model-shaped job.
- **A7 is weaker than listed.** Retractions went from 47% to 100% through prompt changes [checked: report 16.4–16.5].
- **The blocking table** puts C2 behind A4–A6, yet deleting a message already works perfectly. What fails is "forget a person" (worst person 62% removed), a filing problem.

**What is missing**
- **M1. "The phone model is used as shipped; only prompts and parsers change."** Every dev-phase fix was a prompt or parser patch [checked: 16.2], although the generator gives unlimited labelled examples.
- **M2. "Memory is written in order, each batch filed against the index built so far."** That is recurrence (state carried from step to step), with its illnesses [checked: 16.3–16.4]:
  - identical code gave 5.5% and 6.3% wrong merges;
  - one parsing change shifts every later batch;
  - saved replies cannot be re-parsed.
- **M3. "Generated worlds stand in for a life."** No real person's messages have ever been tested.
- **M4. "The user is outside the loop."** The AI can ask "which Mira?", and every correction is a free label.
- **M5. "The phone gives the night shift hours."** iOS background processing gets "several minutes", when the system chooses, usually on the charger [checked: developer guides and Apple forum threads; no official figure]. Every night-shift design needs a one-day check on a real phone.

**The single removal that unblocks the most: M2.** I rebuilt B's dev index from the saved writer replies (no model call; script `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/founder/merge_breakdown.py`) and sorted its 49 wrong fact rows (of 781) [checked]:

| Error | Rows |
|---|---|
| "Uma's colleague plays the mandolin" filed under Uma (the anchor) | 15 |
| "She plays the trumpet", where "she" was named in an earlier message | 9 |
| A named person's fact sent to someone not mentioned at all (wrong person number, or two people merged) | 18 |
| Second-hand news filed under the teller | 7 |

So 42 of 49 (86%) are binding or bookkeeping across messages, which the running index makes the 4B do in its head. If code did that binding perfectly, wrong merges would be 7 of 781 = 0.9%, under the 1% line. That is a ceiling, not a forecast.

## 2. My candidate: read each message alone, bind in code, rebuild from the journal

**In plain words.**
- **Reading.** The 4B reads each message, with the few before it, and writes what it says in a fixed form, never choosing a person number. Examples: "a person called Mira, from work"; "the flatmate of Kofi"; "she = the person named in the previous message"; "source: Soren; about: Chloe".
- **Binding.** Plain code, run over all the readings at once, decides who is who: full names, "my sister" through relation statements valid at that date, nicknames, and first names unique at that date. "Kofi's flatmate" stays an open placeholder until some message, earlier or later, says who that is (the moderator's deferred reference). Nothing unknown is guessed.
- **Rebuilding.** The memory is a pure function of the journal. Deleting a message means recomputing, and the result is identical to never having seen it. Next year's better model re-reads the whole past overnight, and the memory improves backwards in time.

**Why it could be a shift.** It is the 2017 move at memory scale: remove the recurrence. Recurrence makes today's writer non-reproducible, error-compounding, untrainable from labels (each step depends on its own earlier mistakes) and unable to delete exactly. Without it:
- every message is an independent training example;
- the work runs in parallel;
- deletion and upgrades become recomputation.

It is the project's founding thesis, order-free memory, applied where the data says it bites. A user would notice it stops confusing people.

**Closest prior art [checked by search]:**
- query-time entity resolution, which resolves references together when a question needs them (Bhattacharya & Getoor, JAIR 2007);
- rebuilding derived state from an append-only log when code changes (event sourcing; Kappa architecture, Kreps 2014);
- Hindsight (arXiv 2512.12818), which resolves entities by name similarity, co-occurrence and closeness in time;
- Zep/Graphiti (arXiv 2501.13956), which resolves entities as each message arrives: the recurrent design.

I found no system that reads with no index in view, binds globally and order-free, and deletes by recomputation (one extended search: absence of evidence, not proof). Verdict: **a re-arrangement with a 2017 shape, not a 2017 size.**

**Runner-up: train the reader on generated lives** (removes M1). The direction is known:
- Memory-R1 trains its memory operations by reinforcement learning from 152 examples [checked: arXiv 2508.19828];
- Mem-α trains Qwen3-4B to build memory [checked: arXiv 2509.25911].

Do it only if my experiment shows the remaining errors are in reading single messages.

## 3. The smallest decisive experiment

**Built**
1. **A reader prompt** for Qwen3.5-4B, thinking off, with structured output: the server forces a fixed JSON shape, so "NEW: He" or "name: axolotl" cannot occur. Input: one message plus the 5 before it.
2. **A binder** in Python, about 500 lines. It writes rows in B's format, so the existing reader, scorer, filing and forgetting checks run unchanged.
3. **Rebuild on delete.**
4. **Fast path.** Notes are handed to the reader with each bound full name attached ("[about Ana Varga] Ana from choir got a ferret"), thinking off. Thinking-off failed because the 4B does not connect identities across messages (perfect pages: 66.7%); this tests whether code binding cures the 3.5-minute answers.

**Data**
- Develop on the first 1,500 dev messages.
- Judge on the first 1,500 messages of a fresh, never-viewed test-split seed with normal wordings (it has the held-out kinds), with B run on the same slice. Commit the lines before the run.
- The sealed world comes only after the registered verdict, as a labelled extra arm.

**PASS**, on the fresh slice, all of:
- wrong merges ≤ 1.5% (B on dev: 6.27%);
- "X's flatmate" filed right ≥ 75% (11.1%);
- worst-person forgetting ≥ 90% (62.2%);
- hard set ≥ B + 8 points, bootstrap interval above 0, and ≥ 0.9 × oracle;
- 100 of 100 deletions identical to never-seen;
- two runs within 1 point;
- within the registered writing caps.

Fast path, its own line: thinking off ≥ 80% of the hard set.

**STOP**, any of:
- wrong merges > 3%;
- hard set no better than B;
- a fall of more than 10 points from dev to the fresh seed. That means the binder learned the generator's habits, which real life breaks.

Then the errors are in single-message reading: go to the runner-up.

**Same week, reported without a line:** the owner marks who each fact is about in 300 of his own messages (about 2 hours). Both writers run on his laptop: a 4-bit Qwen3.5-4B is about 2.8 GB and runs in recent llama.cpp [checked: Hugging Face model cards]. Whether it fits his 6 GB card is unmeasured. This would be the project's first real-life number.

**Cost:**
- About 5–7 L40S hours, roughly $15 [recall: about $2 an hour]. Reading is minutes; answering with thinking is about an hour per arm.
- 8–10 days of work.

## 4. The moderator's living belief system

**Good engineering, as the moderator says.** Most parts exist [checked by search]:

| Part | Where it exists |
|---|---|
| Valid-time facts that are retired, not overwritten | Zep/Graphiti |
| Entity resolution and confidence-scored beliefs | Hindsight |
| Night-time memory work | Letta's sleep-time compute: about 5× less answer-time compute at equal accuracy on its own benchmarks (arXiv 2504.13171) |
| Trained memory operations | Memory-R1, Mem-α |
| Late reference resolution | Bhattacharya & Getoor |
| Truth maintenance | Doyle 1979 [recall] |

**The proactive overnight feature already shipped:** ChatGPT Pulse, September 2025. Search summaries say OpenAI's help page now marks it as being retired into scheduled tasks [checked via search only; the page refused my fetch, so treat as unconfirmed]. If true, it is a warning: unasked-for surfacing is hard to make people want, even with unlimited compute.

**New in it:** exact retraction by justification, and the measured whole at phone size. Built in full, it is months for a small team. Build the slice the data points at (my candidate), measure, then add the next.

**To be more than engineering, one of these would have to be true:**
1. The typed memory language is learned, not designed: a trained 4B's own format carries over to unseen relation kinds and to real chat.
2. C4 is measured: a new fact changes answers inside the model's thinking, without the engine joining facts first.
3. C5 is measured on a real life, with a false-alarm rate a person tolerates.

The current design contains none of these, and side-steps (2) on purpose.

## 5. Odds

**(a) A measured win over the strongest baselines** (B and the search agent, under the pass line): **about 35%** [speculation].
- **For:** 42 of 49 errors are fixable by code; the ceiling is 0.9%.
- **Against:**
  - the reader must still get "Kofi's flatmate" and back-references right alone;
  - only 78.2% of facts end up filed under the right person [checked], partly through facts never written down;
  - binding rules tuned on one generator may not survive a new seed.
- Wrong merges alone under 1.5%: about 50%. Fast path at 80% or more: about 45%.

**(b) Something the field would call a breakthrough: about 1%.** Every part exists. At best it is a good systems paper: order-free memory writing for small models, with exact deletion. The chance of a 2017-class memory shift from this team in the next few years, by any route, is under 1%. That is no reason to stop. It is a reason to aim at a memory one real person trusts, and to watch for the bottleneck real use reveals: that is where 2017 came from.
