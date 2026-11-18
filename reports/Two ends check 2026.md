# The "two ends" check: can the phone model read without thinking, and write without the memory? (pre-registration)

Written and committed **before any model call** for this check. It is the first step of the plan all five members of the
breakthrough panel voted for (`reports/Breakthrough panel 2026.md`, section 6). The owner approved it on 16 November
2026 with a budget of about $3 of GPU time.

## The question

The panel's candidate design is **"describe, don't decide"**:
- the phone model describes each message without looking at its memory;
- code works out who everyone is and what is true now;
- the phone model answers from pages that code has already bound.

Before anything is built, two unknowns must be measured. The code in the middle is cheap and already simulated.

1. **Reading end.** With names, current values and links already worked out by code, can Qwen3.5-4B answer the hard
   questions **with thinking off**, in a few dozen tokens? If not, thinking is doing more than bookkeeping, and the
   phone-speed case dies for every design.
2. **Writing end.** Reading one message at a time, with only the previous message for context and no memory in view,
   can Qwen3.5-4B (thinking off) describe who is mentioned and attach each claim to the right mention?

## Data (development world only)

- Test 2's dev world (`brain/messy_world.py`, seed 101), first 1,500 messages, with the same 35 deletions as the dev run
  in `reports/Messy chat index test 2026.md` §16.7.
- **Reading end:** the 288 hard-set questions of that slice (the same questions as test 2's dev results).
- **Writing end:** 300 messages from that slice, drawn by a fixed rule with `random.Random(20261116)`:
  - all hard fact messages, up to 100 (hard = a fact message with a confusable name, a "through someone" or second-hand
    mention, or a pronoun pointing to the previous message);
  - 60 lure messages;
  - all definition messages up to 60 (introductions with a relation to me or a group, nicknames);
  - 40 relation messages;
  - the rest from other fact and event messages.
- **No test world, no sealed material, no training.**

## Arms

**Reading end.** The 4B, thinking off, greedy, at most 256 tokens. Answer prompt 4 (the current `ANSWER`). The pages
arm's messages (`pages()`), built four ways:

| Version | What the model sees |
|---|---|
| V0 | The messages as they are (the baseline; dev result 66.7% on the hard set) |
| V1 | V0 + **names**: each message carries the full names of everyone it mentions, with the words used ("Ana Varga, as 'Ana from choir'; Kofi Mensah, as 'he'") |
| V2 | V1 + a **"true now" header** for every person mentioned in the shown messages, built by code from all messages so far (not only the shown ones). It gives other names, how I know them, the current value of each attribute with the date it started, earlier values, things that stopped, and times with me. |
| V3 | V2 + **links**: every relation between people, in both directions ("Kofi Mensah's flatmate is Ana Varga") |

The header is built from the generator's truth. **So the reading end measures a ceiling:** what the phone model does
once the bookkeeping is perfect. It does not show that code can do the bookkeeping from real descriptions; that is the
writing end and step 2.

**Writing end.** The 4B, thinking off, greedy. Its output is forced into a fixed JSON form (structured outputs) with:
- **mentions:** the words used, the kind of mention, the name part, the person "through" whom someone is named, and
  the relation;
- **claims:** which mention the claim is about, a label, a value, and a status (true now / no longer / correction /
  plan or wish / tried or temporary / past / about me);
- **links** between mentions;
- **who told me**, for second-hand news.

Input: the message, its date and the previous message. Nothing from the memory.

## Lines

**Reading end, judged on V3** (panel §6):
- **PASS:** hard set ≥ 88%; each bookkeeping kind ≥ 75% (two-hop, retractions, roles, last meeting, nicknames);
  median answer ≤ 64 tokens.
- **STOP:** hard set < 80%.
- **Between:** reported as is.
- V1 and V2 are reported to show which part helps.

**Writing end**, scored by code against the generator's truth:

| Measure | Line |
|---|---|
| Claims attached to the right mention, easy fact messages | ≥ 98% |
| Claims attached to the right mention, hard fact messages | ≥ 93% |
| Relation direction (one-way relations: landlord, mentor, godparent) | ≥ 95% |
| Lures tagged (no "true now" claim carries the lure's value) | ≥ 90% |
| Definitions captured (relation to me, group, nickname: on the right mention with the right value) | ≥ 98% |
| Fact recall (each true fact's value appears in some claim) | reported; the neuroscientist's line was ≥ 90% |

- **STOP:** hard-message attachment < 85% **after a week of prompt work on dev.** This first run is the first
  measurement. Prompt changes are allowed on dev, and every version is logged here with its result.
- **How a claim is matched:** to a true fact by its value (the scorer's answer keys). Its mention is matched to a true
  mention by the words used. "Right mention" means the true mention of the fact's subject: for "Shirin's colleague is
  allergic to mustard", the mention "Shirin's colleague", not "Shirin".

**Not in this run:**
- the 27B writer arm, which needs a separate server;
- the owner's 300 labelled messages, which need the owner.

Both stay in the plan.

## Cost

- Reading end: 4 versions × 288 questions, thinking off. Writing end: 300 calls.
- Well under 1 L40S hour including start-up: about $1–3.

## Code

`brain/two_ends.py`. Results go to `results/two_ends/`.

---

## Results (16 November 2026)

All runs: Qwen3.5-4B on the project's Modal L40S, thinking off. Code and outputs: `brain/two_ends.py` and
`results/two_ends/` (`reading.jsonl`; `writing_v1.jsonl`, `writing_v2.jsonl`, `writing.jsonl` = v3; one summary per
version). GPU use was well under an hour, about $1–2 at the assumed rate. The exact bill is on the owner's Modal page.

### Reading end: BETWEEN (as registered), and a large cost result

| Version | Hard set | two-hop | retractions | roles | last meeting | nicknames | desc | who | Median answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| V0 plain pages | 66.3% | 23.5 | 26.7 | 33.3 | 33.3 | 52.9 | 73.1 | 87.5 | 11 |
| V1 + names | 72.9% | 29.4 | 60.0 | 50.0 | 33.3 | 70.6 | 84.6 | 87.5 | 12 |
| V2 + "true now" header | 90.3% | 58.8 | 73.3 | 100 | 55.6 | 91.2 | 96.2 | 90.6 | 17 |
| **V3 + links** | **91.7%** | **94.1** | **80.0** | **100** | **44.4** | **85.3** | 88.5 | 90.6 | **18** |

**V3 against the lines.** The hard set (≥ 88) and the median tokens (≤ 64) are met. One kind misses its ≥ 75 line: last
meeting, 44.4% on only 9 questions. Its failures give the date a note was written instead of the meeting date shown in
the header.

**Verdict: BETWEEN.**

**What it means.**
- Once code has done the bookkeeping, the phone model answers 91.7% of the hard questions with thinking off, in a
  median of 18 tokens.
- The same model needed about 4,200 thinking tokens for 95.5% on plain pages (test 2 dev, §16.7). That is about
  1/200 of the tokens for 4 points less.
- Names alone help little (+6.6 points). The "true now" header does most of the work (+17.4). Links add two-hop
  questions (58.8% to 94.1%).
- **This is a ceiling.** The header was built from the generator's truth. Whether code can build it from real
  descriptions is the writing end, then step 2.
- Some V3 failures are answers that give the right value with its history ("she previously had a labrador"). Test 2's
  frozen scorer marks those wrong in every arm alike. It was not changed.

### Writing end: below its lines after one day of prompt work; above the stop line

Three describer versions on the same 300 messages, each logged:
- **v1:** the registered prompt.
- **v2:** an introduction is one mention; "Name from group" is one mention; someone named through another person is
  not that person; relations go in links; trips are temporary; a second made-up example.
- **v3:** relations as "who is_the X of whom"; never describe the previous message; every message about a person gets
  a claim; things said in passing count.

**A corrected scorer is reported beside the registered one.** Reading run 1's failures showed four scorer false
negatives:
- a definition value with an article ("a party" vs "party");
- a mention given without its group ("Rosalind" for "Rosalind from work") falling just below the word-overlap
  threshold;
- a claim attached to no mention counted as a lure leak, though code would drop it;
- a relation written in its natural direction ("rents from") read backwards.

The corrected scorer also accepts the generator's own surface words for each group ("a colleague of mine" for work).
Both scorers give 100% on descriptions built from the truth, and both catch all 19 anchor misfilings ("Uma's colleague"
filed under Uma).

| Measure (line) | v1 registered / corrected | v2 registered / corrected | **v3 registered / corrected** |
|---|---|---|---|
| Right mention, easy (≥ 98) | 87.5 / 100 (n 24) | 87.0 / 91.3 (n 23) | **96.2 / 96.2 (n 26)** |
| Right mention, hard (≥ 93; stop < 85 after a week) | 79.4 / 91.8 (n 97) | 92.0 / 92.0 (n 88) | **88.8 / 89.8 (n 98)** |
| Relation direction, one-way (≥ 95) | 40.0 / 85.7 (n 17) | 45.5 / 46.7 | **78.6 / 64.7** |
| Lures tagged (≥ 90) | 78.3 / 83.3 | 93.3 / 95.0 | **91.7 / 93.3 ✓** |
| Definitions (≥ 98) | 55.4 / 78.5 | 52.3 / 72.3 | **70.8 / 92.3** |
| Fact recall (reported) | 93.8 | 86.0 | **96.1** |
| Tokens per message, out / in | 178 / 924 | 165 / 1,324 | 168 / 1,453 |

**What it means.**
- **The best version (v3) recalls 96% of facts.** It puts about 89–90% of claims in hard messages on the right
  mention, against B's writer at 78.2% of facts filed right (different measure, same direction). It tags 93% of lures
  and captures 92% of definitions (corrected).
- **Relation direction is the weak spot.** On 17 one-way relations, "Maya rents from Nuno" often came out as "Maya is
  Nuno's landlord".
- **Definitions are below their line.** Both matter more than their counts suggest: one misread definition or relation
  misfiles many later facts (the architect's fan-out finding).
- **The writing end has not reached its stop line** (hard attachment 88.8–89.8%, against the stop at < 85 after a
  week). The registered week of prompt work is not used up. The three versions moved the numbers by several points in
  both directions, so a 4B prompted this way is noisy to steer.
- **Small samples:** 26 easy facts, 17 relations and 9 last-meeting questions. Each could move by 10 points with a
  different sample.

### Not done in this run

- The 27B writer arm. It settles whether the root is the 4B's size or the procedure, and needs a separate server.
- The owner's 300 labelled messages.
- Step 2, the code resolver run on the describer's real output: the measure that decides wrong-person filing end to
  end.

### Next, in order (for the owner to decide)

1. **Step 2 on v3's descriptions** (CPU only): run them through a resolver, then measure wrong-person filing,
   "Kofi's flatmate" filing, forgetting and exact deletion against B's 6.27%, 11.1% and 62.2%.
2. **The describer's two weak spots,** with the rest of the prompt-work week, or a small LoRA trained only on
   generated lives:
   - relation direction;
   - definitions.
3. **The 27B arm, and the owner's 300 messages.**

---

## Step 2: the whole chain on the dev world (pre-registered before its model calls, 16 November 2026)

The owner approved step 1 of the "next" list.
- It is not CPU-only as first said: describing the other 1,200 messages and answering need about 10–20 GPU minutes
  (well under $1).
- It is a **development measurement on the dev world**. Prompts were developed on it, so it is not a judged run. The
  judged run comes later on a fresh seed, with the lines committed again.

**The chain** (`brain/describe_bind.py`):
1. **Describe.** Describer v3 (above) describes all 1,500 dev messages: one message, the previous one for context, no
   memory. The 300 already described are reused.
2. **Bind.** The engineer's resolver from the panel (`research_notes/Breakthrough panel 2026/engineer_scripts/resolver_sim.py`)
   reads the real descriptions instead of simulated ones. It re-solves who everyone is over the whole set, in date order,
   repeated until nothing changes. It never guesses: an unresolved mention becomes a placeholder person ("Kofi's
   flatmate, unknown"). It is ported to read describer JSON:
   - a mention's kind comes from the describer;
   - a pronoun's antecedent comes from the describer's "previous" field, or the only person in the previous message;
   - relation words are normalised ("tenant" becomes the reverse of "landlord").
3. **Index.** The bindings become index rows in test 2's format, so test 2's filing, forgetting and deletion measures
   run unchanged. Deleting a message means recomputing from the journal without it.
4. **Answer.**
   - Test 2's reader retrieves messages from this index, using the final clue prompt and retrieval code.
   - Pages are built as V3 above, from the resolver's view instead of the truth: names from the bindings; a header from
     the resolved claims (latest value per label with dates; "no longer" marked); links from the resolved relations.
   - The 4B answers with thinking off.

**Lines (dev; reference values are B's dev results, §16.7 of the test 2 report):**

| Measure | Line | B on dev |
|---|---|---|
| Wrong merges (fact rows under the wrong person) | ≤ 1.5% | 6.27% |
| "X's flatmate"-style facts filed right | ≥ 75% | 11.1% |
| Worst-person forgetting | ≥ 90% | 62.2% |
| Facts filed right | ≥ 85% | 78.2% |
| Hard set, the chain with thinking off | ≥ 77.3% (B with thinking + 5 points) | B thinking on 72.3%, thinking off 50.3% |
| Median answer | ≤ 64 tokens | B thinking on about 4,200 |

**Stop:** wrong merges > 3%, or the chain with thinking off < 70% on the hard set.

**Also reported:**
- hard-case merges (B 33.3%);
- filing by mention style;
- link precision;
- share of mentions left unresolved;
- deletion by recomputation: how many bindings change when the 35 deleted messages are removed;
- that a shuffled input order gives the same view.

### Step 2 results (16 November 2026): STOP on dev

**Run 1, the chain as registered** (`results/two_ends/chain_summary_v1.md`, `chain_answers_v1.jsonl`):

| Measure | Result | Line | B on dev |
|---|---|---|---|
| Wrong merges | 12.90% | ≤ 1.5% | 6.27% |
| "X's flatmate" filed right | 45.0% | ≥ 75% | 11.1% |
| Worst-person forgetting | 41.3% (average 58.8%) | ≥ 90% | 62.2% |
| Facts filed right | 77.2% | ≥ 85% | 78.2% |
| Hard set, thinking off | 52.1% | ≥ 77.3% | B thinking on 72.3% |
| Median answer | 22 tokens | ≤ 64 | about 4,200 |

**Verdict: STOP** (wrong merges above 3%, hard set below 70%).

**Self-check, run before the model calls** (`brain/describe_bind_selfcheck.py`). With descriptions built from the truth,
the same code gives:
- 0% wrong merges;
- "X's flatmate" filed right 100%;
- worst-person forgetting 96.6%;
- no unresolved mentions.

So the losses come from how the describer's real errors pass through the resolver.

**Diagnosis.** A few wrong *definitions* fan out, as the architect warned in the panel.
- **Split introductions.** "My dad Chloe Constantin called today" was often described as two unconnected mentions, so
  "my dad" was never defined.
- **Invented definitions.** "Forgot to reply to Ugo Mensah" was given "relation to me: dad", and "Sarai Quinn forwarded
  me a recipe" was given "relation to me: partner". These look like definitions carried over from the previous message.
  - In the votes for "sister", the real sister had 2 and two other people had 6 each.
- **Unrecovered "X's colleague" mentions.** The describer split "Uma from football's colleague" into pieces, and the
  facts went to Uma.

**Five resolver fixes, all development changes on dev.** Each was re-checked against the truth-built descriptions, which
still give 0% wrong merges.
1. A role and a full name in apposition ("my sister, Ugo Mensah") are one person.
2. A "my dad" mention cannot define "my dad".
3. Definitions (roles, nicknames, links) go by majority over the whole history, and a role changes hands only on "new".
4. A definition must be grounded: its value occurs in its own message. This applies to definitions only, because facts
   may be paraphrased.
5. A name beginning "my" is a role.

**After the fixes** (`results/two_ends/chain_summary.md`, `chain_answers.jsonl`):

| Measure | Result | Line | B on dev |
|---|---|---|---|
| Wrong merges | 9.42% | ≤ 1.5% | 6.27% |
| "X's flatmate" filed right | 45.0% | ≥ 75% | 11.1% |
| Worst-person forgetting | 49.2% (average 83.9%) | ≥ 90% | 62.2% |
| Facts filed right | 79.1% | ≥ 85% | 78.2% |
| Link precision | 86.1% | reported | 57.1% |
| Hard set, thinking off | 54.9% | ≥ 77.3% | B thinking on 72.3% |
| Chain minus B thinking on | −15.3 points [−22.7, −8.0] | interval above 0 | – |
| Median answer | 22 tokens | ≤ 64 | about 4,200 |

**Verdict: STOP.**

Filed right by how the person was mentioned:

| Mention | Filed right |
|---|---|
| full name | 98.6% |
| first name | 96.6% |
| first name with a group | 96.2% |
| previous-message pronoun | 97.1% |
| nickname | 91.7% |
| role | 81.6% |
| same-message pronoun | 66.7% |
| "X's flatmate" | 45.0% |

By construction:
- a shuffled input order gives the same view;
- removing the 35 deleted messages changes 3 bindings;
- no row carries a deleted value.

### What step 2 means

1. **With a 4B describer, "describe, don't decide" loses to B on dev.** It loses on wrong merges (9.4% against 6.3%)
   and on answers (54.9% thinking off against B's 72.3% thinking on). It wins on links (86% against 57%), on "X's
   flatmate" (45% against 11%) and on answer cost (22 tokens against about 4,200). By its pre-registered lines this
   route **stops** with the 4B describer.
2. **The panel's simulation was too kind** (the engineer predicted about 1–2% wrong merges at 5–10% describer error).
   The real describer's errors are a different kind: invented and split definitions, and pronouns attached to the wrong
   mention. A code resolver amplifies these, because one definition decides many later facts.
3. **The gap between the reading-end ceiling and the chain is all in the view.** Pages bound from the truth gave 91.7%
   with thinking off (step 1). Pages bound from this view gave 54.9%. The 4B can read; the view built from its own
   descriptions is too noisy.
4. **What would reopen the route** (the engineer's stop branch):
   - a better describer on the same resolver: the 27B at night, the cheapest test (one server, about 1,500 short calls);
   - or a 4B describer trained on generated lives.

   If the 27B gives wrong merges ≤ 1.5% through the same code, the design is right and the clerk is the problem. If
   not, the design itself fails on messy chat.
5. **Not tried, and possibly cheaper:** B's own index, which writes with memory in view, feeding the same kind of bound
   header. Step 1 showed the header does most of the reading work.

## Step 3: the owner confirms definitions (pre-registered before the run, 17 November 2026)

The owner approved the next step after step 2's STOP. It is a **free CPU simulation**: no model calls. It reuses the 4B's
cached descriptions of all 1,500 dev messages (`results/two_ends/describe_all.jsonl` and `writing.jsonl`) and the same
resolver.

**The question.** Step 2's diagnosis blamed a few wrong *definitions* that fan out to many later facts. Definitions are:
- "relation to me" (my dad);
- "known from" (from work);
- nicknames;
- links between people (Shirin rents from Tobias).

If the phone AI asks the owner to confirm new people and relations, the definitions become right. Does the rest of the
4B's output then file facts well enough? Or do its other errors (split mentions, facts attached to the wrong mention,
pronouns) also sink it?

**A correction to what was said before this step.** Definitions are not "about 4% of messages".
- 204 of the 1,500 dev messages that remain after deletion carry a definition (13.6%).
- 92 of the 204 are "known from" introductions.
- The dev world spans 176 days, so this is about 1.2 a day.
- No role changes hands inside the dev window (the two changes fall in October 2025 and February 2026), so "my new boss"
  is not tested here.

**Conditions** (`brain/confirm_defs.py`; the resolver in `brain/describe_bind.py` gains a path for confirmed
definitions, with the default path unchanged):
- **Step 2 as is** (reproduced): the 4B's own definitions.
- **A, every definition confirmed** (upper bound). The describer's definitional claims are dropped, including the
  apposition rule's role claims (the apposition *binding* stays). The truth's definitions in each message are added,
  dated by that message and naming people by full name, as the owner would. Everything else is the 4B's: mentions,
  facts, pronoun antecedents, who told what. One owner question per message that carries a definition.
- **B, a realistic asking policy.**
  1. The AI asks about each message where the 4B proposed a definition that survives the resolver's filters ("I noted:
     Sarai Quinn is your partner. Right?"). The owner corrects it to that message's true definitions, or none.
  2. When a mention cannot be resolved (a role, a nickname, "X's flatmate", an unknown first name), the AI asks "who is
     this?" at its first unresolved occurrence. The owner's answer binds that mention. For a role, nickname, group or
     "X's flatmate", it also defines the form for later messages. Pronouns are never asked about.

     Rounds repeat until no new question arises. A mention that is not a person costs a question and adds nothing. The
     owner is always right.

**Measures.** Test 2's filing, forgetting and deletion measures are unchanged, plus one added measure:
- **Wrong merges on everyday-fact rows: primary for this step.** This is the registered wrong-merge count with
  definition rows left out (relation to me, known from, nickname).
- **Why it is primary.** The registered count includes definition rows. Confirmed definitions are right by
  construction, so they would pad the denominator and flatter A and B.
- It is computed the same way for step 2's chain and for B's dev index, so all comparisons use one yardstick.

**Lines (each condition, dev):**

| Measure | Pass | Stop |
|---|---|---|
| Wrong merges on everyday-fact rows | ≤ 1.5% | > 3% |
| "X's flatmate" filed right | ≥ 75% | – |
| Worst-person forgetting | ≥ 90% | – |
| Facts filed right | ≥ 85% | – |
| Owner questions (B; A reported) | ≤ 2 per world-day (≤ 352 on dev) | – |

The question line is a judgement: an assistant that asks more than twice a day about who is who becomes a chore. It
depends on this world's pace of 8.5 messages a day, so questions per 100 messages are reported as well.

**Also reported:**
- the registered wrong-merge count;
- hard-case merges;
- filing by mention style;
- link precision;
- unresolved mentions;
- deletion by recomputation;
- for B, questions by type.

**What the results would mean (decided now):**
- **A fails the filing lines.** Confirming definitions is not enough: the 4B's other errors matter too. Next is the
  bigger describer (27B) diagnostic.
- **A passes, B fails.** Confirmation works only if the AI knows when to ask; the asking policy is the next problem.
- **A and B pass.** This is the first phone-size memory design that files well on dev. Next:
  - the answer run (GPU, under $1) against step 2's reading lines;
  - then the owner's 300 labelled messages and the sealed exam.

**Rules:**
- This is a development measurement on the world the resolver was tuned on, not a judged run.
- The resolver is not changed after seeing these results. Any change is logged as a dev change, with the numbers before
  it.
- **Regression checks before the run.**
  - The default path must reproduce step 2's numbers (wrong merges 9.42%).
  - Truth-built descriptions must still give 0% wrong merges.

### Step 3 results (17 November 2026): STOP on dev, for A and for B

`brain/confirm_defs.py` (`results/two_ends/confirm_summary.md`, `confirm_result.json`, `confirm.log`). It runs in about
10 seconds on a CPU.

**Regression checks passed:**
- the default path reproduces step 2 (registered wrong merges 9.42%);
- truth-built descriptions still give 0% wrong merges (worst forgetting 98.3%);
- B's dev index rebuilt from its cached replies reproduces 6.27%.

| Measure | Step 2 as is | A: every definition confirmed | B: asking policy | Line | B's index (test 2) |
|---|---|---|---|---|---|
| **Wrong merges, everyday facts (primary)** | 10.10% | **8.52%** | **8.77%** | ≤ 1.5% (stop > 3%) | 6.53% |
| Wrong merges, registered count | 9.42% | 7.36% | 7.73% | reported | 6.27% |
| "X's flatmate" filed right (of 20) | 45.0% | 65.0% | 65.0% | ≥ 75% | 11.1% |
| Worst-person forgetting | 49.2% | **92.1%** | **92.1%** | ≥ 90% | 62.2% |
| Average forgetting | 83.9% | 97.1% | 97.4% | reported | – |
| Facts filed right | 79.1% | 81.1% | 81.0% | ≥ 85% | 78.2% |
| Hard-case merges | 32.8% | 26.3% | 27.0% | reported | 33.3% |
| Link precision | 86.1% | 100% | 100% | reported | 57.1% |
| Mentions left unresolved | 9.2% | 4.3% | 3.2% | reported | – |
| Owner questions | – | 204 (1.16 a day) | 281 (1.60 a day; 19.2 per 100 messages) | B ≤ 2 a day | – |
| Verdict | STOP | **STOP** | **STOP** | | |

**What confirming definitions fixed:**
- **Forgetting.** "Forget Kofi" now finds 92% or more of the worst person's messages, against 49%. This passes its line.
- **Roles.** Facts about "my dad" are filed right 98.0% of the time, against 81.6%.
- **Links.** Link precision is 100%.
- **Cost.** The realistic asking policy (B) does as well as confirming everything (A), at 1.6 questions a day in this
  world.
  - 60 of B's 225 confirmations were the 4B's invented definitions, which the owner turns down.
  - 30 "who is this?" questions were about things that are not people.
  - B recovers 165 of the 204 true definitions.

**What it did not fix:** wrong merges. Everyday-fact wrong merges fell only from 10.1% to 8.5%, against a line of
1.5%. "X's flatmate" reached 65% (line 75%) and facts filed right 81% (line 85%). On the same yardstick the chain still
files worse than B's index (6.5%).

**Correction to step 2's diagnosis.**
- Step 2 said a few wrong definitions fan out. That is true for forgetting and roles. It is **not** what drives wrong
  merges: definitions account for 12 of step 2's 81 wrong everyday rows.
- This step had been proposed to the owner on that diagnosis.

**Post-hoc diagnosis** (`brain/confirm_defs_diag.py`, output in `results/two_ends/confirm_diag.txt`; written after
seeing the results; no measure or verdict changes). A's 69 wrong everyday rows (of 810) are about *which person in the
sentence* a fact belongs to, not who is who across messages:

| Cause | Rows |
|---|---|
| A pronoun that clashes with the usual gender of the person's first name ("Heard from Hana today. He trains for marathons.") | 20 |
| A "told me / says" claim filed on the teller ("Uma Popescu told me that Shirin Teixeira drives a Subaru" → Uma: "told me: that Shirin Teixeira drives a Subaru") | 17 |
| A pronoun with no name cue ("they", or a name not labelled), 10 of them in the frame "In her free time, Priya loves chess." | 15 |
| The fact put on the anchor of "X's relation" ("Shirin's colleague is allergic to mustard" → Shirin) | 7 |
| A pronoun that matches the name | 6 |
| Other | 4 |

**A defect in the practice world: pronouns are random.**
- The generator gives each person she, he or they at random, whatever the first name (`brain/messy_world.py`, line
  466; only family roles are fixed). No design reason is recorded.
- So where a name has a usual gender, the pronoun often clashes with it. The 4B uses that cue, as a person would, and
  looks for someone else.
- Same-message pronoun facts are filed right **40% of the time when the pronoun clashes** with the name (25 cases),
  against 82–91% when it matches or carries no cue (29 cases).
- The frame "In her free time, &lt;name&gt; loves …" puts the pronoun before the name. It fails even when the gender
  matches (Priya, Nadia, Kasia). That is a real weakness, but this generator uses the frame far more than chat does.
- Caveats: the gender labels are the analyst's own (`FEM`, `MASC` in the script), and the counts are small.

**Bounds, so the verdict does not rest on the defect** (arithmetic on the diagnosis, not a run):
- With "told me / says" claims dropped (a reporting verb is not a fact about the teller): 52 / 793 = **6.6%**.
- Also counting every clashing pronoun and every "In her free time" case as right: 22 / 793 = **2.8%**.

Even the most generous reading stays above the 1.5% line. The STOP stands with or without the defect. B's index carries
the same defect, so the comparison with B is fair as it stands.

### What step 3 means

1. **The problem has moved.**
   - Who is who *across* messages ("my dad", "Goose", "Kofi's flatmate") is now solved: code plus about 1–2 owner
     questions a day. Forgetting passes as a result.
   - What is left is reading *one sentence*: which person a fact belongs to (the pronoun, the teller, the anchor). The
     owner cannot be asked about that, because it comes up in nearly every message. The describer has to get it right
     alone.
   - This is a narrower problem than before. It needs no memory, so it can be trained from generated examples.
2. **The pre-registered next step is the bigger describer (27B) diagnostic.** Before it runs, its pre-registration
   should:
   - report wrong merges with and without clashing pronouns and the "In her free time" frame. A bigger model likely
     leans harder on name-gender cues, so the unsplit number would understate it on this world;
   - add the reporting-verb filter as a dev change, registered before the run.
3. **For test 2's FREEZE.**
   - The random pronouns affect every arm of test 2, including its sealed exam, which uses the same generator.
   - Before FREEZE, the owner should decide: should pronouns mostly follow the name's usual gender, with a realistic
     minority of exceptions reported separately?
   - Changing this changes the world hash (`b424cd579d89fec5`) and must be logged as a dev change.
4. **Not tried, by design.** The describer's format could be changed so that each fact names its person directly, as
   B's rows do, instead of pointing at a pronoun mention. That would be prompt tuning on the practice world, which this
   project said it would stop doing.

## Step 4: the bigger describer (pre-registered before its model calls, 17 November 2026)

The owner approved the pre-registered branch after step 3's STOP: the bigger describer on the same resolver.

**The question.** Is reading one sentence (which person a fact belongs to) a capacity problem that a bigger model
solves? Or does the describe-then-bind format itself fail at it?

**Setup** (`brain/bigger_describer.py`; server `brain/modal_qwen27b.py`):
- **Model.** Qwen3.5-27B-FP8: the 4B's family, about 7 times larger. vLLM 0.33.0 on one H100.
- **Settings, unchanged from the 4B:**
  - describer v3, word for word;
  - thinking off;
  - temperature 0, at most 700 tokens, structured JSON output, seed = message id;
  - one message plus the previous one, no memory.
- **Scope.** The 27B describes all 1,500 dev messages itself; none of the 4B's descriptions are reused.
- **Binding.** The same resolver, and the three step-3 conditions:
  - its own definitions;
  - A, every definition confirmed;
  - B, the asking policy.
- **Caveat.** vLLM is not batch-invariant, so a rerun can differ slightly.

**Dev change, registered now and applied to both models:** the reporting-verb filter.
- A fact claim whose label begins with told, tells, says, said, heard, hears, reported, reports, mentioned, mentions or
  according is dropped ("told me: that Shirin drives a Subaru" is not a fact about the teller).
- On the 4B, measured before this registration, it moves A from 8.52% to 6.57%.

**Split measure, registered now for interpretation only.**
- What it is: the primary wrong-merge count, leaving out the 66 of 1,465 messages touched by the practice world's
  pronoun quirk.
- How those messages are chosen, from the truth alone:
  - a person with an everyday fact is referred to by a pronoun that clashes with the usual gender of their first name
    (the labels in `brain/confirm_defs_diag.py`);
  - or the message uses the frame "In her free time, &lt;name&gt; loves …".

**The 4B under exactly this code** (reference, computed before any 27B call; `results/two_ends/bigger_reference_4b.json`):

| Measure | 4B own | 4B A | 4B B |
|---|---|---|---|
| Wrong merges, everyday facts | 8.17% | 6.57% | 6.83% |
| Same, without the pronoun quirk | 5.11% | 3.23% | 3.51% |
| "X's flatmate" filed right | 45.0% | 65.0% | 65.0% |
| Worst-person forgetting | 49.2% | 92.1% | 92.1% |
| Facts filed right | 78.8% | 80.8% | 80.7% |

**Lines (the 27B, condition A, filter on):** the same as step 3.

| Measure | Pass | Stop |
|---|---|---|
| Wrong merges, everyday facts | ≤ 1.5% | > 3% |
| "X's flatmate" filed right | ≥ 75% | – |
| Worst-person forgetting | ≥ 90% | – |
| Facts filed right | ≥ 85% | – |

**What the results would mean (decided now):**
- **The 27B passes.** The design works with a good enough clerk, and the 4B is the problem. Next: train the 4B on the
  27B's descriptions of freshly generated worlds (teacher and student), judged on a fresh seed.
- **The 27B fails, but its split measure is ≤ 1.5%.** The 27B reads sentences well except where the practice world's
  pronouns mislead it. Next: fix the world's pronouns (the owner's decision, before test 2's FREEZE), then judge.
- **The 27B's split measure is above 1.5%.** Even a model 7 times larger cannot attach facts within one sentence in
  this format. The format becomes the suspect: claims that point at mentions, and pronouns as mentions. Next is a
  format change, not a bigger model.

**Also reported:**
- the 27B's own and B conditions;
- its numbers without the filter;
- wrong rows by mention style;
- the writing-end scores on the 300 sample with the corrected scorer (the 4B: easy 96.2, hard 89.8, relation direction
  64.7, lures 93.3, definitions 92.3, recall 96.1);
- valid JSON, and tokens per message.

**Not in this step:** answers. The reader stays the 4B. An answer run happens only if filing passes, and needs its own
approval.

**Cost and access:**
- One H100, about 20–40 minutes with start-up: about $1.50–3.
- The endpoint is protected by a vLLM API key, held as the cloud environment's network secret.
- The URL is never written into the repository.

### Step 4 results (17 November 2026): STOP as registered, narrowly; post-hoc, most of what remains was the resolver's pronoun rule

The run used `brain/bigger_describer.py`; the outputs are `results/two_ends/bigger_summary.md`, `bigger_result.json`,
`bigger.log` and `describe_all_27b.jsonl`.
- The 27B described all 1,500 dev messages in 3.9 minutes on one H100. All 1,500 replies were valid JSON, at 145
  tokens out and 1,452 in per message.
- The 4B reference reproduced exactly under the same code.
- **Infrastructure.** The first deploy crashed after loading. vLLM's default of 1,024 sequences in flight needs more
  of Qwen3.5's linear-attention state blocks than fit beside the 27B (801). Modal then held the GPU until its
  30-minute start-up limit, about 40 H100 minutes lost.
- **The fix**, in `brain/modal_qwen27b.py`:
  - at most 64 sequences in flight;
  - a cache for the FP8 kernel build;
  - a watchdog that ends the container when vLLM dies.

  The second start took 7 minutes.

| Measure | 4B own | 4B A | 4B B | 27B own | 27B A | 27B B | Line |
|---|---|---|---|---|---|---|---|
| **Wrong merges, everyday facts (primary)** | 8.17% | 6.57% | 6.83% | 4.02% | **3.44%** | 3.72% | ≤ 1.5% (stop > 3%) |
| Same, without the pronoun quirk | 5.11% | 3.23% | 3.51% | 2.92% | **1.98%** | 2.29% | interpretation |
| "X's flatmate" filed right | 45.0% | 65.0% | 65.0% | 81.0% | 100% | 100% | ≥ 75% |
| Worst-person forgetting | 49.2% | 92.1% | 92.1% | 42.9% | 100% | 100% | ≥ 90% |
| Facts filed right | 78.8% | 80.8% | 80.7% | 88.0% | 89.1% | 88.9% | ≥ 85% |
| Link precision | 86.1% | 100% | 100% | 90.1% | 100% | 100% | reported |
| Mentions left unresolved | 9.2% | 4.3% | 3.2% | 22.6% | 1.4% | 0.6% | reported |
| Owner questions | – | 204 | 281 | – | 204 | 332 (1.89 a day) | B ≤ 2 a day |

**Writing end on the 300 sample** (corrected scorer), 4B → 27B:

| Measure | 4B | 27B | Line |
|---|---|---|---|
| Claims on the right mention, easy | 96.2% | **100%** | 98 |
| Claims on the right mention, hard | 89.8% | **97.1%** | 93 |
| Relation direction | 64.7% | **100%** | 95 |
| Lures tagged | 93.3% | 95.0% | 90 |
| Definitions | 92.3% | **84.6%** | 98 |
| Recall | 96.1% | 98.4% | – |

**Verdict, as registered: STOP.**
- The 27B with every definition confirmed files 3.44% of everyday facts under the wrong person, above the 3% stop line.
- Without the pronoun quirk it files 1.98%, above 1.5%. The registered reading of that case: "even a model 7 times
  larger cannot attach facts within one sentence in this format; the format becomes the suspect". With 13 wrong rows of
  655, the interval around 1.98% is wide (roughly 1–3.4%).

**What the 27B fixed** (post-hoc diagnosis, `brain/confirm_defs_diag.py --descs … --filter`, output in
`results/two_ends/bigger_diag.txt`):
- Of the 4B's 69 wrong rows in step 3, the 27B makes none of these kinds:
  - "told me / says" junk on the teller;
  - facts put on the anchor of "X's relation";
  - other errors.
- One "In her free time" case is left.
- **All 25 of its remaining wrong rows are pronouns:** 15 clash with the name, 6 have no name cue, 4 match the name.

**Where those 25 come from, read row by row:**
- **14 are the resolver's fault, not the describer's.** The 27B's "previous" field named the right person, but the
  code dropped it:
  - **The code looks for that person only in the previous message.** In "Heard from Lucia Zulu today. He says his
    street in Dresden is lovely", "He" was given as Lucia Zulu. The code did not find her in the previous message and
    fell back to someone there.
  - **The code never lets a pronoun refer to the teller.** The 27B marks "Fadil Morozov says…" and "Beatriz texted
    me…" as told by Fadil or Beatriz.
  - **Names matched as substrings.** "he" counted as part of "Hedda".
- **11 are the describer's own:**
  - 8 are clashing pronouns: "Got a message from Petra. He adopted a ferret" → Ivan from the previous message;
  - 1 is the "In their free time" frame;
  - 2 are genuine slips.

**Post-hoc: the rule fixed** (`brain/bigger_posthoc.py`, `results/two_ends/bigger_posthoc.md`; a switch,
`describe_bind.ANTE_V2`, off for every registered number).
- **The fix.** The describer's "previous" may name someone in the same message, the teller included, and names match as
  whole words.
- **Checks.** With the switch off, the 27B summary reproduces exactly. With it on, truth-built descriptions still give
  0% wrong merges.

| Condition A | 4B | 4B fixed | 27B | 27B fixed | Line |
|---|---|---|---|---|---|
| Wrong merges, everyday facts | 6.57% | 5.56% | 3.44% | **1.52%** (11 of 726) | ≤ 1.5% |
| Same, without the pronoun quirk | 3.23% | 2.66% | 1.98% | **0.61%** | – |
| "X's flatmate" / worst forgetting / facts filed | 65 / 92.1 / 80.8 | 65 / 92.1 / 81.7 | 100 / 100 / 89.1 | **100 / 100 / 90.5** | 75 / 90 / 85 |

With the asking policy (B) instead of confirming everything, the 27B fixed gives 1.79% (0.92% without the quirk) at
1.89 questions a day.

**This is not a pass.**
- The rule was changed after seeing these rows, on the world the resolver was tuned on.
- It can only be judged on a fresh world with lines registered first.

### What step 4 means

1. **The design works with a good enough describer, on dev, after a post-hoc fix to its code:**
   - the 27B describes;
   - the owner confirms new people (about 1–2 questions a day);
   - code binds and deletes by recomputation.

   With these three, 1.5% wrong-person facts (0.6% without the world's pronoun quirk), 100% forgetting and 100% "X's
   flatmate". This is the first configuration in this project to reach all the filing lines, if only in a post-hoc
   reading.
2. **The 4B is the weak link.** Even with the fix it files 5.6% wrong. The 27B's writing-end scores show where it is
   better: the hard attachment cases (97% against 90%), relation direction (100% against 65%), and no junk claims.
3. **The 27B is worse at proposing definitions on its own** (84.6% against 92.3%; its own-definition forgetting is
   42.9%). Owner confirmation is part of the design, not an extra.
4. **Next, in order:**
   - a judged confirmation on a fresh-seed world: the 27B, the fixed rule, conditions A and B, lines registered first.
     It costs about $1–2 now that the server starts in 7 minutes;
   - only if that passes, the route to the phone: teach the 4B from the 27B's descriptions of generated worlds
     (teacher and student);
   - still open for test 2's FREEZE: the pronoun quirk.

## Step 5: the judged run on a fresh world (pre-registered before its model calls, 17 November 2026)

The owner approved it. Step 4's post-hoc configuration is judged on a world the resolver was never tuned on.

**World.** `messy_world.World("dev", seed=20261117)`, first 1,500 messages. The seed is new, used nowhere in this
project:
- dev is 101;
- test 2's are 20261115 and 7, and the scored seed comes from the beacon;
- 20261116 picks step 1's writing sample.

| Property | Value |
|---|---|
| Messages after 25 deletions | 1,475 |
| Days | 185 |
| Messages carrying a definition | 203 |
| Messages touched by the pronoun quirk | 63 |
| People | 260 |

- **What it tests.** The same generator and sentence templates as dev, so it tests new people, names, facts and
  sequences, **not new phrasing**.
- **What it leaves out.** Dev slots and relations only. Nothing sealed is read.

**Frozen configuration.**
- **Describer.** Qwen3.5-27B-FP8, describer v3, thinking off, memory-blind; the same settings as step 4.
- **Resolver.** As in step 4's post-hoc reading:
  - the antecedent rule fixed (`describe_bind.ANTE_V2`);
  - the reporting-verb filter on.
- **Labels.** The usual-gender labels now cover all 233 first names the generator can pick (`brain/fresh_check.py`),
  fixed before the run. They are a superset of the dev labels; 18 names are left as unisex.
- **No changes after seeing the results.**

**Checks run before any model call** (`python brain/fresh_check.py --selfcheck`): truth-built descriptions on this world
give 0.14% wrong merges, 100% "X's flatmate", worst forgetting 98.2% and 2 unresolved mentions of 1,544, with either
rule. So the resolver itself makes a few errors off the world it was tuned on, against 0% on dev. That floor is small
beside the 1.5% line.

**Lines (condition A, every definition confirmed):**

| Measure | Pass | Stop |
|---|---|---|
| Wrong merges, everyday facts | ≤ 1.5% | > 3% |
| "X's flatmate" filed right | ≥ 75% | – |
| Worst-person forgetting | ≥ 90% | – |
| Facts filed right | ≥ 85% | – |

**Also reported:**
- the split without the quirk (interpretation);
- condition B, with the same lines plus ≤ 2 questions a day;
- the 27B's own definitions;
- wrong rows by mention style;
- link precision, unresolved mentions, valid JSON and tokens.

**What the results would mean (decided now):**
- **PASS.** "Describe, confirm, bind" generalises past the world it was tuned on, within this generator's phrasing.
  Next: teach the 4B from the 27B's descriptions (teacher and student), and an answer run with the 4B reader.
- **BETWEEN** (1.5–3%). Dev's 1.52% was partly fitted to dev. Look at the errors before teaching the 4B anything.
- **STOP** (> 3%). Step 4's post-hoc reading did not hold. The configuration was fitted to dev.

**Not tested:** new phrasing, real chat (the owner's 300 labelled messages), and answers.

**Cost.** One H100 already running: about 5 minutes of describing, about $0.50.

### Step 5 results (17 November 2026): BETWEEN, as registered

`brain/fresh_check.py` (`results/two_ends/fresh_summary.md`, `fresh_result.json`, `fresh.log`,
`describe_fresh_27b.jsonl`). The server was woken in 18 seconds. It described all 1,500 fresh messages in 4.0 minutes,
all valid JSON.

| Measure | own definitions | A: every definition confirmed | B: asking policy | Line | Dev, post-hoc (A) |
|---|---|---|---|---|---|
| **Wrong merges, everyday facts (primary)** | 2.56% | **1.70%** (12 of 706) | 1.70% | ≤ 1.5% (stop > 3%) | 1.52% |
| Same, without the pronoun quirk | 1.72% | 0.93% | 0.93% | interpretation | 0.61% |
| Wrong merges, registered count | 2.79% | 1.43% | 1.47% | reported | – |
| "X's flatmate" filed right | 86.7% | 100% | 100% | ≥ 75% | 100% |
| Worst-person forgetting | 36.9% | 100% | 100% | ≥ 90% | 100% |
| Facts filed right | 91.3% | 91.8% | 91.8% | ≥ 85% | 90.5% |
| Link precision | 88.1% | 100% | 100% | reported | – |
| Owner questions | – | 203 (1.10 a day) | 306 (1.65 a day) | B ≤ 2 a day | – |

**Verdict: BETWEEN.**
- Every line passes except the primary one, which misses by about 1 row (1.70% against 1.5%).
- The registered reading: "dev's 1.52% was partly fitted to dev; look at the errors before teaching the 4B anything".
- On the numbers, the fresh result is within noise of dev's: 12 of 706 against 11 of 726, each with an interval of
  roughly 1–3%. So the fresh world shows **no sign of fitting beyond chance**. Both worlds sit just above the line.

**The errors** (`results/two_ends/fresh_errors.txt`; diagnosis after the results). All 12 are pronouns whose person is
named in the **same** message, where the 27B's "previous" points to someone in the previous message instead:
- **8 come from a clash between the pronoun and the name's usual gender.** "Got a message from Dmitri Okonkwo. In her
  free time, She loves fishing." → Maya from the previous message. Six fall in the quirk set; the other two are Noor,
  labelled unisex, and "My boss … her".
- **4 are other errors:**
  - singular "they" ("Got a message from Flora Lindgren. They have a pet corn snake.");
  - a nickname not taken as the person ("Got a message from Kit. Correction: Her hobby is …", twice);
  - "Svenja says her street …" → Edda from the previous message.
- **None are** teller junk, anchors, definitions or links.

### What step 5 means

1. **The describe, confirm and bind design holds on a world it was not tuned on**, within this generator's phrasing:
   - wrong-person facts 1.7% (0.9% without the quirk);
   - 100% forgetting and 100% "X's flatmate";
   - about 1.1–1.7 owner questions a day.

   It misses the pre-registered 1.5% by about one row, so the verdict is BETWEEN, not PASS.
2. **What is left is one mechanism:** a pronoun whose person is in the same message, resolved to someone earlier. It is
   mostly triggered when the pronoun does not match the gender the name suggests.
   - That case is far more common in this world than in real chat, because pronouns here are random.
   - But it is also the case that **misfiles real people whose pronouns differ from what their name suggests**. An
     accountable memory should not misfile them more than anyone else.
3. **Still untested:** new phrasing, real chat, the 4B as describer after teaching, and answers.

## Step 6: the real-chat check on REALTALK (pre-registered before its model calls, 18 November 2026)

The owner approved it, in place of labelling their own messages.

**Data.**
- **Source.** REALTALK (Lee et al. 2025, arXiv 2502.13270; github.com/danny911kr/REALTALK at commit `48b3ab8`,
  Apache-2.0): 21 days of real messaging-app chats between consenting participants.
- **Selection.** Chats 1–4, 1,761 messages, between four participants who each appear in two of them: Emi, Elise, Kevin
  and Paola.
- **Kept out of the repository.** The data stay in a local clone outside the repository. Nothing that quotes a message
  is committed: not the descriptions, not the judging files. Only claim ids, person ids, the summary and the log are.

**Setting: one memory per person.** Each participant is an owner, whose journal is their own messages across their two
chats, in time order (Elise 464, Emi 439, Kevin 453, Paola 405).
- **Describing.** Each message is described memory-blind from its writer's point of view, with the chat's previous
  message (either speaker) for context.
- **Binding.** Each journal is bound on its own.
- **The closest real-chat stand-in for "the owner tells their AI about their life".** The difference: these messages
  are written to a person, so "you" is that person.

**The configuration** (`brain/realchat.py`):
- **Describer.** Qwen3.5-27B-FP8: describer v3, thinking off, memory-blind, the same settings as steps 4–5.
- **One registered change, for chat.** The first line names the writer and the listener. One added rule: "you", "your",
  "u" and "ur" mean the chat partner, listed as a mention of kind "first name" with the partner's name. "We" and "us" are
  not mentions. The rest of v3 is unchanged, including "facts about me, the writer, are not claims".
- **Resolver.** Frozen at step 5: the antecedent rule fixed, the reporting-verb filter on, the 27B's own definitions (no
  confirmation simulated: REALTALK has no answer key for definitions).

**Judging** (there is no answer key, so the analyst, Claude, is the judge; a different model family from the
describer). Two passes, written before any description was seen:
1. **Pass 1, claims.** Each claim is shown with its message and the chat's four previous messages, in random order. The
   mention it was attached to and where it was filed are **not** shown. The judge writes who it is about:
   - a person id;
   - `OWNER` (a fact about the writer);
   - or `NONE`: no fact about a person. That covers a question, a guess or a greeting. A plan, a past event or a
     feeling stated or clearly implied is a fact.
2. **Pass 2, people.** Each of the resolver's people that holds a claim is shown with its mention words and example
   messages. The judge writes which person id it is.

A claim is **filed right** when the two passes agree. `OWNER` claims are counted as wrong-person (a fact about the
writer filed under someone else).

**Sampling rule.** If there are more than 600 claims, every claim not attached to the partner's mention is judged, plus
200 of the partner's drawn at random (seed 6), and the overall rate is weighted.

**Check on the judge.** The owner checks 30 judged claims drawn at random (claim, context, the judge's label). If fewer
than 27 of 30 agree, no verdict is drawn and the step is reported as "judge not reliable enough".

**Lines (pooled over the four journals):**

| Measure | Pass | Stop |
|---|---|---|
| Wrong-person rate, over judged claims that are facts about a person | ≤ 1.5% | > 3% |

**Also reported:**
- wrong-person rates for claims on the partner's mention, on third parties and on pronouns;
- the share of claims that are no fact (`NONE`);
- the share of facts about the writer turned into claims;
- claims per message, and valid JSON.

**What the results would mean (decided now):**
- **PASS.** Filing survives real phrasing in this sample. Next: pronouns confirmed by the owner, then teach the 4B.
- **BETWEEN.** Look at which real-chat phenomena break it before any 4B work.
- **STOP.** Real phrasing breaks filing as configured, so the synthetic results do not transfer. Diagnose before any 4B
  work.

**Not measured:**
- recall: facts the describer misses, because there is no independent fact list;
- definitions and confirmation;
- answers;
- owners other than four 18–25-year-old US English speakers chatting with study partners.

**Cost.** About 1,761 calls on the 27B server, under $1.

### Step 6 results (18 November 2026): STOP (the owner agreed with 30 of 30 checked labels)

`brain/realchat.py`. The committed files are `results/two_ends/realchat/`: `result.json`, `scored_claims.json`,
`labels_claims.json`, `labels_entities.json`, `spotcheck_ids.json`, `run.log`, plus the first 16 characters of the
descriptions' SHA-256. The descriptions themselves quote messages, so they stay outside the repository.

- **Describing.** The 27B described all 1,761 messages in 4.0 minutes. 10 replies were not valid JSON. The result was
  1,113 claims (0.63 per message), filed under 152 people.
- **Judged.** 548 claims, under the registered rule: all 348 not on the partner's mention, plus 200 of the partner's.
- **Two departures from the written protocol, both for reading effort, neither revealing any filing:**
  - claims were judged in conversation order, not shuffled, so each context was read once;
  - 2 previous messages were shown by default, with more read where needed.
- **Two labelling conventions, fixed before scoring:**
  - **Sets.** A set label counts as a match on overlap. It is used for identical claims about several people named
    together, and for entities whose mentions the text cannot tell apart.
  - **Writer facts.** A fact about the writer (`OWNER`) is always wrong-person, as registered.

| Measure | Result |
|---|---|
| **Wrong-person rate, judged facts about a person (weighted)** | **5.47%** (line ≤ 1.5%, stop > 3%) |
| On the partner's mention / on third parties / on pronouns | 3.5% / 8.5% (n = 176) / 12.5% (n = 48) |
| Claims that are no fact about anyone (weighted) | 40.4% |
| Facts about the writer turned into claims | 2.3% |
| Without the 12 "your X" errors (below) | 3.73% |

**Verdict: STOP.** The owner checked 30 random labels (`spotcheck_ids.json`, drawn with seed 30) and agreed with all 30,
above the registered 27.
- 30 of 30 still allows a judge error rate of up to about 10% (95% upper bound).
- The labels are therefore checked, not proven.

**What breaks on real chat** (25 wrong of 339 judged person claims, unweighted):

| Cause | Rows |
|---|---|
| **"Your X" read as "my X"**: "Your mother looks like she is a great cook" filed under the writer's own mum; "your brother", "your best friend", "your ex" likewise | 12 |
| A fact about the writer made into a claim about someone else ("Yes I have been to Italy" → the partner) | 7 |
| Filed under a non-person (a "we" mention, a pronoun for an office, Ancient Egypt) | 4 |
| Other | 2 |

**About "your X".**
- **Where it comes from.** It is a gap in this step's registered chat adaptation, which defined "you" but not "your
  &lt;relation&gt;".
- **Why it matters less for the target.** It hardly arises when the owner talks to their own AI.
- **Why it does not decide the verdict.** Without these 12 rows the rate is still 3.73%, above the stop line.

**What real chat adds that the practice world never had:**
- **Most of the 27B's claims are not facts.** 40% are questions ("What is your favorite thing about Las Vegas?" became
  "favorite thing: Las Vegas"), greetings, reactions, and facts about pets, companies or countries.
- **People use aliases.** Emi is "Kate" to Elise and "Bella" to Paola. The resolver splits these into separate people,
  which this measure does not count as wrong-person.

### What step 6 means

1. **The synthetic results do not transfer to real phrasing as configured.** On real chat the same describer and
   resolver file 5.5% of facts about people under the wrong person. On generated worlds the figure was 1.5–1.7%. Third
   parties (8.5%) and pronouns (12.5%) are worst.
2. **Real chat has failure types the practice world never had:**
   - questions, greetings and reactions written down as facts (40% of claims);
   - "your X" read as "my X";
   - the writer's own facts handed to the other person;
   - one person split under aliases.

   None of these comes from memory or from who is who across messages. All of them come from reading one real message
   from the right point of view.
3. **What this test is not:**
   - two young strangers chatting with each other, not an owner telling their own AI about their life;
   - four people, with few third parties;
   - judged by Claude and spot-checked by the owner.
4. **Next, by the registered reading** ("diagnose before any 4B work"):
   - no teaching of the 4B yet;
   - the diagnosis above names the problems: a describer that writes down non-facts, and that loses the writer's
     point of view.
   - **REALTALK chats 5–10** (7,183 messages, six other people) have not been read by anyone in this project. A chat
     describer could be developed on chats 1–4 and judged once on chats 5–10, with lines registered first.
5. **A question for the owner.**
   - The 1.5% line was the analyst's judgement.
   - With receipts (every fact linked to its message) and owner confirmation, a wrong-person fact is visible and
     correctable.
   - What misfiling rate is acceptable for the product is the owner's call, and it changes which next step is worth
     doing.

## Step 7: a chat describer, judged once on unread chats (pre-registered before its model calls, 18 November 2026)

The owner approved option 1 after step 6. The 1.5% line stays: the owner gave no other bar.

**Development on chats 1–4** (step 6's data, already read; `brain/realchat.py --stage step7dev`). Describer v4 adds
rules for chat aimed at step 6's diagnosis. It was compared with v3 using step 6's labels, by message, with no new
labelling.

| Chats 1–4 | v3 (step 6) | v4a | v4b (frozen) |
|---|---|---|---|
| Claims per message | 0.63 | 0.26 | 0.27 |
| Claims on step 6's no-fact messages (147) | 272 | 43 | **33** |
| Claims on messages ending in "?" (654) | 485 | 124 | 117 |
| Facts judged right in step 6 on messages where it writes no claim at all | – | 102 of 314 | 98 of 314 (46 of them questions) |

The two versions:
- **v4a.** A question, greeting, thanks, compliment, reaction, wish or advice is not a claim. "Your X" is the partner's
  X (a "through someone" mention). The writer and "we"/"us" are never mentions, and nothing about the writer is a
  claim. Only people are mentions.
- **v4b** (one dev revision). A message that *only* asks, greets, thanks or reacts has no claim, but a reaction that
  names a fact keeps it ("Congrats on the new job!"). In "me and my mom", the mom is a mention.

On step 6's 18 error messages, v4b gives "your mother" its own person, "the mother of your mother" (the partner's
mother), and writes no claim for "Yes I have been to Italy", "my name is Bella" or "we both found activities".

v4b is **frozen**: the prompt, the resolver (step 5's), the filter and the code.

**The exam: REALTALK chats 5–10.** 7,183 messages and six people (Nicolas, Nebraas, Vanessa, Akib, Muhhamed, Fahim
Khan), each in two chats. Their journals hold 520–1,730 messages. Until this commit only counts and names were read from
them, never a message. As in step 6, nothing that quotes them is committed.

**Measures:**
1. **Wrong-person rate**, as in step 6: two blind judging passes on a registered sample of claims.
   - **Sample.** Up to 300 claims not on the partner's mention and 150 on it, each drawn at random (seed 7), with
     each stratum weighted.
   - **Conventions.** Step 6's: sets match on overlap, and `OWNER` is always wrong.
2. **Facts captured (new).** 150 messages are drawn at random (seed 70) and labelled by the judge, **before the
   descriptions are seen**, for the facts about people other than the writer that they state or clearly imply (step
   6's rules).
   - After the run, each fact is matched to the system's claim that expresses it, or `MISS`.
   - It counts as captured when that claim is filed under the right person.
   - Without this measure, a describer that writes almost nothing could pass the first one.

**Lines:**

| Measure | Pass | Stop |
|---|---|---|
| Wrong-person rate | ≤ 1.5% | > 3% |
| Facts captured and filed right | ≥ 70% | < 50% |

PASS needs both. STOP comes with either. Anything else is BETWEEN.
- The 70% and 50% are the analyst's judgement: a memory that misses more than half of what it is told is not usable.
- The practice runs suggest at least 17% of non-question facts are lost, so 70% is a real test, not a formality.

**Also reported:**
- rates by group (partner, third party, pronoun);
- the share of claims that are no fact;
- claims per message;
- people split under aliases.

**Check on the judge.** As in step 6: the owner checks 30 random claim labels, and fewer than 27 agreements means no
verdict.

**What the results would mean (decided now):**
- **PASS.** A big model with chat rules reads real chat well enough to file facts. Next: teach the 4B from its
  descriptions, judged on held-out chat.
- **BETWEEN.** The rules help but do not reach the lines. Look at which errors remain before more work.
- **STOP.** Better instructions do not make a 27B read real chat well enough in the describe-then-bind form. The
  describe step itself needs rethinking: for example reading several messages together, or keeping the conversation
  rather than one message.

**Cost.** About 7,183 calls on the running 27B server: about 10 minutes, about $1.

### Step 7 results (judged 18 November 2026)

**Verdict: STOP, on both lines,** pending the owner's check of 30 labels (below). The judge's own slips are fixed
post-hoc in a logged second count; that count is also STOP on both lines.

| Chats 5–10, describer v4b | As registered | Judge's slips fixed (post-hoc, logged) | Pass | Stop |
|---|---|---|---|---|
| Wrong-person rate (weighted, 322 judged facts) | **9.12%** | 7.44% | ≤ 1.5% | > 3% |
| Facts captured and filed right (19 facts) | **26.3%** (5) | 36.8% (7) | ≥ 70% | < 50% |

**Also reported** (as registered):

| Measure | Value |
|---|---|
| Claims, and claims per message | 978, 0.14 (step 6, v3 on chats 1–4: 1,113 and 0.63) |
| Judged | 450: 322 facts about a person, 128 no fact |
| Share of claims that are no fact (weighted) | 26.5% (step 6: 40.4%) |
| A fact about the writer filed under someone | 3.9% of judged facts |
| Wrong-person, partner's mention | 19.8% (step 6: 3.5%) |
| Wrong-person, third parties | 6.7% (4.4% with the judge's slips fixed; step 6: 8.5%) |
| Wrong-person, pronouns | 6.6% (4.6% fixed; step 6: 12.5%) |
| People split under several records | 32 of 132 people holding judged claims; 130 of the 242 records are one-message pronoun placeholders |

**The judge's slips** (`results/two_ends/realchat_v4/reconcile_log.json`, 11 changes). A second count fixes only these
two kinds, both found while reading the disagreements:
- the two passes naming one person in two ways;
- a pass misreading a record's own example messages.

Semantic calls stay as judged, even arguable ones. Examples:
- "owed money: 200" filed under the man owed it;
- "community with: Akib" filed under the neighbour.

The fixes:
- Vanessa's ex was named "creepy man" (Nebraas's word) in two pass-1 labels.
- Pass 1 missed that "bro" was Naruto.
- Pass 2 read a record of the boyfriend's family as the boyfriend.
- Three coworker records and one recall fact used different names. One of these merges was planned before scoring and
  then forgotten.

The fixes can only lower the error count. This is why the registered count is the verdict and the fixed one is shown
beside it.

**Two leniencies in the judging**, both in the system's favour:
- **Nebraas's partner** is "girlfriend" to Vanessa and "boyfriend" to Nicolas, in the same stories (the bakery
  coworker, the Weeknd concert), and was judged as one person.
- **Nebraas's two roommates** (the best friend and the one "a bit like Julio") were not told apart, so a mix-up between
  them is not counted.

**What the 27 remaining wrong-person errors are** (unweighted, after the fixes):

| Kind | Errors | Example (paraphrased) |
|---|---|---|
| A "you" that is not the partner, filed under the partner | 13 | 11 general "you" (living in the city "you can walk to the pharmacy"; "the bagels come out wrinkled when you boil them"); 1 "you" inside a quoted text message; 1 "pal" meaning someone else |
| The writer's own feeling or relation filed under the other person | 5 | "never heard of him" filed on Logan Paul as "known to me: never heard"; "I don't care about him" filed on Gordon Ramsay as "interest: none" |
| The writer's own fact filed under the partner | 2 | "neither do I have a nuclear engineering background" became the partner's background |
| "Your family" filed under the partner, not as the partner's family | 2 | |
| One third party's fact filed under another | 3 | Harry Potter's "meets his godfather" filed on Sirius Black; Dr Phil's baldness filed on Gypsy Blanchard |
| Arguable relation wording | 2 | the two semantic calls above |

The first kind is self-inflicted: v4's rule says "you" means the partner. Even if all 13 were right, the rate would be
4.30%, still over the stop line.

**Why 12 of the 19 recall facts were missed** (the recall labels were written before any description was seen):

| Reason | Facts missed |
|---|---|
| The describer wrote nothing for the message. In most of these, the person is named only in an earlier message ("like Shutter Island, Inception and Catch Me If You Can", the actor named two lines before) | 8 (7 messages) |
| v4b's "a message that only asks has no claim" rule ("how did the cat adoption go?" says the partner is adopting a cat), or a plan that is not written as a claim | 2 |
| Only a definition was written, not the fact | 1 |
| The describer wrote the fact, but the binder dropped it as "past" | 1 |

**A pipeline leak found in this step.** The frozen binder (step 5's) keeps a claim only if its status is "true now",
"correction" or "no longer". On the exam chats it silently drops 510 of the describer's 1,507 non-definition claims
(34%):
- 249 marked "past";
- 233 "plan or wish";
- 27 "tried or temporary";
- 1 other.

The judging rules count a past event or a plan as a fact. So in real chat this filter throws away a third of what the
describer writes. It came from the synthetic world and I did not check it during development. Here it costs at most
one recall fact (8 of 19, 42%, still under the stop line). On a describer that writes more it would matter more.

**Check on the judge.** 30 claims are drawn from the 450 labels by step 6's rule (seed 30, over the claim ids;
`spotcheck_ids.json`) and given to the owner: 24 facts about a person and 6 no fact. If the owner agrees with fewer than
27, the verdict is withdrawn and the step is reported as "judge not reliable enough".

**What it means.** The registered meaning of STOP applies: better instructions alone do not make the 27B read real chat
well enough in the describe-then-bind form. The describe step itself needs rethinking. The errors say why:
- **Most of what is wrong needs the conversation, not the message:**
  - who "she" is (130 records are pronouns the resolver could not attach, so Nicolas's crush is spread over 16
    records);
  - whether "you" is the partner or anyone;
  - facts whose person was named a few lines earlier (8 of the 12 recall misses).
- **Two parts are self-inflicted and cheap to fix,** but fixing them does not reach the lines:
  - the "you means the partner" rule (4.30% without those errors);
  - the status filter (42% recall at most).
- **v4b moved the errors rather than removing them.** It cut junk from 40% to 26%, and claims on the partner's mention from two thirds of
  the judged weight to a fifth. The partner claims left are mostly the hard ones: general "you".
- **The wrong-person measure does not see splits,** and splits are as large a problem for a memory as wrong filing.
  "Everything about my crush" lives in 16 places.

**Next, for the owner to choose** (REALTALK has no unread chats left, so any new judged run needs another human chat
set):
1. **Describe with the conversation** (the analyst's recommendation). The describer sees the last few messages from
   both people and writes who each "she", "he" or "you" is, in words the resolver can bind ("my crush", "Vanessa's
   ex", or "nobody, general you"). The binder keeps past events and plans. Develop on REALTALK, then judge once on a
   new, unread chat set.
2. **Fix the two cheap problems only** and re-measure on chats 5–10 as development. It costs little, but the numbers
   above say it cannot pass.
3. **Change where memory comes from.** Use the owner's own notes and messages, where the writer is the owner and "I"
   is unambiguous, before other people's chat.

## Step 8: let the models work out who is who (pre-registered before the exam is read, 18 November 2026)

The owner asked why everything is hand-written, and approved this plan instead of more rules: the describer sees the
conversation, the chat rules go, and the small model teaches itself. Code: `brain/convchat.py`, `brain/whoquiz.py`,
`brain/who_train.py`, `brain/modal_who4b.py`.

**The systems** (all bind with step 5's resolver; one memory per owner):
- **A. Conversation describer v5 (27B).** It sees the 12 messages before the latest one, from both people, and
  writes for each person mentioned "who" they are: a name, "my crush", "Ana's boss", or "nobody" for a general "you".
  - Step 7's chat rules are removed; the base describer (step 6's) is kept for the output format.
  - The code binds the "who" words. A group ("Julio and Nebraas") becomes one mention per member, minus the writer.
  - Claims marked past, plan or tried are kept, fixing step 7's 34% leak.
- **B. A, with the self-taught 4B deciding which person each he/she/they is.**
  - The 27B still decides whether a pronoun is one particular person at all (not "nobody", the writer or a group).
  - The 4B (Qwen3.5-4B with LoRA) answers the rest; "unclear" leaves the mention unattached.
  - A and B hold exactly the same claims and differ only in where pronoun claims are filed.
  - The 4B learned only from a quiz the chat made by itself: a person named outright ("my sister", "Julio") is
    hidden behind a pronoun, and the answer is the hidden words.
  - Quiz sources: REALTALK chats 1–8 and EmpatheticDialogues' train split, 1,170 items. Held out for checking:
    REALTALK 9–10 and the valid split, 155 items.
  - No person labels were used. The only hand-written part is the quiz maker (which words count as a person, which
    pronoun replaces them).
- **B0. As B, with the 4B before training** (the control for the training).
- **0. Step 7's frozen system** (v4b, one message at a time) as a baseline.

**Development on REALTALK chats 5–10** (already read; judged by the analyst unblinded, 120 random claims):

| | Step 7's system | A |
|---|---|---|
| Wrong-person | 8.4% (27/322, step 7's blind count) | 7.9% (6/76) |
| Recall facts captured and filed right (step 7's 19) | 5–7 | 10 |
| One-message pronoun placeholders | 130 of 242 records | 0 |
| Claims that are no fact about a person | 26% | 37% |

- One dev fix: as a two-word "name", a group had taken over every later "Julio". Groups are now split.
- The A errors are:
  - the writer's own feeling or plan filed under its object (3; "I love Gordon Ramsay");
  - a pronoun given the wrong person (2);
  - a merge (1).
- A new problem: one person described in different words far apart ("the stalker", "the weird guy"). The describer
  sees 12 messages, not the journal.
- **Expectation, written before the exam:** on REALTALK-like chat, A would still be STOP on wrong-person.

**The exam: EmpatheticDialogues' test split** (Rashkin et al. 2019, CC BY-NC 4.0; tarball sha256 `56f234d7…`).
- **What it is:** 2,547 conversations, about 4 turns each, 10,973 messages. Real people (paid crowdworkers) type to
  each other about something that happened to them, often involving family, partners, friends and coworkers.
- **Unread:** nothing but counts has been read from it.
- **Limits, stated now:**
  - The conversations are short and between strangers, so this tests reading a conversation, not keeping track of
    people over weeks.
  - A dataset this well known may be in the 27B's or the 4B's pre-training text. It holds no who-is-who labels, so
    the task itself has not been seen.
- **Journals:** each speaker's messages in one conversation are a journal ("Speaker A" opens; "Speaker B" replies).

**Measures**, step 7's, judged the same way:
1. **Wrong-person rate.**
   - Systems A, B and B0 share one sample: 450 claims, up to 150 on the partner's mention and 300 others, each
     drawn at random (seed 80), each stratum weighted.
   - Pass 1 labels the claims once, for all three.
   - Pass 2 labels the union of their records: a record is named by its words, so one name gets one label.
   - System 0: 150 random claims (seed 82), with their own two passes.
2. **Facts captured and filed right.**
   - 150 messages are drawn at random (seed 81) and labelled for the facts about people other than the writer,
     **before any exam description is seen**. Their SHA-256 is committed before the run.
   - Each fact is matched to a claim, or MISS. A, B and B0 share the match, since their claims are the same.
3. **Also reported:**
   - the rates by group (partner, third party, pronoun);
   - the share of claims that are no fact;
   - claims per message;
   - **people kept in one record:** of people with two or more judged claims filed right, the share whose claims all
     sit in one record.

**Lines** (for A and for B; PASS needs both, STOP comes with either):

| Measure | Pass | Stop |
|---|---|---|
| Wrong-person rate | ≤ 1.5% | > 3% |
| Facts captured and filed right | ≥ 70% | < 50% |

**Check on the judge.** As before: the owner checks 30 labels drawn by step 6's rule (seed 30), and fewer than 27
agreements means no verdict.

**What the results would mean (decided now):**
- **A passes.** With the conversation in view, the 27B reads short real chat well enough to file facts. What remains
  is naming people the same way across weeks (seen in dev), which this exam does not test. Next: give the describer
  the journal's list of names, and test on long chat.
- **A is STOP or BETWEEN.** Even with the conversation, misfiling remains. The kinds of error decide the next step:
  - writer's facts attached to others: a structured "about me" field, or the owner confirming;
  - wrong pronoun choices: more training;
  - merges: the binder.
- **B against A and B0** (wrong-person on the pronoun group, and overall):
  - **B below B0:** the quiz taught the 4B something that carries over to real pronouns.
  - **B at or below A:** a phone-sized model can take over the who-step from the 27B. Next: teach the 4B the rest of
    the describer the same way.
  - **B not below B0:** the self-made quiz did not carry over. It is too easy (the answer is always nearby) or too
    noisy.

**Cost.**
- Training: about 15–25 minutes on one H100 (the owner runs `modal run brain/modal_who4b.py`).
- Exam: two describers on about 11,000 messages (roughly 30 minutes of the 27B), plus the 4B's answers on one L40S.
- In all, about $4–6.
