# Breakthrough panel: is there a 2017-class shift for memory? (November 2026)

**The owner's question.** Before 2017 today's AI looked impossible, and one new idea changed that. Is there a
comparable new idea for the "brain": memory and learning that work like a person's? Can it connect things learned
weeks apart, the owner's "shower moment"? He asked the moderator to:
- list the assumptions today's AI memory rests on;
- find the one that blocks;
- propose replacements with tests that prove or kill them;
- have experts judge honestly whether anything here can be a breakthrough.

**How the panel ran.**
- **Five roles,** the four from earlier panels plus a new one:
  - a machine-learning architect;
  - a curious founder;
  - a senior neuroscientist;
  - an AI research scientist;
  - **a first-principles engineer:** told to accept no existing tool or framework as given, to reason down to bits and
    energy if needed, and to keep working until he had a concrete design and a test.
- **Material:** each read the moderator's assumptions document (`research_notes/Breakthrough panel 2026/assumptions_v1.md`)
  and a brief with the whole project record, and could read the repository, search the literature and run CPU scripts.
- **Round 1:** each wrote alone.
- **Round 2:** each read the others and the moderator's checks (`moderator_checks.md`), then corrected themselves and
  voted.
- **What the moderator checked:** every claim about this repository was re-run and confirmed, and every simulation the
  members ran reproduced exactly. Nine outside papers were opened.
- **Caveat:** all five are the same underlying model with different instructions. Their agreement counts as about one
  opinion, as three of them said themselves.
- **Transcripts:** `research_notes/Breakthrough panel 2026/` (`r1_*.md`, `r2_*.md`, plus the members' scripts).

---

## 1. The answer, in plain words

**No. Nothing the panel found is a 2017-class breakthrough, and all five said so.**
- Every candidate re-arranges known parts. Several close relatives were published in the last few months (DISCERN,
  User as Code, DeepDive's lineage, MemReader-4B).
- Each member put the chance that their own idea becomes a field-level breakthrough at about 1–2%.
- The 2017-class problem is still untouched by everything on the table: new experience used inside the model's
  thinking, as if always known.

**But the panel located the blockage precisely, and agreed on one cheap test.** On this project's own data:

| Finding | Evidence |
|---|---|
| **The phone model can connect facts.** With the right messages in front of it and time to think, it answers 95.5% of the hard questions. With full names written out, it answers 95.1% even without thinking. | tests 1 and 2 |
| **It fails at bookkeeping across messages.** Who is "she", who is "Kofi's flatmate", which value is current. It does this badly in one pass (two-step questions: 93% with 10 notes, 40% with 100, 5% with 5,000). | test 1 answers |
| **The index writer fails the same way.** It files "Uma's colleague plays the mandolin" under Uma. 42 of its 49 wrong rows are links across messages. | test 2 dev |
| **Thinking fixes it, but costs about 4,200 tokens per answer:** roughly 3.5 minutes on a phone. | test 2 dev |

**The design all five voted to test first: "describe, don't decide".**
- The phone model reads each message on its own, without looking at the memory. It writes down who is mentioned
  exactly as written ("Kofi's flatmate", "my old boss", "she") and what is said about them.
- Plain code then works out who everyone is, from all the descriptions together. It redoes this whenever a message is
  added or deleted. Deletion is therefore exact by construction, and every belief has receipts.
- The phone model then answers from pages where code has already filled in names and current values, ideally without
  thinking.

**The honest best case for the next month** (the members put it at about 1 in 4 to 1 in 3):
- wrong-person memories fall from about 6 in 100 to about 1 in 100;
- answers take seconds instead of minutes;
- "forget Kofi" removes all of Kofi, provably.

That would be the project's first memory a person could trust daily. It would not be new science.

**On the shower moment.** The experience is real, but the evidence says the shower mostly gets the credit:
- a break helps modestly;
- what sleep does is disputed;
- connections usually form when the second thing arrives and reminds you of the first, because a question was still
  open in your mind;
- false "aha" moments are common.

The engineer found that combining two facts is not the missing piece: the phone model does it 93% of the time once the
right ten notes are in front of it. The missing piece is **bringing those notes forward, weeks apart, without being
asked, at a false-alarm rate a person tolerates**. Nobody on the panel found a test of exactly that, and the panel
designed one. That is where a real advance could be.

---

## 2. The votes

**Round 1 candidates:**

| Member | Candidate |
|---|---|
| Architect | **Write locally, read globally:** each message stored alone (exact deletion, receipts); all connecting done at question time by a reader trained on simulated lives. Runner-up: an "undoable sleep" that teaches a small per-person cache. |
| Founder | **Remove the recurrence:** read each message alone, bind people in code over the whole diary, rebuild the memory from the diary. |
| Neuroscientist | **A memory that re-reads itself, keyed by what things are for** (needs, offers, risks, open concerns). Runner-up: a learned relational structure (Tolman–Eichenbaum Machine). |
| Researcher | **Bind first, read fast:** code binds identities and current values at night; the phone model answers without thinking. Runner-up: a shower test on science papers. |
| Engineer | **Describe, don't decide:** a memory-blind describer emits mentions as written in a small description language; code resolves everything over the whole set; deletion by recomputation, checked by equality. Runner-up: noticing as a nightly join over resolved people. |

**Round 2: all five voted for the same merged experiment** ("describe, don't decide", with the reading-side check as
its first gate):

| Member | First gate passes | Full pass line | Headline win (thinking off beats today's index with thinking) | Breakthrough |
|---|---|---|---|---|
| Architect | 65% | 20% | 35% | ~1–2% |
| Founder | 70% (gate 0a) | 15% | 35% | ~1% |
| Neuroscientist | 60% | 15% (core line 25%) | – | ~1% |
| Researcher | 60% | 15% | 35% | ~1% |
| Engineer | 65% (reading end), 40% (writing end) | – | ~30% | ~1% |

The project's forecasting record is 0 for 5. These are opinions.

---

## 3. What the panel agreed on

1. **The diagnosis rests on checked numbers:**
   - the founder's breakdown: of B's 86 wrong hard-set answers, about half (43) trace to who-is-who at writing;
   - the engineer's two-step data;
   - the contrast between 95.1% (full names, no thinking) and 66.7% (messy chat, perfect pages, no thinking).
2. **Code should do the bookkeeping that has an exact answer.** The model should do what needs language understanding.
   A full re-solve of the 6,292-message dev world takes 0.04 s.
3. **Personal facts must never go into the model's weights.** Anything that blends facts can be deleted only by
   recomputing it, so accountability rules weights out as the store. Weights may hold skills learned from generated
   data, never the owner's life.
4. **Use the owner as the ground truth.** He should mark who is meant in 300 of his own messages (about two hours).
   That gives the first number from a real life and is the cheapest guard against the designers' advantage.
5. **No from-scratch architecture until a cheap check shows the existing model is the limit.** Four from-scratch
   rebuilds in this project ended at their stop lines.

## 4. Where the panel disagreed, or could not settle it

**What is the root?** Three explanations are still open:
- (a) the model decides identity while writing, with the memory in view, and the decision is final;
- (b) one pass of the 4B skips the middle step, in writing and reading alike (the engineer's two-step data);
- (c) the 4B is simply too weak, and a 27B running today's procedure would be fine.

The merged experiment includes the control that separates them: the same messages written by the 27B, memory-blind and
with today's procedure.

**Half the loss is on the reading side.**
- 36 of the 86 wrong answers had the facts filed right, but the reader didn't find them, or misread them with every
  needed message in hand.
- 16% of facts were never written at all, three times the wrong-person rate.

"Describe, don't decide" does not touch these.

**The test world is kind.**
- It always defines a relation or nickname before using it, so a code resolver partly just runs the generator
  backwards.
- The researcher counted real chat (REALTALK, 8,944 messages). People rarely use a name before saying who it is, but
  they say "my girlfriend" for weeks without ever naming her.
- So "deferred reference" is not the selling point. The selling points are reproducibility, exact deletion, batched
  writing, and re-reading the past with a better model.
- The hardest real cases ("her new friend" vs "the friend she's talking to") need judgement. They may end up as
  "unknown", or need the AI to ask the owner.

**"Never guess" has a price.**
- It trades false memories for incomplete forgetting: "forget Elif" must remove every message that *could* be about
  her. In the engineer's stress world the worst-served person stayed at 76%, which is unsolved.
- A misread definition ("my brother, Gaia Okafor") misfiles many facts at once (the architect: 13 on average, up to 53).
  Code errors are rarer but bunched.

**Test 2's sealed run.**
- The neuroscientist, architect and engineer: run it as registered, as the honest baseline.
- The founder: the sealed world can be used honestly only once, and B's verdict is already predictable. Add the
  describer as a registered arm before the FREEZE, with a fixed date. If its gates have not passed by then, freeze
  without it.
- The moderator's recommendation is the founder's version. It keeps the pre-registration intact and does not waste the
  sealed world. The owner decides.

## 5. Corrections to the record

**The architect's citation.**
- He cited arXiv 2607.27539 with no version, and paraphrased it more strongly than the text.
- The moderator's round-1 check read only the current version, a different paper, and called the citation false. That
  verdict was too strong.
- Version 2 and the companion paper (arXiv 2609.06872) do contain the finding: in a 48B delta-rule model, a record's
  contribution "changes by 12–49% with the suffix", and replay matched the never-stored state in all 80 contexts.
- Lesson: cite versions.

**The moderator's assumptions document had five errors** that the panel found:
- A1 claimed the 4B's fast-weight state "integrates by construction". It is small, order-dependent and can't be deleted
  from, and the whole-notebook run scored 20%.
- A5 read "Kofi's flatmate, 11%" as a decision made too early. It was a failed lookup, because the test world always
  defines first.
- A6 over-read the truth index as evidence for a "belief state".
- The blocking table put "real deletion" (C2) behind the wrong assumptions. Deleting a message already works perfectly;
  forgetting a *person* is what fails.
- A4–A7 were one assumption counted four times.

**The members corrected themselves:**
- the founder: run-to-run noise comes from the server, not from recurrence;
- the neuroscientist: the late-connection gate allowed 20% false fits while the pass line allowed one false alarm a
  week;
- the researcher: dropped the science-paper runner-up;
- the engineer: his round-1 simulation could not produce the commonest real error (filing under the anchor). With it,
  wrong filing at a 5%/30% describer error rate is 2.65%, not about 1%. His win odds went from 40% to 30%.

## 6. The plan the panel converged on

**Step 1. The "two ends" check.** Dev world only, about 1.5 L40S hours, 4 days.
- **(a) Reading end** (the researcher's stage 0, the architect's quick check, the founder's gate 0a).
  - The 4B, thinking off, answers the 288 dev hard questions from pages where code has filled in names, a "true now"
    header and pre-joined links. There are three versions to see which part helps.
  - **Pass:** ≥ 88% overall, each bookkeeping kind ≥ 75%, median answer ≤ 50–64 tokens.
  - **Stop:** < 80%. That means thinking is doing more than bookkeeping, and the phone-speed case dies for every
    design.
- **(b) Writing end** (the engineer's gate 1, the founder's reader).
  - The 4B, thinking off and memory-blind, describes 300 dev messages (at least 100 hard, 60 lures) in a fixed format.
  - **Pass:** claims on the right mention ≥ 98% on easy and ≥ 93% on hard messages; relation direction ≥ 95%; lures
    tagged ≥ 90%; definition messages ≥ 98% (the architect's fan-out point).
  - **Stop:** hard-message attachment < 85% after a week of prompt work.
- **(c) Reported only:**
  - the same 300 messages written by the 27B, memory-blind and with today's procedure (this settles the root);
  - the owner's 300 labelled messages.

**Step 2. If (a) and (b) pass: the full describer and resolver.**
- **Where:** developed on dev, then judged on a fresh, never-viewed seed and on test 2's sealed world (per section 4).
  The lines and the resolver's hash are committed first.
- **Pass:**
  - thinking off ≥ today's index with thinking + 5 points, with the people-resampled interval above 0;
  - wrong merges ≤ 1–1.5%;
  - worst-person forgetting ≥ 90–95%;
  - deletions identical to never-seen;
  - reversed order identical;
  - facts filed right ≥ 85%, so it can't win by refusing to decide;
  - median answer ≤ 64 tokens.
- **Stop:** wrong merges > 3%; thinking off < 70%; a fall of more than 10 points from dev to the fresh seed.
- **Guards against the designers' advantage:**
  - an outside stress world, built by someone who wrote no resolver code, with unseen forms, roles never named,
    perspective flips and name clashes;
  - a REALTALK labelling set.
- **Cost:** about 5–8 L40S hours (about $10–20) and about 3 weeks of one person.

**Side bets (one day in five):**
- **The late-connection gate** (neuroscientist, under 1 GPU hour). Shown a pair directly, does the 4B say "fits" for
  ≥ 80% of planted pairs while staying near 2% on lures? If not, the shower-moment feature is out of reach at phone
  size.
- **"Train the step, never the facts"** (engineer, 2 weeks, 6–10 L40S hours).
  - A small add-on to the 4B (LoRA), trained only on generated lives, to take the middle step of a two-step question in
    one pass among distractors. Personal data never enters the weights.
  - Pass: two-step at 100 notes ≥ 85% (today 40%) with no loss on real text.
  - This is the closest anyone came to the 2017-shaped question. It is still known territory (Guo et al. 2025).
- **The architect's from-scratch "write locally, read globally" model:** a laptop-scale long bet, only after step 1.

## 7. Odds, as each member gave them (round 2)

See the table in section 2. Every member: about 1–2% for a field-level breakthrough. About 15–20% for the full pass
line; about 35% for the headline win; about 25–30% for passing both ends of step 1.

## 8. What the panel did not deliver

- **No 2017-class idea.** The engineer, told to keep going until he found an answer, concluded after real effort: "on
  strong C4, no shift is available within this team's resources". The achievable version is a nightly-rebuilt belief
  summary in the model's context.
- **No real-chat evidence** beyond the researcher's hand count of REALTALK. The owner's own messages are the missing
  data.
- **No independent votes.** Five instructions to one model.
- **The 2017 lesson cuts both ways.** As the founder and researcher put it, 2017 was not first principles. It was one
  deletion, forced by the bottleneck of a working system, proven on a shared test. The project is close to having a
  working system; the shift, if there is one, will show up where that system breaks in real use.
