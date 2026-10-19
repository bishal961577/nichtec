# Learning on the Job From First Principles (October 2026)

The order-gap test stopped. Its saved data showed where the learning broke:
- the 27B model wrote 1,640 valid guesses at the hidden rule;
- 61% of them contradicted the very examples it was looking at;
- they averaged 4 conditions, against the truth's 2;
- 0.4% were right.

All 40 examples together fit exactly one simple rule, the true one. So the information was there, and the learning
step failed. This report derives a learner from what learning is, not from existing methods, and pre-registers the
first two tests. Sections 1–4 were written and committed before any run.

## 1. What learning is

1. **Learning means finding the simplest explanation that fits everything seen so far, then acting on it.**
   - A simple explanation that fits many cases is unlikely to fit by luck, so it predicts new cases.
   - A complicated one can be bent to fit anything, so it predicts nothing. That is memorising, not understanding.
   - Formally this is Bayes' rule with a simplicity prior: the minimum description length principle, and Solomonoff
     induction as its ideal.
   - The 27B model broke both halves. Its guesses did not fit the evidence, and they were not simple.
2. **The evidence for an explanation is a sum over experiences, so its order does not matter.**
   - The log-probability of the data under an explanation is the sum of one term per case.
   - Counting how many cases an explanation gets right is exactly this sum.
   - So the repo's starting idea, learning by counting so that order cannot matter, is not a trick. It is what
     correct inference looks like when the cases come from a fixed world. (When the world changes, the explanation
     must include the change; the sum stays a sum.)
3. **The hard part is search, not scoring.**
   - Possible explanations are endless, so they cannot all be tried. A learner needs a guesser with good intuition
     about which kinds of explanation are worth trying.
   - A language model is a strong guesser: it has seen many kinds of explanation.
   - The test showed it is a poor judge of its own guesses, so it should not be the judge. Code can judge exactly:
     run each guess on every case, count, and charge for length.
4. **The information about what to try next is in the mistakes.**
   - The cases the current best explanation gets wrong are the only ones that say how to change it.
   - So the guesser should be shown those cases, not an arbitrary recent window. The order-gap test used the last 10
     cases. In a typical 10-case window about 9 different simple rules fit, so the guesser was starved by design.
5. **Thinking is slow, so proven lessons should become instinct.**
   - People turn practised lessons into fast intuition, largely during sleep, by replaying them.
   - For an AI, that means training proven lessons into its weights. Weight updates do not commute, which was the
     repo's original problem.
   - The fix follows from point 2. Never train on the stream. Rebuild the weights each night from the current *set*
     of proven lessons. The weights then depend on what was learned, not on the order it arrived in. This needs no
     new mathematics, only a rule about what is trained, and when.
6. **Accountability follows.** Every behaviour traces to a lesson, every lesson to its tally of evidence, and every
   piece of evidence to the notebook. Deleting a case changes the tallies, and the next night's rebuild removes what
   depended on it.

**What could break this (stated before testing):**
- **The guesser cannot imagine the right kind of explanation.** Then no judge helps. A small model imagines less than
  a large one, and search (more guesses, aimed at mistakes) may or may not make up for it.
- **Real lessons are fuzzy.** "This client prefers short emails" is not a Python expression. Checking it against past
  outcomes needs the model's own judgement, which is noisy.
- **The world changes.** Explanations then need time in them.
- **Rebuilding weights every night costs compute.** It stays small only if lessons are compact, which point 1 says
  they should be.

## 2. What this stands on

This is not presented as new in its parts:
- **Bayesian concept learning with a simplicity prior** (Tenenbaum's number game; Solomonoff; MDL).
- **DreamCoder's wake–sleep cycle (Ellis et al., 2021):** solve tasks by search while awake, compress what worked
  and train the guesser while asleep.
- **Language-model guessers with Bayesian scoring:** Ellis 2023; Hypothesis Search, ICLR 2024; Piriyakulkij et al.
  2024.
- **CL-Bench, as the measure:** frontier agents capture 25% of what could be learned on the job.

What none of them shows, as far as the earlier checks found:
- a phone-sized model with this loop learning on the job as well as a large model;
- a whole system whose final state, and later its weights, depend only on the set of experiences, with every
  behaviour traceable and deletable.

Prior work is used here as a foundation, not as a reason to stop. The question is whether the system works, not
whether its pieces are new.

## 3. Agent D, the scientist

Agent D runs on the same 12 tasks, 5 orders and test sets as the order-gap test, so it can be compared directly.

**What it keeps:** every case seen (the notebook), and every guess ever made with its exact tally on all cases.

**How it judges a guess**, by its description length in nats (lower is better):
- **Errors:** each case the guess gets wrong costs ln(0.95/0.05) ≈ 2.94. This is a deterministic rule with an
  assumed 5% noise.
- **Length:** each condition (each comparison) in the guess costs ln(58) ≈ 4.06. That is the cost of naming one
  condition out of the 58 single conditions over the four attributes.
- So an extra condition is worth adding only if it fixes at least 2 errors.
- Ties go to the shorter text, then alphabetical order.

**How it guesses:**
- After every 5 cases (8 rounds), the model gets 4 new rules. Sampling settings are as in the order-gap test.
- Its prompt holds **all cases seen so far**, listed in a fixed sorted order, so the prompt does not depend on
  arrival order.
- It also holds **D's current best rule and the cases that rule gets wrong.** The model is asked for the simplest
  rule that fits every case.
- Round 1 has no current best, so it gets the cases only.

**How it answers:** with the best-judged rule. Before the first round it answers with the majority answer so far.

**Free references, computed without a model:**
- **Brute force:** the same judging applied to every one- and two-condition rule (2,578 rules). It shows what
  perfect search would reach. It is not a competitor, because in open-ended work no such list exists.
- **D-count:** the same pool of guesses judged by count alone, without the length charge, like agent B. It shows
  what the simplicity charge adds.

## 4. Tests and lines, fixed before any run

**Test 1, Qwen3.5-27B-FP8.**
- **Comparison:** the A and B numbers already measured on the same tasks (A 83.7%, B 81.2%, random orders).
- **Pass:** D scores at least 95% on random orders, and a good rule (≥95% agreement over all 576 items) is in D's
  pool in at least 80% of streams. Then giving the model the two habits, checking against all evidence and
  preferring simple, fixes the induction failure.
- **Stop:** D scores under 90%. Then the loop does not fix induction, and the next step is to diagnose the guesses
  again, not to try a variant.
- **In between:** partial. Report it, diagnose, and decide with the owner.

**Test 2, Qwen3.5-4B (phone-sized), the same agent D.**
- **Pass ("small with the habits beats big without"):** 4B-D scores at least 88.7%, which is A on 27B plus 5
  points.
- **Stop:** 4B-D scores under 83.7%, A on 27B. Then a phone-sized guesser is too weak for this loop, even on simple
  rules.

**Reported for both, with no line:**
- D's order gap, measured as in the order-gap test;
- D-count;
- how often D's guesses contradict the cases they were shown, against 61% before;
- the average number of conditions per guess, against 4.0 before;
- brute force.

**Cost:** two dedicated Modal endpoints, used one at a time. That is about 480 guessing calls per model, likely a
few dollars in total. No forecast is made.

**After this, only if test 1 passes:**
- an open-ended task, where no list of rules exists and lessons are text judged against recorded outcomes;
- then the nightly rebuild ("sleep").

Each gets its own pre-registration.

## 5. Test 1 result (Qwen3.5-27B-FP8, 18 October 2026): stop, by 0.7 points

**How it ran:**
- On a dedicated Modal endpoint (SGLang), in 3.6 minutes: 480 guessing rounds and 1,540 valid guesses.
- The probe passed.
- The raw data are in `results/scientist_Qwen3.5-27B-FP8/`.

| Agent | Random orders | Decoy first | Decoy last | Order gap | Final rule good |
|---|---|---|---|---|---|
| **D, scientist (pre-registered)** | **89.3 ± 3.0** | 91.7 | 91.5 | −2.2 | 43% |
| D-count (same guesses, judged by count only) | 92.4 ± 1.5 | 91.7 | 94.2 | 0.8 | 43% |
| A, full history (order-gap test) | 83.7 | 81.2 | 82.5 | 2.5 | — |
| B, old ledger (order-gap test) | 81.2 | 86.2 | 81.2 | −0.1 | 12% |
| Brute force over 2,578 simple rules | 99.4 | | | | |

**Line:** D scored 89.3, under 90. **Stop.** The loop as specified does not fix induction. The pre-registered next
step is a diagnosis, not a variant.

**Diagnosis from the saved guesses** (no new calls):
1. **The guesser improves as evidence grows.** The share of good guesses rose from 0.4% after 5 cases to 48.9% after
   40. In 68% of streams the pool held a rule that was at least 90% right on all items, against 12% good rules in
   the order-gap test. Showing all the evidence and the mistakes helps.
2. **It patches instead of rethinking.** 41% of guesses kept D's current best rule and bolted exceptions onto it. That
   is "rigid early beliefs", the failure CL-Bench names, caused here by the design: showing the current best invites
   patching. Guesses grew from 2.5 conditions after 5 cases to 4.8 in the middle of the stream.
3. **It still cannot check itself.** Even with every case in front of it, only 49–76% of guesses per round fitted
   all of them (37% contradicted them overall, against 61% before). Code has to do the checking, and when it did, the
   result beat the full-history agent by 5.6 to 8.7 points.
4. **My simplicity charge was the wrong measure.** D and D-count chose different rules in 20 of 60 streams, and
   D-count won 14 of those against 2.
   - The usual case: the pool lacked the true rule. The length charge then preferred a short wrong rule (for
     example 2 conditions with 4 errors) over a longer rule made of the true rule plus a narrow exception (3 errors,
     95% on the test set).
   - Counting comparisons charges an exception that covers one item of 576 as much as a condition that splits the
     items in half.
   - A better-founded charge would cost a rule by how much it claims (the size principle), not by how many symbols it
     uses. Choosing it now, after seeing the data, would be a forking path. It needs fresh tasks.
5. **The system is unaffected by order,** as designed: decoy orders scored slightly higher than random ones.

**What this means:**
- The bottleneck is the guesser's search: it patches, does not check itself, and is not drawn to simple rules even
  when asked.
- The code-side judge works.
- Neither D (89.3) nor D-count (92.4) reached the 95 line, against 99.4 for exhaustive search, which shows the
  information was there.

**Next, as pre-registered:**
- Test 2 (Qwen3.5-4B) still runs, with its own lines.
- Any redesign drawn from this diagnosis must be pre-registered and tested on fresh tasks, not on these 12. A
  redesign drawn from one diagnosis is not a "variant", but it gets one attempt only.

## 6. Test 2 result (Qwen3.5-4B, 18 October 2026): stop

**How it ran:** the same agent D on a dedicated Modal endpoint, in 0.6 minutes. The raw data are in
`results/scientist_Qwen3.5-4B/`.

| Agent | Random orders | Decoy first | Decoy last | Order gap | Good rule in pool |
|---|---|---|---|---|---|
| **4B, D (pre-registered)** | **75.2 ± 3.4** | 65.0 | 76.0 | 10.2 | 7% of streams |
| 4B, D-count | 79.0 ± 2.3 | 65.4 | 77.3 | 13.6 | 7% |
| 27B, D (test 1) | 89.3 | 91.7 | 91.5 | −2.2 | 43% |
| 27B, A (order-gap test, the line) | 83.7 | 81.2 | 82.5 | 2.5 | — |

**Line:** 4B-D scored 75.2, under 83.7. **Stop.** The phone-sized model with these habits does worse than the 27B
model without them.

**Why:**
- **96% of the 4B model's guesses contradicted the cases they were shown,** against 37% at 27B. Only 7% of streams
  ever held a good rule. A judge cannot pick a good guess that is never made.
- **With almost no good guesses, the pool depends on order.** The rules guessed from the misleading block at the
  start stay among the best available. That is the 10-point order gap, against −2.2 at 27B. Order-freedom in the
  judge does not make the whole system order-free when the guesser fails. Section 1 listed this risk: the guesser
  cannot imagine the right explanation.

**What tests 1 and 2 show together:**
- The weak part is the language model as the writer of whole explanations. It patches, does not check itself, and
  at 4B rarely produces anything consistent with what it was shown.
- The code-side parts work: exact checking, tallies, and indifference to order when the guesses are good.
- Exhaustive search over a small rule language reaches 99.4, so the evidence was always sufficient.

## 7. The one redesign: the model names the ingredients, code combines them (pre-registered 18 October 2026)

Written and committed before any code or run. This is the single redesign that section 5 allowed. If its test 1
stops, this line of work ends.

**From first principles:**
- A rule has two parts: ingredients (traits that matter, such as "can it fly?") and a way of combining them
  ("flies or lives in water").
- **Naming ingredients takes knowledge of the world**, which language models have.
- **Combining and checking takes exact, exhaustive work.** That is what code does perfectly (99.4% in test 1) and what
  the models failed at (37% and 96% of guesses contradicted the shown cases).
- Tests 1 and 2 gave both jobs to the model. Here they are split:
  - the model only names candidate traits and answers yes or no about one trait of one item;
  - code combines the traits, checks every combination against every case, and keeps the simplest that fits.
- The toy task handed the ingredients over as four fields, so code could solve it without the model. A real test
  needs **hidden ingredients**: items given only by name, with the rule depending on traits that the model has to
  know or notice.

**What it stands on:**
- language-model-generated features combined by a classic learner: CAAFE 2023, Aug-imodels 2023;
- concept bottleneck models with model-named concepts: LaBo, Label-free CBM, 2023;
- concept induction from text: LLooM, 2024;
- hypotheses written in plain language and checked by a model: Ellis 2023.

The question here is narrower than any of these: does splitting induction this way beat letting the model write
whole explanations, and reading the whole history, when learning from experience? And does it rescue a phone-sized
model?

### Task

**Data:** UCI Zoo (Forsyth, CC BY 4.0).
- Animals given by name only, with their recorded traits.
- `zoo.data` sha256 `cddc71c2…f0315`.
- The name "frog" appears twice with different values, so both rows are dropped, leaving 99 animals.

**Hidden rules:** ten rules, each over two traits. Each was chosen in advance by three conditions:
- it uses only these traits: hair, feathers, eggs, milk, airborne, aquatic, toothed, backbone, breathes, venomous,
  fins, tail;
- 40–59 of the 99 animals are in the "yes" group;
- no single trait (or its negation) agrees with it on more than 85% of animals, so both ingredients are needed.

There is one rule per distinct pair of traits:

| # | Hidden rule | Animals in "yes" |
|---|---|---|
| 1 | airborne or aquatic | 53 |
| 2 | eggs and backbone | 40 |
| 3 | airborne or fins | 41 |
| 4 | not airborne and breathes | 54 |
| 5 | fins or not tail | 40 |
| 6 | milk or not backbone | 59 |
| 7 | not milk and tail | 40 |
| 8 | airborne or not tail | 42 |
| 9 | not aquatic and tail | 49 |
| 10 | feathers or not breathes | 41 |

**Tasks:**
- Each rule has 2 draws, giving 20 tasks.
- A task is a 40-case stream (20 yes, 20 no) and a separate 40-animal test set (20 yes, 20 no) drawn from the rest.
- **There is no order manipulation.** The order question is settled; this test is about whether the lesson gets
  learned. Each task has one random stream order.

### Agents (same model for all)

- **A, history:** the full stream in arrival order in the prompt. It says yes or no for each test animal from the
  top-20 log-probabilities, with the median threshold, as in the order-gap test.
- **W, whole-rule writer with a code judge:**
  - From all 40 cases, listed in sorted order, the model writes the hidden rule as one sentence: 4 samples,
    temperature 0.8.
  - The model then judges each written rule on each of the 80 animals ("Does this animal satisfy the rule?").
  - Code keeps the rule with the fewest errors on the stream (ties go to the shorter text) and uses it on the test
    set.
  - This is test 1's D-count, with whole explanations in plain language.
- **E, ingredients:**
  1. **Naming:** from all 40 sorted cases, the model lists up to 8 short yes-or-no questions about an animal's traits
     that might explain the groups. That is 4 samples at temperature 0.8. Lines ending in "?" are kept,
     de-duplicated by their lower-case letters and digits, up to 16 per task in order of appearance.
  2. **Reading:** the model answers each question for each of the 80 animals, yes or no, from the top-20
     log-probabilities at threshold zero. Answers are cached by (question, animal).
  3. **Combining (code):** every rule with one or two questions is tried, each question possibly negated and two
     joined by "and" or "or". Its cost in nats is the errors on the stream × ln 19, plus the number of questions
     × ln(2K) (picking one of K questions and its sign), plus ln 2 for the joining word. The cheapest wins; ties go
     to fewer questions, then text order. It is applied to the test animals with their read answers.

     Because no rule can hold more than two questions, the exception-patching that defeated test 1's length charge
     cannot occur.

### References and checks (same model)

- **Oracle:** the same combining over the 12 traits with their recorded values. It shows what the evidence allows.
- **E with the true questions:** the same combining over 12 fixed questions read by the model:
  - "Does it have hair?"
  - "Does it have feathers?"
  - "Does it lay eggs?"
  - "Does it produce milk?"
  - "Can it fly?"
  - "Does it live in water?"
  - "Does it have teeth?"
  - "Does it have a backbone?"
  - "Does it breathe air?"
  - "Is it venomous?"
  - "Does it have fins?"
  - "Does it have a tail?"

  This separates naming the wrong ingredients from reading them wrongly.
- **Reading accuracy:** the model's answers to those 12 questions for all 99 animals, against the recorded values.

### Lines, fixed now

- **Check 1, the task:** the oracle scores at least 95% on the test sets. If not, the task is faulty and nothing is
  concluded.
- **Check 2, reading:** the model's reading accuracy is at least 85%. If not, its knowledge disagrees with the
  records too often, and the test is inconclusive for that model.
- **Check 3, answering:** at most 10% of the reading and A calls have neither yes nor no among their top 20 tokens.
- **Test 1, Qwen3.5-27B-FP8:**
  - **Pass:** E is at least 5 points above both A and W.
  - **Stop:** E is less than 2 points above the better of A and W. Then splitting the jobs does not help, and this
    line of work ends.
  - In between: report, diagnose, and decide with the owner.
- **Test 2, Qwen3.5-4B, run whatever test 1 shows:**
  - **Pass:** 4B-E is at least as good as 27B-A from test 1, meaning the small model with split jobs matches the big
    model reading its history.
  - **Stop:** 4B-E is more than 5 points below 27B-A.
- **Reported without lines:**
  - A, W and E on both models;
  - E with the true questions;
  - the questions E chose, and how often they match a true trait;
  - per-trait reading accuracy;
  - the number of calls.

**Cost:** about 25,000 short calls per model, many repeated across tasks and cached. That is likely 10–20 minutes on
each dedicated endpoint, and a few dollars in all. No forecast is made.

## 8. Ingredients test 1 result (Qwen3.5-27B-FP8, 18 October 2026): in between

**How it ran:**
- From this session against a dedicated Modal endpoint, in 9.6 minutes.
- 16,786 answer calls, after caching. No answer was anything other than yes or no.
- The raw data are in `results/ingredients_Qwen3.5-27B-FP8/`.

| Agent | Test accuracy (20 tasks) |
|---|---|
| A, full history | 86.2 ± 1.5 |
| W, whole rule in words, judged by code | 83.6 ± 2.2 |
| **E, model names traits, code combines** | **89.4 ± 1.7** |
| E with the 12 true questions (reference) | 93.2 ± 1.3 |
| Oracle: recorded traits, same combining | 99.2 |

**Checks:** all passed.
- The oracle scored 99.2, against a line of 95.
- Reading accuracy was 95.4%, against a line of 85.
- No answer fell outside yes or no.

**Lines:**
- **Pass** needed E at least 5 points above both A and W.
- **Stop** applied if E was less than 2 points above the better of them.
- E is 3.1 above A (± 1.6 paired; 13 wins, 6 losses) and 5.8 above W (± 2.7). **In between.** By the plan:
  report, diagnose, and decide with the owner.

**Where the points go** (paired, across the 20 tasks):
- **Reading the traits wrongly costs 6.0 points** (oracle 99.2 → E with the true questions 93.2).
  - The model's yes-or-no answers about single traits are 95.4% right: teeth 90%, tail 91%.
  - Two traits per rule compound the errors.
  - Some of the disagreement is the dataset's own labelling.
- **Naming the wrong traits costs 3.9 points** (93.2 → E's 89.4). E often used near-synonyms or class questions
  instead of the traits: "Is it a fish?" for fins, "Does it have wings?" for airborne, "Is it a mammal?" for milk.
  Its final rule used one of the hidden rule's own traits in 80% of tasks.
- **Splitting the jobs beats writing whole rules:** E beat W by 5.8 points. W's sentences were often wrong
  generalisations, such as "the animal is a mammal or a reptile".
- **With the right ingredients, splitting beats reading the history clearly:** E with the true questions beat A by
  7.0 ± 2.1, winning 16 tasks and losing 4. This is a reference, not the agent under test.
- **At 27B, reading the history is strong on this task** (86.2, against 83.7 on the abstract toy rules), because the
  model knows these animals.

**What it means:**
- Splitting is the better design.
- Its limits are the model's yes-or-no knowledge of single traits and its habit of naming near-synonyms.
- At 27B the margin over plain history reading is real but small: +3.1, inside the in-between band.
- Test 2 (4B) runs next, as pre-registered. That is where splitting matters most for a phone.

## 9. Ingredients test 2 result (Qwen3.5-4B, 19 October 2026): pass

**How it ran:**
- From this session against a dedicated Modal endpoint, in 7.9 minutes, with 16,582 answer calls.
- The 4B model starts its answers to the history prompt with "Based…", so the code's own probe, which is not a
  pre-registered line, was moved to a trait-reading question before the run (commit "Add --probe-reading").
- Pre-registered check 3 governs A's answers. It passed: 0.0% of answers had neither yes nor no in the top 20.
- The raw data are in `results/ingredients_Qwen3.5-4B/`.

| Agent (4B unless marked) | Test accuracy (20 tasks) |
|---|---|
| A, full history | 59.0 ± 2.2 |
| W, whole rule in words, judged by code | 73.6 ± 2.3 |
| **E, model names traits, code combines** | **87.4 ± 2.1** |
| E with the 12 true questions (reference) | 88.9 ± 1.5 |
| 27B A, full history (test 1, the line) | 86.2 ± 1.5 |
| 27B E (test 1) | 89.4 ± 1.7 |

**Checks:** all passed. The oracle scored 99.2, reading accuracy was 94.0%, and no answer fell outside yes or no.

**Line:**
- Pass needed 4B-E to be at least 27B-A. **4B-E is 87.4, against 86.2: pass.**
- Paired, the difference is +1.1 ± 1.9 (10 wins, 7 losses). That is "matches", not "beats". The pre-registered
  line was "at least as good as", and it was met.

**Within the 4B model, splitting the jobs is the whole difference:**
- **E against A: +28.4 ± 2.6, winning all 20 tasks.** The small model reading its history is close to chance.
- **E against W: +13.8 ± 2.7** (16 wins, 3 losses).
- Naming costs only 1.5 points (E with the true questions 88.9, against E 87.4).
- 4B-E is 2.0 ± 1.6 below 27B-E.

**What it means:**
- A phone-sized model that cannot learn these rules from its history (59%) learns them as well as a model about
  seven times larger reading its history (87–86%). The change is not more knowledge. It is a division of labour: the
  model names and reads single traits, which it can do, and code combines and checks, which the model cannot.
- This is the first result in the project that supports the owner's idea: a small model with good learning habits
  matching a large model without them.

**Limits:**
- One task family (animals).
- Rules of two traits.
- Traits the model already knows.
- Labels that are exact.
- A 40-case stream.
- Not yet open-ended work, fuzzy lessons, a changing world, or lessons moved into the weights ("sleep").
- 4B-A's 59.0 may be held down by the model's habit of starting with "Based…". That does not affect the line,
  which compares with 27B-A.

**Status of the redesign:**
- Test 1 (27B) was in between: E beat A by 3.1, against a 5-point pass line.
- Test 2 (4B) passed.
- What comes next is the owner's decision. Anything further gets its own pre-registration.
