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
