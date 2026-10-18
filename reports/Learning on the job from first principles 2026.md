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
