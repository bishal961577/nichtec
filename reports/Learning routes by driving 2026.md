# Learning routes by driving (pre-registered 20 November 2026)

**The question, in one line:** does a memory that learns its routes from being used and corrected file new facts
better than the same memory with fixed routes plus spot fixes?

## 1. Why this test

The owner's picture: a station master's panel, where every fact sits at a station and the present moment sets a route
to the stations that hold what is needed. Then a second point: someone who builds the routes and drives them learns to
navigate, the way London taxi drivers learn the city by driving it and correcting their mistakes.

What this project already knows:
- Running trains on good track works: in test 2 a perfect index beat a fair search agent (86.1% against 62.8% on the
  hard questions).
- Laying the track is what breaks: the models file some facts under the wrong person. With every definition
  confirmed, the 27B still files 1.52% (dev) and 1.70% (fresh world) of everyday facts under the wrong person; the 4B
  5.56%.
- Confirming definitions does not fix this (two ends check, step 3: 10.1% to 8.5% for the 4B). Those confirmations
  were about who people are, not about where a fact went.
- A diagnosis on the dev world, done before this plan: in most wrong filings the right person is named in the same
  message, and the fact went down the wrong switch, usually a pronoun. 27B with every definition confirmed: 11 of 11.
  4B: 31 of 44 (the other 13 need better reading).

So the test: let the owner correct the few wrong facts they happen to see, and let the memory learn from each
correction which switches to throw, so that later facts of the same kind go the right way.

## 2. What is reused (no model calls, free, on a CPU)

- The generated chat worlds of the two ends check: the dev world (seed 101) and the fresh world (seed 20261117), first
  1,500 messages each, with their deletions; the truth is known.
- Cached descriptions: Qwen3.5-4B on dev (step 2), Qwen3.5-27B-FP8 on dev (step 4) and on the fresh world (step 5).
- The resolver as frozen after step 4 (antecedent rule fixed, reporting-verb filter on), in two conditions: own
  definitions (nothing confirmed) and every definition confirmed.
- Test 2's everyday wrong-merge count, unchanged. The simulation reproduces it exactly before anything is changed
  (self-check): 7.41%, 5.56%, 2.49%, 1.52%, 2.56% and 1.70% for the six conditions.

## 3. The track and its switches (`brain/drive_routes.py`)

- The resolver has worked out who every mention in a message is. A fact is filed under the person of the mention the
  describer says it is about (the default route), or under another person named in the same message.
- Each possible re-routing has a **route type**, built only from what the system can see: both mentions' kinds
  (pronoun, full name, first name, role, "X's flatmate", name with group), whether each is the teller, whether the
  other mention is the pronoun's antecedent, whether the message reports speech ("told me", "says"), and which mention
  comes first.
- **A route type is a switch.** A correction saying a fact belonged to the other mention adds 0.6 to it. A correction
  saying a re-routed fact belonged where it was before takes 0.6 away. The switch is thrown when its weight passes 1.0,
  that is after two net corrections, and from then on facts of that type go the other way.
- No truth is used to route. The truth only plays the owner, who corrects facts shown to them that are filed wrong.

## 4. The owner's corrections

- Each day, 3 of the facts filed so far are shown to the owner (as an assistant would use them in a chat), the same
  facts in every arm (common random numbers).
- The owner corrects each one that is filed wrong, at most 2 a day, by naming who it is about. If the owner cannot
  tell (the fact matches none of the message's true facts and the message has several people), there is no correction.
- 10 random seeds for which facts are shown. Sensitivity, reported without lines: 1 and 10 facts shown a day.

## 5. Arms

- **fixed:** no corrections (the step 5 system as it is).
- **patch:** each correction moves that one fact (a fixed map with errata).
- **drive:** each correction moves that fact and teaches the switches; every night all earlier facts not corrected are
  re-routed with the switches as they are.
- Reported: the routing floor, the facts no route in their message could file right.

## 6. Measures and lines (fixed now)

- **Primary: the wrong share among facts never shown to the owner,** drive against patch. Nothing corrected these
  facts directly, so any difference comes from what the switches learned. Pooled over the 10 seeds; the 95% interval
  of the difference comes from resampling messages (2,000 resamples).
- **Lines, per condition:**
  - **LEARNS:** drive files at least 30% fewer of the never-shown facts wrong than patch, with the 95% interval of the
    difference above zero.
  - **NO GAIN:** less than 10% fewer, or the interval reaches zero.
  - **BETWEEN:** anything else.
  - **INCONCLUSIVE:** patch leaves fewer than 10 never-shown facts wrong per seed: too few errors to learn from or to
    measure.
- **The main verdict is the 4B on dev with its own definitions:** the most errors, and nothing confirmed. The other
  five conditions get their own verdicts. I expect the 27B conditions to be INCONCLUSIVE, since they have only 11-18
  wrong facts in all.
- **Reported without lines:** wrong share over all facts at the end; corrections given; facts moved (wrong to right,
  right to wrong); switches thrown; test 2's everyday count on each arm's final index (seed 0); the 1 and 10 a day
  sensitivity.

## 7. Forecast (written before the run)

For the main condition: LEARNS 35%, BETWEEN 25%, NO GAIN 30%, INCONCLUSIVE 10%. The errors are concentrated in a few
route types, which helps. But a switch that sends pronoun facts to the other named person can also move facts that
were right, and two corrections are thin evidence.

## 8. Limits, stated before the run

- **Generated worlds, the same templates in both.** The route types may fit the generator's sentence patterns better
  than real chat would. Real chat (REALTALK, judged by Claude) is not tested here.
- **The generator's pronoun quirk:** she, he or they is given at random whatever the name. The switches do not use
  gender, so the quirk neither helps nor hurts them, but it makes pronouns harder than in real text.
- **The resolver's bindings are computed once over the whole history,** the same for every arm, as in the registered
  measures.
- **The owner is simulated from the truth,** with a fixed policy.
- **No language model learns here.** The learning is in the track. Whether the model's own weights can learn the
  routes is the GPU version, worth paying for only if this one shows that driving repairs the map.
- **Designer's advantage:** the error diagnosis in section 1 looked at the dev world's errors before this plan, and the
  candidate set (other mentions in the same message) came from it. Nothing was tuned on any condition: the step
  (0.6), the margin (1.0), the route type, the 3 a day and the cap are fixed here, before any arm is run. The fresh
  world's errors were not looked at, though the self-check printed its baseline (1.70% and 2.56%, already reported in
  step 5) and its routing floor.

## 9. Procedure

1. Self-check (done): the fixed arm reproduces test 2's everyday count on all six conditions; drive with a step of 0
   equals patch; routing uses no truth.
2. Run once: `python brain/drive_routes.py --stage all`. Results in `results/drive_routes/`.
3. Write up, whatever the result.

## 10. Result (run once, 20 November 2026, commit 5d59996; 23 seconds on a CPU; `results/drive_routes/`)

**Main verdict (4B, dev, own definitions): NO GAIN.** Driving filed 5.8% fewer of the never-shown facts wrong than
spot fixes alone (7.22% to 6.80%), against a 30% line, and the interval of the difference reaches zero (-0.04 to
+0.86 points).

| Condition | Wrong at start (fixed) | Wrong at the end: patch / drive | Corrections given | Never-shown facts wrong: patch / drive | Reduction | Verdict |
|---|---|---|---|---|---|---|
| **4B dev, own definitions (main)** | 7.41% | 4.62% / 4.38% | 21.8 | 7.22% / 6.80% | 5.8% | **NO GAIN** |
| 4B dev, every definition confirmed | 5.56% | 3.68% / 3.35% | 14.9 | 5.71% / 5.16% | 9.6% | NO GAIN |
| 27B dev, own definitions | 2.49% | 1.73% / 1.70% | 5.5 | 2.81% / 2.77% | 1.6% | NO GAIN |
| 27B dev, every definition confirmed | 1.52% | 1.07% / 1.03% | 3.2 | 1.74% / 1.67% | 3.8% | INCONCLUSIVE |
| 27B fresh, own definitions | 2.56% | 1.62% / 1.35% | 6.6 | 2.51% / 2.07% | 17.6% | BETWEEN |
| 27B fresh, every definition confirmed | 1.70% | 1.18% / 0.85% | 3.7 | 1.92% / 1.39% | 27.7% | INCONCLUSIVE |

Corrections are means over the 10 seeds. About 36-39% of facts were shown to the owner at least once.

**What happened:**
- **The switches learned the right thing, but almost never had enough to learn from.** Over about six months of
  messages the owner gave the 4B memory about 22 corrections, and the 27B memory 3-7, because few facts are filed
  wrong and the owner only sees a few a day. Each switch needs two net corrections, and the corrections were spread
  over several kinds of mistakes, so only a handful of switches were ever thrown: on average 4 facts were re-routed in
  the main condition.
- **When a switch was thrown, it was usually right.** Main condition: 3.3 facts moved from wrong to right, 0.6 from
  right to wrong. On the fresh world: 2.1-2.4 wrong to right, 0 right to wrong. The switch thrown most often is the
  same in every condition: a fact the describer put on a pronoun belongs to the full name earlier in the same message,
  which the antecedent rule did not pick.
- **Spot fixes did most of the work.** Correcting the facts the owner happened to see cut the 4B's wrong filings from
  7.41% to 4.62%. Learning from those corrections added only 0.24 points more.
- **More feedback helps, in proportion.** With 10 facts shown a day (about 38 corrections for the 4B), the reduction
  on never-shown facts was 16-41% (BETWEEN for the main condition). With 1 a day it was 1-4%. Learning by driving
  needs many drives.
- **Some errors cannot be fixed by routing at all:** 26 of the 4B's 58 wrong facts (own definitions) have the right
  person named nowhere in their message. Those need better reading, not better switches.
- **Test 2's count on each arm's final index (seed 0)** agrees: 4B main 7.41% (fixed), 4.85% (patch), 4.60% (drive);
  27B fresh, definitions confirmed 1.70%, 1.27%, 0.85%. No condition reaches the 1.5% line except the 27B with
  definitions confirmed, with or without driving.

**What it means for the owner's idea:** the route-builder that learns from its own mistakes goes in the right
direction and almost never makes things worse. But at the rate a person would realistically correct an assistant,
it gets too few lessons to matter: a London taxi driver learns from thousands of drives, and this memory got about
twenty corrections in six months. For learning by driving to pay off, the memory needs to learn from every use, not
only from corrections: for example, a fact the owner sees and accepts is evidence for its route, and a conflict
between two facts is evidence against one of them. That would be a new test with its own lines, not a re-run of this
one.
