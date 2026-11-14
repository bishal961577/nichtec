# Test 2 review: senior neuroscientist

Scripts are in the session scratchpad (asym.py, sep.py, cands.py, hub.py, sc.py, ev.py, linksim.py, lure.py, world_checks.py, stub_an.py). I did not change any repository file. Every number below is from the dev world.

**1. Verdict: run with changes.** The core design is sound. Rows are only added, each points to its raw message, and the reader sees only raw messages. That is the hippocampal index done right, and it makes deletion clean by construction. But the generator, scorer and lines freeze after this review, and three defects would decide the verdict on their own.

**2. Does the world test the phenomena that matter?**
- **Pattern separation at write time: barely.** [checked] After both people of a same-first-name pair are known, a bare "Mira" appears in only 5 fact-bearing messages. Same-name people are otherwise told apart by full name (401) or circle (85). Near names (92 bare cases) are different strings. The writer's candidate code matches exact words, so it usually never even shows Joaquin when the message says "Joao". The code separates them, not the 4B. Pattern completion (one person written several ways, typos, "Jo") is absent.
- **Pronouns.** [checked] All 353 have their antecedent in the same message; none crosses messages.
- **Other gaps.** No message states two facts, and "my boss" or "my flatmate" never changes holder over 22 months.
- **Fan effect.** It is present (close people have ~240 messages each) and causes the opposite of the human effect (flaw H1).
- **Encoding specificity: weak.** [checked] 39% of job and hobby descriptors appear word for word in the source message. The other five slots always share the value word.
- **Events.** These are single-message (person, activity, date) records: date lookups plus calendar arithmetic, not episodic recall. There is no "where", no multi-message episode, no order or frequency question.
- **Updates and source monitoring.** History is kept and every row has its message id: good. Link precision and source misattribution are never reported.
- **Lures.** [checked] 242 hard questions involve a person with a same-slot lure. Report their accuracy as a false-memory rate.

**3. Flaws, ranked**

**Critical**
- **C1. Asymmetric relations are inverted in the generator.** [checked] In 23 of 180 two-hop questions, the gold says "Agnes Pereira's landlord" is Rosa Kiss, but the note says "Rosa Kiss rents from Agnes Pereira" (Agnes is Rosa's landlord). 41 "via" mentions are mislabelled the same way: "Gideon Molnar's mentor" is labelled as Ugo, whom Gideon mentors. The held-out godparent relation inherits this.
  - Fix: at `messy_world.py:713`, for asymmetric relations use `way == "in"`; at `:426`, test `link_known[(pid, rel, b)]`.
- **C2. The scorer marks right answers wrong.** [checked]
  - In event_who, 35 of 70 right answers are a close person, and "Your mum." scores wrong (`messy_index.py:628-633`).
  - 214 non-update attribute questions (40 of 53 role questions) have an earlier value. "flute (previously bass guitar)" scores wrong because only update questions pass `old`.
  - Fix: accept "my/your <role>", the nickname, or a unique first name plus circle. Pass `old` for all attribute kinds.
- **C3. The audit is one-sided and misses lenient scoring.**
  - Only B's wrong answers are audited, and the audited score decides. That raises B but not C, which inflates the B − C margin.
  - It never catches wrong answers scored right. [checked] A cut-off (empty) answer counts as right on deleted questions. A stale value plus "don't know", "gave" or "sold" counts as right on retractions. On event_last, any listed date counts.
  - Fix: score every cut-off answer wrong; require an explicit "not in my notes"-style abstention for deleted; event_last must name the gold date as the latest; audit B and C alike, sampling both right and wrong answers.

**High**
- **H1. Hub capture in desc/who matching.** Taking the best match over all of a person's rows rewards people with many rows. [checked] With a perfect index and perfect clues, the right person ranks first only 74.3%; a close person wrongly ranks first in 44 of 300. 11 of 12 oracle desc failures are this.
  - Fix: subtract 0.03 × ln(rows). That gives 88.7% first and 2 hub errors. This is the ACT-R fan term ("S − ln fan") [recall].
- **H2. The wrong-merge metric has false positives and is diluted.**
  - A correct link row filed under the anchor of "Kofi's flatmate" counts as a merge, because the anchor is not in the truth mentions (138 via mentions are in fact-bearing messages).
  - Kinship lures ("My sister's mother works as a pharmacist", which is mum) punish correct inference.
  - The near-name rate is computed over all 1,407 facts about same/near-name people, when only about 180 need separating.
  - Fix: add via anchors to the truth mentions; drop kinship lures for close people; compute the near-name line on the hard subset.
- **H3. The generator is too polite.** Fix: about 150 bare "Mira" mentions resolvable only by recency or topic; 30% of pronouns in the next message; some role turnover.
- **H4. The writer's candidate list misses people.** [checked] Even with a perfect index, a via target is not shown 47% of the time, first name with circle 6.5%, bare first name 4.5%.
  - Fix: add the linked people of every name-matched candidate, not just the top 10; parse "X's <noun>".
- **H5. C's third search is dead.** `found[:60]` keeps the oldest messages (`:546-550`), so the third follow-up adds nothing.
  - Fix: 13 new messages per follow-up, or rerank all found messages.

**Medium**
- **M1. A leak is a guessing lottery.** [checked] B's retrieved notes contained the deleted value 0 times in 100 (C's: 35). So a B "leak" can only be a guess, yet one guess means STOP.
- **M2. Events are confounded.** 39 of 170 event_last answers need the weekday of a date, and messages carry no weekday stamp. 11 of 170 have a later "ran into", "saw" or "picked me up" message.
  - Fix: stamp the weekday; exclude those 11 cases.
- **M3. The link threshold of 0.75 fails open-vocabulary kinds.** [checked] "neighbour" vs "lives next door to" scores 0.63; "colleague" vs "works with" 0.54.
  - Fix: let the 4B pick among the anchor's link rows.
- **M4. Encoding specificity.** Add a third wording or category cues ("the one with a reptile").

**Low:** "A cat." scores wrong for tabby cat (also bass, snake, bees). The writing-cost line ignores prompt tokens.

**The three things you asked about.** Matching every described fact (min over facts) is right in principle; its scaling is flaw H1. The one-link limit is fine. The 60-message cap binds in about 6% of questions and becomes rarer once H1 is fixed.

**4. Changes to the section 5 lines**
- **Near names.** Judge the line on the hard subset: messages that refer to the person without the full name while a same- or near-name competitor is known. PASS ≤ 5%, STOP > 10%. As written, a writer that misfiles 1 in 5 hard cases passes.
- **Leak.** STOP only on a traceable leak: the value survives in a row or in a shown message. Report raw value-naming against the pages arm.
- **New outcome, "SEARCH WINS".** If C ≥ 85% with desc, who and two each ≥ 75%, and B − C < 5, choose search even when B fails. Otherwise a strong C beside a weak B ends as BETWEEN.
- **Report three more measures:** link precision, lure intrusion, and rows whose value is not in their own message.
- Keep the other lines.

**5. Forecast (test world, after the critical fixes)** [judgement]

| Measure | Forecast |
|---|---|
| Wrong merges | 1.5–4% (median 2.5%) |
| Near-name wrong merges, as coded | 1–3% |
| Near-name wrong merges, hard subset | 8–20% |
| B, hard set | 70–82% (median 76) |
| B desc / who / two | 50–68 / 55–72 / 60–78 |
| B on events | 62–78% (about 60 as coded) |
| C, hard set | 55–70% |
| B − C on desc + who + two | +5 to +20 |

**Verdict probabilities, as registered:** STOP 50%, BETWEEN 40%, PASS 6%, C 4%.

This test cannot tell whether a nightly re-filing pass, from the raw notes, would repair merges. That is the consolidation question my "what would change my mind" condition depends on.
