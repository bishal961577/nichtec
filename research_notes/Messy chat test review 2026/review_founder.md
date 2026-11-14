# Test 2 review: curious founder

FOUNDER REVIEW, TEST 2. VERDICT: RUN WITH CHANGES (do not run it as it stands).
All numbers come from the committed code, 18b2849. During my review someone else changed brain/messy_world.py and brain/messy_index.py in the working tree; I did not. My scripts are in the session scratchpad (probe1-4.py, stubvar.py). Labels: [C] I checked it, [R] from memory, [E] my estimate.

1. WILL THE RESULT TELL THE TEAM WHAT TO BUILD? Not as the outcomes are wired.
- PASS is almost out of reach. The stub's writer files 94% of facts with 0.8% wrong merges, and its reader is perfect. That combination still fails: desc 69.3%, who 74.7%, ratio to oracle 0.898 [C]. I reran the stub with worse writers and a perfect reader [C]:
  - 10% of facts missed and 2% misfiled: B 80.8%, verdict BETWEEN;
  - 20% missed and 3% misfiled: B 67.9%, verdict STOP.
  So PASS needs a 4B, writing with thinking off, that files about 95% of facts.
- "C: choose search" only fires when B passes everything else. If search simply beats the index, the verdict is BETWEEN, so "build search" can never be the answer.
- STOP points the team the wrong way. "A bigger model overnight" means writing in the cloud, because a 27B does not run on a phone. "A as a window" already failed in test 1 (20.4%). The cheap step nobody has tried is the same 4B writing with thinking on.
- The oracle arm does separate filing problems from finding problems. Write down now what the team builds in each case.

2. DO THE MESSAGES LOOK LIKE A REAL USER? No. They read like a form filled in one line at a time [C]:
- median 7 words, at most 20;
- never two facts in one message, and 58% of messages carry a fact;
- all 353 pronouns refer to someone named earlier in the same message.

Missing, and likely to change the result:
- the user's own life (job, health, plans, mood): there are zero facts about "me";
- the AI's own replies, half of a real journal; users answer them ("She got it!");
- roles changing hands (a new boss, an ex, a new partner): nothing changes in 22 months;
- long messages about several people, typos, dictation, sarcasm, second-hand news ("Kofi says Mira's moving");
- birthdays, plans and recommendations, which are what people most want remembered [E];
- natural questions such as "is Mira still in Porto?". The described-person (desc) questions read like logic puzzles.

Generator oddities [C]:
- 42 decoy facts (lures) such as "My sister's mother works as a pharmacist", which is the user's mum, are treated as facts about nobody;
- 148 mentions name a close person through someone else, for example grandma as "Ivan Lopes's flatmate".

3. FLAWS, RANKED

CRITICAL: the wrong-merge scorer marks correct rows as wrong.
- In "Shirin Teixeira's colleague can't have mustard", only the colleague is recorded as mentioned.
- Suppose a writer files every fact correctly and, in half of these messages, also writes the link it used ("Shirin | colleague | P45"). It scores 4.4% wrong merges, which is a STOP [C, probe2.py].
- Rows about "me" in 30% of event messages add another 2.6%.
- Fix: record Shirin as mentioned too, count merges on fact rows only, leave out a "me" person, and add both behaviours to the stub.

CRITICAL: the outcome map in section 1. The fix is in section 4.

HIGH: the phone's costs are tested the wrong way round.
- On a phone the writer can think overnight on the charger. The reader cannot: 700-1,500 thinking tokens take about 30-100 seconds per answer at 15-25 tokens per second [R/E].
- The test does the opposite: the reader thinks and the writer does not. C thinks at up to four steps, which is minutes per answer.
- Fixes:
  - decide the product verdict on B with thinking off;
  - add an arm "C-cues": it uses B's own clue call, runs one combined search per clue, keeps the top 60 and uses the same answer call. This is the equal-cost opponent the panel asked for;
  - pre-register a thinking-on writer on the frozen test world (about 3-4 L40S hours, about $8 [E]).

HIGH: the deletion check cannot fail by design.
- Rows are keyed by message number, the reader skips deleted messages, and the generator makes sure the value appears nowhere else. A "leak" can only be the model guessing, and a guess still triggers STOP.
- The real ways deletion fails are not tested:
  - the AI repeating the fact in its own reply;
  - a row filed under the wrong message number and so surviving the deletion. The audit of rows left behind is reported, not a line;
  - "forget my ex".
- The team's own forget_about would leave 48% of the mentions of a close person ("Mum says hi"), even with a perfect fact extractor [C, probe4.py].
- Fixes:
  - the line becomes: no row anywhere carries a deleted value, and no deleted value reaches any reader prompt;
  - add whole-person forgetting for the 10 close people and 10 same-name people. Target: 98% of the messages that truly mention them removed; report what else is removed with them.

HIGH: B's errors multiply. A described-person question needs 3-4 facts filed correctly. With 78% of facts filed, the three association kinds fall to 51% even with a perfect reader [C].
- Try a variant in the development phase: the writer only tags the people each message mentions (who, not what), and facts are matched on the message's own words.
- This also gives "forget Kofi", the hook for "use memory now" (which is otherwise untested), and a free measure of how well each kind of mention is resolved.

MEDIUM: the designer's advantage on questions.
- Each attribute has one question wording, the same in dev and test [C].
- The clue labels mirror the generator's question kinds, and C gets no such tuning.
- Fix: write held-out wordings for each kind before the freeze, and have the owner write 100 questions after the test world exists.

MEDIUM: the writing-cost line (1,000 tokens per message) cannot fail. The stub uses 6 tokens per message; a real writer will use about 30 [E].
- At 1,000 tokens, a 100-message day means 100K tokens, 1-2 hours of phone time. That is more than iOS or Android reliably allow in the background [R].
- Fix: at most 150 completion and 500 prompt tokens per message, plus phone-minutes per 100 messages.

LOW: events are a thin slice: one person, one activity, a past date. There are no plans, no group outings and no "how long since". They are still 26% of the hard set, so they move B's score.

LOW: dropping the A arm is right; test 1 settled it.

4. THE LINES I WOULD USE
- PASS (build B): as now, plus:
  - B with thinking off at least 80% on the hard set;
  - the margin measured against C-cues in the same mode;
  - the fixed merge scorer;
  - a clean deletion audit;
  - whole-person forgetting at 98%;
  - at most 150 tokens of writing per message.
  Keep wrong merges at 1%: a false memory costs more trust than "I don't know".
- SEARCH (build search), whatever B does: C-cues at least 80% on the hard set and within 3 points of B on association.
- WRITER-LIMITED (replaces most of STOP): oracle at least 85% with B below 0.9 × oracle, or wrong merges above 3%. Next is the thinking-on writer; the cloud only if that fails too.
- READ-LIMITED: oracle below 85%. Fix retrieval first.
- STOP only on a leak the design caused.

5. FORECAST, COMMITTED CODE [E]
- Wrong merges as scored: 3-6%, of which 1-2 points are the scorer bug.
- Facts filed: 70-85%.
- B, hard set: 58-74%, median 65%. With thinking off, 8-15 points lower.
- C, hard set: 45-65%, median 55%. B ahead of C on association by 5-20 points.
- Verdict: STOP 70%, BETWEEN 25%, PASS 4%, C 1%.

The likely story is "the index beats search, but a 4B without thinking cannot write it." As wired, that result would send the team to a cloud writer before they try the cheap fix.
