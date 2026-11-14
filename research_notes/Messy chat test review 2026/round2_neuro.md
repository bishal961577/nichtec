# Test 2 sign-off (round 2): senior neuroscientist

**Neuroscientist sign-off, round 2 (commit d36350b). Verdict: GO WITH LISTED FIXES.**

I checked each fix against the code. I re-ran the offline stub with its output redirected to my scratchpad, so no repository file was touched; it matches the report's section 14.9 (BETWEEN; B 92.2%, oracle 96.6%). Scripts are in the session scratchpad r2/. Everything marked [checked] comes from running them.

**My flaws, one by one**
- **C1, relation direction: FIXED.** [checked] One-way relations now point the right way: 0 of 30 two-hop questions and 0 of 31 "X's mentor" mentions are wrong.
- **C2, right answers scored wrong: FIXED, small leftovers.** [checked]
  - Now scored right: "Your mum."; "flute (previously bass guitar)"; "switched from X to Y"; "A cat."; "Bass."; "Dust.".
  - Still scored wrong, though right:
    - "The notes don't mention it." on deleted and never-stated questions, because the abstain list lacks "n't mention".
    - A who-answer that contains "or" anywhere ("sails on weekends or holidays"), because the hedge pattern catches a bare `or`.
    - The right name plus a ruled-out name.
    - "your old boss".
- **C3, wrong answers scored right, and the audit: FIXED.** [checked] An empty answer, a stale value plus "don't know" or "gave", and "old, previously current" are now scored wrong. "Current or old" counts as a hedge and is wrong. The audit now covers both arms, half right and half wrong answers.
- **H1, busy people winning described-person searches: FIXED.** [checked] On the truth index, a close person is wrongly ranked first in 2 of 300 questions (was 44).
  - The real ceiling is now paraphrase matching: the right person is among the 5 shown 90.7% (desc) and 96.7% (who), with every needed message found 86.7% and 96.0%.
  - The oracle's desc score is 86.7% even with a perfect index, so desc ≥ 75% leaves the writer only about 12 points of loss.
- **H2, the merge count: PARTLY.** The false positives are gone. But "Kofi's flatmate" now marks Kofi as mentioned, and so does a source ("Kofi says Mira…") (`messy_index.py:780, 790-792`).
  - So a fact row about the flatmate filed under Kofi is no longer counted as a merge. That is the 4B's most likely error, and the merge line can no longer see it.
  - Fix: judge each fact row against who the message's facts are about (`{f[0] for f in n["facts"]}`), not against everyone it mentions.
- **H3, a less polite world: PARTLY.** [checked]
  - Added: 123 pronouns in the next message (always the message right before), 2 role handovers (45 "old boss/flatmate" mentions), 107 second-hand messages, 150 messages about "me".
  - Not changed: a bare first name while the namesake is known still appears in only 6 fact messages (was 5). The core "which Mira" load is still untested.
- **H4, the writer's list of known people: NOT FIXED for "Kofi's flatmate".** [checked, simulated with a perfect writer]
  - The flatmate is on the list for only 77 of 141 such mentions (55%), even though the link row exists (140 of 141) and Kofi is always shown.
  - Cause: the partners of every name-matched person (a median of 51) are cut to 25 (`LINKED`), ranked by link count. Without the cut, 141 of 141.
  - Fix: put first the partners whose link word appears after "<name>'s" in the text, or raise the cap to about 60.
  - The stub hides this problem. Its writer looks up anyone it has already written as NEW by full name, even when that person is not on the list (`:903-904`). So "via filed right 94.4%" says nothing about the list.
- **H5, C's third search: FIXED.** Each follow-up adds 15 new messages, up to 80.
- **M1, the leak rule: FIXED.** [checked] STOP now requires a surviving row or a shown message. In the dev world, 0 cases where a co-mentioned person's misfiled value matches a deleted value.
  - Small leftover: `deletion_audit` would also count another person's misfiled message that happens to carry the same value.
- **M2, events: FIXED.** Weekdays are shown in the writer, reader and C prompts. Only 7 messages still imply a meeting, all January "having lunch this weekend" introductions.
- **M3, link following: FIXED.** [checked] Two-hop on the truth index: the right person is among those shown 100%, first 96.1%.
- **M4, wording: PARTLY.** The test world gets sealed wordings, which I could not see.

**New problems the changes introduced**
- The merge-count blind spot (H2) and the stub's NEW-name shortcut (H4).
- The forgetting line (≥ 98%) fails even for the near-perfect stub: 97.96%, worst person 88.5%. PASS now hinges on it.
  - Either count text matches of the person's names as well, or say plainly that this line alone will very likely decide PASS.
- Link precision scores correct "Kofi | link | flatmate" rows as wrong. The stub gets only 86.4% because truth pairs are taken only from the row's own message.
- If the writer records "new boss", a question about "my boss" finds only the old boss. This can be tuned in the dev phase.

**Updated forecast** (test world, as amended)
- B, hard set: 78% (72–84).
- Wrong merges: 1–2.5% as coded; 2–4% once counted against the facts' subjects.
- Hard-case merges: 4–12%.
- Verdict probabilities: WRITER-LIMITED 45, READ-LIMITED 25, BETWEEN 21, PASS 3, SEARCH 3, STOP 3.

**Final verdict: GO WITH LISTED FIXES.** These go in before the freeze:
1. Raise or re-rank the partner cap so "Kofi's flatmate" reaches the writer.
2. Count merges against the subjects of the message's facts.
3. Remove the stub's NEW-name shortcut.
4. Add "n't mention" to the abstain list and drop the bare "or" from the hedge check.

All four are small changes.
