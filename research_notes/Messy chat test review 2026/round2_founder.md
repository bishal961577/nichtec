# Test 2 sign-off (round 2): curious founder

FOUNDER, ROUND 2 (sign-off on d36350b). VERDICT: GO WITH LISTED FIXES.

I checked the code and re-ran my probes in the session scratchpad (r2_probe.py, r2_probe2.py). I changed no repository file and did not open the sealed folder.

**My flaws, one by one**

1. **The merge scorer marking correct rows wrong: FIXED [C].** A perfect writer that also writes the link rows it used now scores 0.00%. Rows about "me" are excluded.

2. **The outcome map: PARTLY [C].** SEARCH can now win on its own, STOP covers deletion only, and READ-LIMITED and WRITER-LIMITED each name a next step. Three gaps remain:
   - SEARCH has no thinking-off condition, but PASS needs B with thinking off ≥ 80%. The search agent could "win" while still needing up to four thinking calls per answer.
   - "PASS with an overnight thinking writer" is written in 14.3 but not in the code. The thinking writer (Bw) is only reported. Its writing would also break the 150-token cap.
   - If the fast path is the only thing that fails, the verdict is BETWEEN, with no next step.

3. **Phone cost: mostly FIXED [C].**
   - PASS needs B with thinking off ≥ 80%.
   - Ccues is the equal-cost search opponent I asked for: B's own clue call, one search per clue, 60 messages, one answer call.
   - The writing caps are 150 completion and 500 prompt tokens per message.
   - The thinking writer runs only behind a flag (point 2).

4. **Deletion line: FIXED in definition [C].** STOP now fires only on rows left behind or a deleted value shown to a reader. It now trips too easily: see new problem B.

5. **Forgetting a whole person: FIXED, with one weakness [C].** With perfect filing and "mentioned" tags it removes 100%. Without the tags it removes 38% (worst person 19%), so the test does catch the gap. But the line is an average over 20 people: the stub passes at 98.0% while its worst person keeps 11.5% of their messages. Make it per person (every person ≥ 95%).

6. **Errors multiplying / file who, not what: FIXED [C].** The writer tags every person a message mentions, and filing is reported by how each person was named.

7. **Designer's advantage: FIXED, better than I asked.** Sealed wordings written by someone else, plus a seed from a public randomness beacon. The owner's own questions are optional.

8. **Realism: PARTLY [C, 40 samples].** Now present:
   - second-hand news ("Gideon Molnar says Edda adopted a tortoise"), 102 messages;
   - pronouns that refer back to the previous message, 123;
   - "my old flatmate", 45 mentions;
   - facts about the user, 150 messages.

   Still missing: messages with more than one fact, the AI's own replies, typos, plans. 14.7 lists these honestly. The kinship lures and close people named through strangers are gone.

**New problems the changes caused**

- **A. The merge line cannot see the two hardest cases [C].** Take a writer that files every "Kofi's flatmate has a ferret" under Kofi, and every "Kofi says Mira moved" under Kofi. It scores 0.00% wrong merges and 0.00% on the hard subset. These cover 243 of 2,112 facts, and every one is a false memory. Only the "filed right by style" table shows them (0.0% for "Kofi's flatmate" mentions).
  - Fix: in `write_quality`, someone named only as the link anchor ("Kofi" in "Kofi's flatmate") or as the source of second-hand news cannot own a fact row.

- **B. One mis-numbered row now stops the whole test [C].** I simulated a writer that cites the neighbouring message's number for some fact rows. The row then survives deletion, and STOP fires:

  | Rows cited to the wrong message | Seeds that hit STOP |
  |---|---|
  | 0.25% | 2 of 10 |
  | 1% | 4 of 10 |
  | 2% | 10 of 10 |

  In the dev world, 7 of the 100 deleted messages are like "She adopted a ferret." where the name is in the previous message. That is the likeliest mis-citation. STOP would then hide the verdict that actually tells the team what to build.
  - Fix: in `parse_rows` (an allowed change), move a row to the message in its chunk whose words match its value. Also print the verdict that would apply without the STOP clause.

**Updated forecast (code as committed) [E]**
- STOP 35%, almost all from problem B;
- WRITER-LIMITED 43%;
- READ-LIMITED 7%;
- BETWEEN 8%;
- PASS 4%;
- SEARCH 3%.

With fixes A and B: WRITER-LIMITED 60%, BETWEEN 15%, READ-LIMITED 8%, PASS 7%, SEARCH 5%, STOP 5%. The likely result then reads "the index beats search; next, the thinking writer", which is the step the team should take.

**Fixes needed before the freeze** (all code, no model calls):
1. Problem A: the anchor or source of a message cannot own a fact row.
2. Problem B: move mis-cited rows to the right message, and report the verdict that would apply without STOP.
3. SEARCH also needs the search arm with thinking off ≥ 80%.
4. Write the Bw verdict into the code, with its own writing cap (about 400 completion tokens per message, reported as phone-minutes per night). Run the thinking writer automatically on WRITER-LIMITED or BETWEEN.
5. The forgetting line applies to every person, not the average.
