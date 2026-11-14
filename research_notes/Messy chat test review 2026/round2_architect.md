# Test 2 sign-off (round 2): architecture expert

**Final verdict: GO WITH LISTED FIXES.** The two bugs that would have decided the result are now fixed. Four items remain, all small code changes that do not need another review round. My scripts are in the session scratchpad r2/ (dir.py, cov.py, cov3.py, merge.py, rank.py, stubvia.py). No repository files were changed.

## My flaws, one by one

- **Critical 1, relation direction: FIXED** [C dir.py].
  - Dev: 0 of 180 two-hop questions and 0 of 197 "Kofi's flatmate"-type mentions point the wrong way.
  - Placeholder test world: 0 of 180 and 0 of 172. One apparent hit was my own script misreading "my best friend's cousin".
- **Critical 2, the writer's candidate list: PARTLY.**
  - The list is now reproducible: identical under two hash seeds [C cov.py].
  - Coverage did not improve. With perfect filing, the person behind "Kofi's flatmate" is on the list 110 of 197 times (55.8%), against 53–59% before.
  - All 87 misses already have the link row in the index. They are cut by `LINKED=25`, which ranks partners by count and then by lowest P-number (messy_index.py:240-255).
  - A variant that ranks first the partners of names followed by "'s" reaches 90.4% at the same list size of about 47 [C cov3.py].
  - The stub still sees identities it should not. `seen_new` holds everyone ever introduced, so 124 of 130 such fact rows are filed by bare full name and only 4 under Kofi [C stubvia.py]. The stub's "Kofi's flatmate 94.4%" therefore says nothing about the list.
- **Hub capture and showing more people: FIXED as code, ceiling remains.**
  - Described questions now show 5 people with 8 matching notes each, plus a busy-person penalty.
  - Truth index with the stub's clues [C rank.py]: for desc the right person is first 74.7%, among the 5 shown 90.7%, all needed messages shown 86.7%. For who: 79.3, 96.7 and 96.0%.
  - The residual comes from the truth index now holding the canonical wording only, which is the fair choice.
- **Link matching: FIXED.** The anchor's best link row within 0.05 is used, compared on the kind and the message (`link_targets`).
- **C's third search and its tokenizer: FIXED.** Each follow-up adds 15 new messages, up to 80. Word search now uses the BM25 tokenizer. A month in a query filters to that month.
- **The thinking writer: PARTLY.**
  - The arm exists (`--writer-think`).
  - "PASS with an overnight thinking writer" is not coded: write_summary only reports Bw (:1285).
  - The limit of 150 completion tokens per message would fail any thinking writer, whose 8K cap allows up to 400 per message. State which cost line applies to Bw.
- **Lines and scoring: FIXED.** Retractions now need an explicit "no longer", an empty answer is wrong, "my mum" is accepted, STOP needs a design leak, SEARCH is an outcome, and the margin bootstrap resamples people.

## New bugs the changes introduced

1. **HIGH: the wrong-merge count cannot see the two most likely messy errors.**
   - `truth_people` now includes Kofi (the anchor of "Kofi's flatmate") and the source of second-hand news. Fact rows are checked against that list.
   - Test [C merge.py]: I moved all 232 such fact rows onto Kofi or the source. Wrong merges stayed at 0.00%, and the hard-case rate stayed at 0.00%. Only the reported "filed right" measure fell (via 0.0%).
   - Fix: a fact row is wrong unless its index person stands for a subject of a fact in that message. Use the mention list only for "mentioned" rows.
2. **MEDIUM-HIGH: STOP on a row left behind can fire for the wrong reason.** `deletion_audit` checks by substring for any row under the person that carries the deleted value.
   - It fires on another person's true fact misfiled under that person, which is a coincidence and not a leak.
   - It also fires on a row the writer numbered with a neighbouring message, which is a real leak. I expect a 4B with thinking off to misnumber some rows [E].
   - Fix: count a row as left only when its cited message does not contain its value. In `parse_rows` (a change the development phase allows), re-attach or drop rows whose value is missing from their cited message.
   - STOP is checked first and hides everything else, so print the index-versus-search verdict next to it.

## The outcome map

The structure is right: STOP, PASS, SEARCH, READ-LIMITED, WRITER-LIMITED, BETWEEN, checked in that order. But WRITER-LIMITED leans on the merge count, which is blind until new bug 1 is fixed. B < 0.9 × the truth index catches only part of it.

## Updated forecast

| Measure | Forecast |
|---|---|
| Wrong merges, as now coded | 1–3% (biased low by new bug 1) |
| B, hard set | 72–84% |
| C, hard set | 58–72% |
| B − search on desc + who + two | +10 to +30 points |

Verdict probabilities: WRITER-LIMITED 35%, BETWEEN 25%, STOP 22% (mostly rows left after deletion), PASS 8%, READ-LIMITED 6%, SEARCH 4%.

## Fixes to make before the freeze

1. Count merges against the people a fact is about.
2. Rank possessive partners first in the candidate list, and make the stub writer use only the ids on its list.
3. Define a row left behind after deletion as a row whose value is not supported by its own message.
4. Code the Bw verdict and its cost line.
