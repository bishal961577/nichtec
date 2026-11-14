# Test 2 review: architecture expert

**Verdict: run with changes.** The design (B against a fair C, with an oracle and pages) can answer the question. As coded, though, two bugs build wrong answers into the questions themselves, and the stub hides the writer's weakest point. A STOP or BETWEEN result now would partly be measuring the code, not the 4B.

[C] = I checked it (file:line or my script's output, scripts in the session scratchpad a1–a9.py). [M] = from memory. No repository file was modified.

## Flaws, ranked

**CRITICAL 1: one-way relations are reversed** [C]. This covers mentor and landlord, plus godparent in the test world.
- **Two-hop questions** keep `way=="out"` (messy_world.py:713). "Out" means p *is* b's mentor, yet the question asks for "p's mentor".
  - Example: "What car does Agnes Pereira's landlord drive?" has gold answer Rosa Kiss. The note says "Rosa Kiss rents from Agnes Pereira", so Agnes is Rosa's landlord, not the other way round.
  - 23 of 180 two-hop questions in dev, 38 of 180 (21%) in the placeholder test world.
- **"Kofi's mentor" mentions** check `link_known[(b,rel,pid)]` (:426), which is the wrong direction. Example: "Hana Gomes's landlord" refers to my uncle, but my uncle rents from Hana. 41 mentions in dev.
- **Effect:**
  - B, the oracle and pages ignore direction when following links, so they "agree" with the wrong gold.
  - A careful reader (C, or any reader that thinks) gets marked wrong.
  - A writer that files these mentions correctly gets counted as a wrong merge.
- **Fix:**
  - Two-hop: keep `way=="in"` for one-way relations.
  - Mentions: test `link_known[(pid,rel,b)]`, and count only b's incoming links of that type.

**CRITICAL 2: the writer's candidate list misses the person behind "Kofi's flatmate"** [C]. I simulated perfect filing:

| How the person is mentioned | True person in the candidate list |
|---|---|
| Full name, "my X", nickname | 100% |
| First name alone | 97.4% |
| First name with context | 94.4% |
| Through another person ("Kofi's flatmate") | 53.0 / 56.4 / 58.9% |

- The three figures for "through another person" come from three runs with different `PYTHONHASHSEED`. Linked people are added only for `top[:10]`, in set order (messy_index.py:227), so the list is also not reproducible.
- The prompt forbids NEW for people already in the index. The 4B will therefore file these facts under Kofi, which is a wrong merge. Such mentions are 5.6% of the subjects of fact messages, so they alone can break the 1% line.
- The stub writes the true full names (:727-731), so it never exercises `candidates()`.
- **Fix:**
  - Add the link partners of every person whose name matched anywhere in the chunk, and sort them deterministically.
  - Make the stub writer use an id only when that person is in the list.

**HIGH**
- **The fact-only read side is biased towards close people (desc/who)** [C].
  - In the oracle index, the right person ranks first only 74.7% (desc) and 74.0% (who) of the time. In the top 3 it is 100% and 99.3%.
  - 51 of the 94 people ranked above the right one are close people: 10 of 260 people, with a median of 55 rows against 11 for others.
  - Cause: min over facts of the max over rows, then only 1–2 targets, each given 40 notes sorted by the "ask" clue (:466-476, :414-439).
  - Fix: show the top 4–5 people, each with only the rows that match the facts or the ask clue.
- **C's third follow-up search does nothing** [C]. Found notes go 20 → 40 → 60, and the third search's results are appended and then cut off at 60 (:547-550). Fix: a cap of 80, or each follow-up adds its top 10.
- **C's word-search re-ranking uses a different tokenizer** [C]: raw `split()` at :516, against the stemmed, punctuation-free, stop-word-free tokens in notebook_read.py:567. The measured effect is small (fname_ctx first-search recall 60.2 → 64.5%), but fix it.
- **Link following uses a fixed 0.75 threshold** on "kind: P12" embeddings (:484) [C]. Measured similarities:

| Link clue vs the writer's row | Similarity |
|---|---|
| neighbour / "lives next door to" | 0.641 |
| colleague / "works with" | 0.533 |
| bandmate / "in a band with" (held-out) | 0.723 |
| neighbour / "colleague" (a wrong link) | 0.666 |

  - Paraphrased kinds fall below the threshold, while a wrong link scores almost as high. The stub hides this because it writes the exact relation names.
  - Fix: take the best match among the anchor's 3–6 link rows, compared on the kind and the message text.
- **The writer runs with thinking off.** This repeats test 1's lesson [M]: thinking off took the 4B to 96% contradiction in Part B. STOP would then claim "the 4B cannot keep its own index". Fix: pre-register a writer with thinking on (one more sequential stream, about 2–3 h), or narrow STOP's wording to the thinking-off writer.

**MEDIUM**
- **Retractions are scored right for "not in my notes"** (:619) [C]. A reader that finds nothing beats one that found the old value but missed the retraction. Retracted and deleted questions are 214 of the 1,531 hard questions.
- **Close people must be named in full** [C]. 35 of 70 event_who questions are about close people, and "my mum" is scored wrong. Accept the role.
- **The wrong-merge line is diluted** [C]:
  - 39% of fact-message subjects are full names.
  - A bare first name shared by two people known at that moment occurs 2 times in 2,365.
  - Every pronoun (353 of 353) refers back within the same message.
  - Fix: add a line on the hard styles ("Kofi's flatmate", first name with context, same-name people).
- **Any answer naming a deleted value is a STOP** [C]. Sara Dimitrov's deleted hobby is archery, while live notes say "Sarai Quinn shoots arrows at the archery club". A near-name mix-up, or a 4B guess with sampling at temperature 1.0, would stop the whole test.
  - Fix: exclude deletions whose value a same-name or near-name partner holds, and STOP only on leaks the design causes.
- **The oracle's rows contain the question's exact alternative wording** (:345). This inflates the 0.9 × oracle line against B.
- **The development phase is uneven**: B's retrieval code may be tuned, C only its round count and prompts.

**LOW**
- The ≤ 1,000 completion tokens per message line is meaningless (the stub writes 6; prompt tokens dominate).
- 40 nonsense lures such as "My brother's brother…".
- `resolve()` turns a bare "my sister" or "Kofi's flatmate" into a new person. These count as splits, not merges, so they should be reported.
- "my X" names come only from the exact kinds "relation to me" or "relation" (:188).

## The world generator
- **The write side is easier than real chat** [C]: pronouns never cross messages, the bare ambiguous first name is almost absent, each message has one subject, values come from closed lists, and lures are marked by their wording. A PASS would overstate real filing. A STOP is informative.
- **C is disadvantaged by construction:** "when did I last see my mum" means finding the newest of about 20 events among about 250 messages, with no date tool. Events carry no line, so this does not touch the margin.
- **The questions' wording is only half different from the messages:** 74 of the 185 job and hobby descriptions appear word for word in a message. That is fine for fairness.
- **No truth leak** into B's write or read side. The only leak is the stub's identities, which is what hides critical flaw 2.

## Lines
1. Remove "≥ 0.9 × oracle" from PASS (report it), or build the oracle with canonical wording only.
2. Add a merge line on the hard mention styles (≤ 3%) beside the overall ≤ 1%.
3. Add "choose search" when C ≥ B on desc + who + two and C ≥ 85% hard, whatever B's other lines.
4. STOP-on-leak only for leaks the design causes.
5. Replace the write-cost line with total tokens per message.
6. Score retractions strictly.

## Forecast (after the fixes)

| Measure | Forecast |
|---|---|
| Wrong merges | median 2.5% (80% range 1.2–5%); about 4% before critical fix 2 |
| Facts filed | 78–90% |
| B, hard set | 70–82% |
| C, hard set | 55–70% |
| B − C on desc + who + two | +10 to +30 points |
| Verdict | PASS 8%, C 2%, BETWEEN 55%, STOP 35% |

The likely failures are desc below 75% and the merge line.

**Cost:** the estimate is plausible (about 1,860 tokens/s aggregate in test 1 [M]). A thinking-on writer adds about $3–5.
