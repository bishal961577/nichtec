**Architecture expert, round 2.** [C] = checked this session; [M] = from memory, not re-checked.

## 1. Where I agree and disagree
**Corrections I accept:**
- **Moderator, check 4:** on two-hop questions, the asker's page ranks first and names the middle person 99% of the time [C]. The middle person's own page is reached only by following that link.
- **Researcher:** the config stores the linear-layer state at float32 (`mamba_ssm_dtype`), so it is about 50 MB, not 25 MB [C].

**Where the others improved my view:**
- **Neuroscientist:** B should be an index of rows that are only ever added (person, slot, value, note id, position), each pointing back to a raw note. Rewritten profile pages are worse. Deleting a note then removes its rows, and no paraphrase survives. This is better than my version.
- **Researcher:** every road needs a step that turns chat into notes. So the real question is how much structure to add at write time, not A versus B. B must be judged against a fair search agent, not against A.
- **Founder:** `companion/companion_brain.py` already builds B with receipts and nightly rebuilt pages, and it has never been tested [C].

**What they got wrong:**
- **Founder:** "more than 500 thinking tokens kills A" would kill B as well. The 4B already writes a median 688 tokens on perfect pages [C]. That cost belongs to the model, not to road A. A's own extra cost is any thinking beyond that baseline plus the cache read for every generated token.
- **Neuroscientist:** an index of closed slots is the test generator's own 9-column table. It will look nearly perfect on this world by construction. Only a messy slice can decide anything: open-ended attributes, values in different words, pronouns.
- **Researcher:** running HippoRAG 2 locally plus a 15K-note notebook roughly doubles the cost for little decision value now.

## 2. Final vote
- **(i) Next test (65%):** an index built at write time against a fair search agent, on messy notes, with A as the reference. Then A at a larger size, then D, then E. My vote changed from "B pages vs A" to index rows, with C as the opponent.
- **(ii) Long-term phone memory (65%): D.** Its parts:
  - the raw journal as the source of truth;
  - the add-only index;
  - recent notes held in the cache;
  - pages rebuilt nightly, for display only.

  E is for style and skills only, never for facts that must be deletable. Ranking: D > B > C > A > E, unchanged in substance.

## 3. Line for the pending 4B whole-notebook run
Read on the hard set, with the automatic score.
- **X = 90%:** A is viable as the phone's working set up to about 60K tokens. Two conditions must also hold:
  - median output is at most twice the 4B's perfect-pages median, i.e. 1,400 tokens or fewer;
  - at most 2% of answers are cut off at the 16K cap.
- **Y = 75%:** A is out on the phone at 5,000 notes, and selection becomes the critical path.
- **Between the two, or at least 90% but over 1,400 tokens:** A only as a small window of recent notes.
- Report each kind of question separately, against Haiku's 100% (described people) and 98% ("who is").

## 4. Thinking-off arms
Do both. Keep the registered 64-token arm for integrity and for comparison with Haiku, and add a 256-token arm. The 64-token cap confuses "cut off" with "wrong": 514 of 995 Haiku answers were cut off [C]. These arms are cheap.

## 5. One merged next test
**Keep:**
- a messier world with 3–6 links per person, nicknames, pronouns, "my sister", same-name pairs inside one circle, and open-ended attributes;
- a slice paraphrased by a model, with every value checked automatically;
- at least 120 questions per hard kind;
- a deletion audit of 100 notes;
- wrong links scored against the generator's truth;
- an event slice (section 6).

**Arms, all on the 4B:**
- A: the whole notebook;
- C2: word search plus meaning search, thinking on, up to 4 rounds;
- B-index: rows written by the 4B, intersection of the question's cues, one link hop, then reading the raw notes;
- B-index without links;
- an oracle index built from the generator's truth.

**Drop:** HippoRAG 2, the 15K-note notebook (unless the pending score is at least 90%), rewritten profiles, and timing on a real phone (a later, separate test).

**Pass (all must hold):**
- B scores at least 85% on the hard set and at least 0.9 × the oracle index;
- described people, "who is" and two-hop each score at least 75%;
- B beats C2 by at least 5 points on those three kinds pooled, in a paired comparison;
- wrong links are at most 1%, and at most 3% on near-name pairs;
- no deleted fact leaks after deletion;
- writing costs at most 1,000 tokens per note.

**If C2 lands within 3 points of B:** drop the links and build plain search.

**Stop if:**
- B and C2 both score below 70%; or
- wrong links exceed 3%; or
- any leak is caused by the design.

Then a bigger model writes the index overnight.

**Cost:** about 6–8 L40S-hours, roughly $12–16 [M], plus 1–2 weeks of code. No paid API is needed.

## 6. What would change my mind, and the risk nobody has named
**What would change my mind:** C2 within 3 points of B on the messy set. I would then drop linking at write time, which is the owner's idea.

**Biggest risk nobody has named:** every test, past and proposed, asks about people's attributes. A life's memory is mostly events and time: "when did I last see Dad?", "what worried me last spring?". The timing questions are exactly the ones the budget cut (21 of the 28 LongMemEval questions not run [C]). A person/slot index has no place for events. We could perfect a people directory and ship a memory that cannot recall a day. The fix is to add 150 dated event notes and 100 "when" or "what happened" questions.

**Runner-up risk:** a malicious instruction stored in a note would stay inside road A's always-loaded context for good.
