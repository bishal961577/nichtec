# Memory road panel: which road the phone memory should take (November 2026)

**The question.** Test 1 found that Qwen3.5-4B, a phone-sized model, answers the hard notebook questions at 99.9% when it
is given the right pages. With word search plus one follow-up search it scores 50.2%. Which road should the phone memory
take, and what is the next test? The owner asked two more things:
- Is the work being done well?
- Does road A ("keep the whole notebook in the model's working memory") burn tokens on every chat?

The owner leaned towards road B, linking memories as they arrive: "When someone asks me 'you know Mira?', suddenly
everything about her comes to mind."

**How the panel ran.** The same four roles as the first panel (`reports/Expert panel 2026.md`): a machine-learning
architect, a curious founder, a senior neuroscientist and an AI research scientist.
- Each read the same brief (`research_notes/Memory road panel 2026/brief.md`) and could read the repository and search
  the literature.
- Round 1: each wrote alone.
- Round 2: each read the other three plus the moderator's checks of their claims (`moderator_checks.md`) and gave a
  final vote.
- All four are the same underlying model with different instructions. The researcher put it plainly: "four experts
  voting D is a consensus built on a shared prior, not evidence." The votes are not four independent votes.
- Full transcripts: `research_notes/Memory road panel 2026/` (`r1_*.md`, `r2_*.md`).

**The roads on the table:**
- **A:** the whole notebook kept in the model's working memory (its cache), never searched.
- **B:** link memories as they arrive (the owner's idea).
- **C:** better search when the question comes.
- **D:** a hybrid of these.
- **E:** put the memory into the model's weights.

---

## 1. The vote

| | Round 1, next test | Round 2, next test | Round 1, long-term | Round 2, long-term |
|---|---|---|---|---|
| Architect | B pages vs A (60%) | B as an index vs a fair C, A as reference (65%) | D (65%) | D (65%) |
| Founder | B, the 4B's own address book (70%) | the merged index test (75%) | D around B (60%) | D (65%) |
| Neuroscientist | B as an index plus a fair C (65%) | the merged index test (70%) | D (55%) | D (60%) |
| Researcher | fair bake-off, B vs C, A as reference (60%) | the merged index test (70%) | D (65%) | D (65%) |

**Next test: 4 of 4 for testing the owner's idea (B) next.** It must be built as an index of pointers back to the raw
notes, and it must face a fair search agent (C) in the same test.

**Long-term phone memory: 4 of 4 for D**, built from four parts:
1. the raw notes (the journal) stay the source of truth;
2. an index the 4B writes as each note arrives, made of rows that are only ever added (person, slot, value, note id,
   position) and point back to the notes;
3. a small window of recent notes kept in the model's cache (road A, kept small);
4. pages for people, rebuilt every night from the raw notes and used for display only.

**Weights (E) only for style and skills, never for facts that must be deletable** (architect; no one disagreed).

**Road A alone for 5–6 years: no one.** The researcher is about 90% sure it cannot work, for three reasons:
- the 262K-token limit;
- replies slow down as the notebook grows;
- deletion.

How the votes moved between rounds:
- **B became an index instead of rewritten pages.** All four adopted the neuroscientist's version: rows that are only ever
  added, each pointing to its source note. Deleting a note then deletes its rows, and no paraphrase of a deleted fact can
  survive on another page.
- **Confidence in testing B rose for three of the four.** The moderator's page check made the difference: filing notes
  under each person turned word search from failing into near-perfect on this notebook.

---

## 2. Is the work being done well? The panel's audit

**What is done well (all four):**
- the lines were registered before the results;
- amendments were written before any model call;
- the scorer was hand-audited;
- failed predictions stay on the record.

The researcher called it "better than most published memory papers".

**What was wrong.** Each item below was checked by the moderator against the code or the answers:

| Problem | Raised by | Effect |
|---|---|---|
| **search2 was a strawman.** Its "enough, or search again?" step ran with thinking off (64 tokens, greedy), as registered. The 4B said DONE on 117 of 120 described-person and 59 of 60 who-is questions, so it rarely searched. When it did search again, two-hop rose from 1% to 54% and first-name from 49% to 89%. | all four | The phone line fails as registered, but "search fails" overstates what was measured. The test measured word search behind an overconfident gate. |
| **"Accuracy flat with notebook size" compared different question sets.** On the same 635 name-keyed questions, Claude Haiku 5.5 scores 97.5 → 95.9 → 95.7 → 93.4% (pages → 100 → 1,000 → 5,000 notes). The hard kinds were never run at the middle sizes. | researcher | Section 9 of the test 1 report must be corrected: a small slide, not flat. |
| **Thinking-off scores were cut off.** With the 64-token cap, 514 of Claude's 995 answers at 5,000 notes were cut off. The cut answers scored 11.5% and the uncut ones 47.4%. | founder | "Thinking off at 5,000: 28.8%" mixes weak answering with cut-off answers. Fixed for the 4B by a separate 256-token arm (amendment 4). |
| **No thinking text was saved.** vLLM 0.33 names the field `reasoning`; the script read `reasoning_content`. | neuroscientist, researcher | Fixed during the run; every later answer has its thinking text. Claude's thinking text is not returned by design (amendment 2). |
| **LongMemEval.** Histories are a median of 180,879 Claude tokens, not ~115K, and the judge was Claude Haiku 5.5 grading its own model's answers. The budget ran out on 28 questions, 21 of them timing questions, so "full = evidence-only" rests on a biased subset. | researcher | Weakens Part C's "long context holds". |
| **The synthetic notebook cannot test road B.** Every note carries the full name, so filing is trivial; perfect pages hold only the question's two people. | architect, researcher, neuroscientist | 99.9% is a ceiling for reasoning, not for picking one person out of 840. |
| **Earlier mistakes** (from the record): the first test 1 design was too easy; the Claude cost was underestimated about 2×; a client bug made Part B with thinking off unreadable; the scorer marked 30 correct retractions wrong. | founder, researcher | Already disclosed in the test 1 report. |

**The moderator's own correction to what was told to the owner.** On first reading of the 4B result I wrote that "a 4B
model writing its own follow-up search doesn't rescue it" and that "word search can't do human-style recall, even with a
follow-up search". The first claim was wrong: the 4B mostly did not write a follow-up search. The second is too broad.
Word search over single notes fails, but word search over person pages does not (below).

---

## 3. The key new fact: filing, not searching, is the crux

The architect claimed in round 1, and the moderator re-checked with his own script, that the same notes grouped into
one page per person, searched with the same BM25 code, find the right person almost every time:

| Hard question kind | Word search over single notes (as registered) | The same word search over one page per person |
|---|---|---|
| Described people (n=120) | the needed notes are in the top 20 for 0% | right page ranked first 97.5%, in the top 3 100% |
| Who is (n=60) | – | first 100% |
| First name (n=60) | 45% | first 96.7%, in the top 3 100% |
| Two-hop (n=200) | 0.5% | the asker's page is first *and* names the hidden middle person 99.0%; the answer needs one link step |

This is the owner's association idea in its simplest form, and on this notebook it works. **The caveat all four
stressed:** it is easy here only because every note names its person in full. Real chat says "she", "my sister", "Mira"
(which Mira?), and those must be filed under the right person. **So the next test asks whether the 4B can file messy chat
correctly.**

---

## 4. The owner's two questions, answered

**"Does road A burn tokens on every chat?"** Partly. The cost is real, but it is not re-reading.
- **Re-reading is not the cost.** The notebook is read once and kept. On the server, the first read took 4.7 s; a new
  question on the same notebook took 0.55 s.
- **What does cost:**
  - **Memory.** The kept reading takes about 32 KB per token: about 2 GB for 5,000 notes, and about 16 GB at 16-bit for
    6 years of notes, beyond the model's 262K limit.
  - **Speed.** Every generated word reads the whole kept notebook. With thinking on (the 4B used a median of 709 tokens
    even on perfect pages, 2,605 with search2, 3,574 with bm25), the panel estimates 30 s to 2 minutes per memory answer on
    a phone. These are estimates, not measured on a phone.
  - **Deletion.** The model's fixed-size layers cannot subtract a note. Deleting one means re-reading everything after it.
- **The founder added:** thinking costs road B too. On perfect pages the 4B still thought for a median of 709 tokens. Every road
  is fast on a phone only if the 4B can answer *without* thinking. This run now measures that, with the 256-token
  thinking-off arms of amendment 4.

**"Is the association idea right?"** All four said the instinct is right. The closest published work:
- **A-MEM (2025)** is almost exactly the idea: links written as notes arrive, and old notes revised.
- **HippoRAG / HippoRAG 2** claim to model the hippocampal index. On multi-hop questions they are 2.8–4.2 F1 above a strong
  embedding retriever (researcher's check of the paper's table; the abstract's "7%" is a different measure).
- **Zep/Graphiti** keeps time intervals on facts.

**Published results are mixed.**
- On LoCoMo, full context (72.9) beat Mem0's graph memory (68.4), Zep (66.0) and A-Mem (48.4) (Mem0 paper).
- On MemoryAgentBench, HippoRAG-v2 (65.1) beat long-context GPT-4o (58.1), but Zep (37.5) and Mem0 (32.6) fell far below.
- The edge belongs to particular designs, not to structure as such.

**What would be new if this team did it:**
- a phone-sized model doing both the filing and the reading, measured against its own whole-notebook reading and a
  fair search agent;
- deletion that provably reaches everything derived from a deleted note.

**The neuroscientist's correction to the instinct.** "You know Mira?" is cued by her name, and name cues already worked
(99%). The failures were cued by *descriptions* ("my friend who teaches piano"). So the index must key on values as well
as names, and intersect them (Krakow + parrot leads to one person).

**The dangers all four named:**
- a wrong merge made at write time (Joao filed as Joaquin) becomes a permanent false memory;
- links can manufacture false memories unless every link records the note it came from;
- "everything comes to mind" floods the answer unless spreading is limited.

---

## 5. The line committed before the 4B whole-notebook result

The panel asked for this before the result exists. It was committed as amendment 4 of the test 1 report (commit
`94aed27`), before the 5,000-note arm and every 4B thinking-off arm ran. It is coded as `phone_whole_line`. The median of
the four proposals was taken:

| Outcome | Condition (hard set, thinking on, automatic score) |
|---|---|
| PASS: A is viable as the phone's working-set reader up to about 60K tokens | ≥ 90%, and described, who-is and two-hop each ≥ 75%, and median completion ≤ 2 × the 4B's perfect-pages median, and ≤ 2% cut off at 16K |
| READS BUT TOO SLOW: A only as a small recent window | accuracy conditions hold, but the length or cut-off condition fails |
| FAIL: A out as the phone's reader at 5,000 notes | ≤ 75% |
| BETWEEN: A only as a small recent window | anything else |

**Forecasts, recorded before the result:**
- neuroscientist about 80%;
- architect about 75%;
- founder about 75%;
- researcher about 72% (80% interval 50–90).

A pass holds at 16-bit on a server GPU only. 4-bit weights lose heavily on long inputs (Mekala et al., EMNLP 2025).

**Added at the same time, reported without lines:** thinking off with a 256-token cap, at perfect pages and at 5,000
notes. The founder's condition for putting A first in D is that the 4B with thinking off scores ≥ 90% on the whole
notebook. The 5,000-note arm here measures that.

---

## 6. The merged next test: "can the phone model keep its own index?"

The four proposals (the neuroscientist's index-then-read, the researcher's bake-off, the architect's Mira test and the
founder's address book) merged in round 2 into one test.

**Notebook, generated by code so every answer and every mention has a known truth:**
- about 300 people with 3–6 links each;
- nicknames, pronouns, "my sister", and the same first name inside one circle;
- open-ended attributes, plus held-out attribute and relation types that appear only in the test world;
- plain-restatement updates;
- 100 deletions;
- about 10 close people with 300+ notes each (the fan effect);
- about 150 dated event notes with about 100 "when" and "what happened" questions (architect, neuroscientist);
- at least 120 questions per hard kind.

**Arms (all Qwen3.5-4B):**

| Arm | What it is | Supported by |
|---|---|---|
| B-index | the 4B writes the add-only rows as notes arrive; code intersects the question's cues, follows at most one link and returns the raw notes | all four |
| Oracle index | built by code from the truth; separates filing errors from finding errors | all four |
| C | a fair search agent: BM25 plus meaning (embedding) search, thinking on, up to 3–4 rounds, same token budget | all four |
| A | the whole notebook, as the reference | all four |
| Perfect pages | the ceiling | researcher |
| B-index without links | isolates what the association adds | architect, researcher, neuroscientist (founder: drop) |
| B and C with thinking off (256-token cap) | phone speed | all four (founder: as the pass condition; the others: reported only) |
| Paraphrased chat slice | about 1,000 notes rewritten by a model, each value checked automatically | architect, researcher (neuroscientist, founder: drop) |

**Pass: choose B as the long-term road.** All of these must hold:
- B ≥ 85% on the hard set and ≥ 0.9 × the oracle index;
- described, who-is and two-hop each ≥ 75%;
- B beats C by ≥ 5 points on those three kinds pooled (paired);
- wrong merges ≤ 1% overall and ≤ 3% on near-name pairs;
- zero deleted facts come back;
- writing costs ≤ 1,000 tokens per note.

**If C comes within 3–5 points of B:** choose C, the simpler road (architect, researcher). The founder dissents: if C ties
but needs thinking while B does not, B still wins on phone cost.

**Stop.** Any one of these:
- B below 70%;
- wrong merges above 3%;
- a deletion leak caused by the design.

Then a bigger model (the 27B) writes the index overnight, or A is used as a window.

**Dropped for now (all four):** a HippoRAG 2 port (run it only if B passes, as the novelty check), a 15K-note notebook,
and free-text profile pages.

**Cost:** about 4–12 L40S-hours (about $10–25 of Modal credit) and 1–2 weeks of code. Scoring is by code, so no paid API
is needed.

**Guard against designer's advantage** (researcher): the code and prompts are frozen before the test notebook is generated
with a fresh seed, and slots are open-vocabulary.

**What would change each mind:**
- **Neuroscientist:** wrong merges above 3% on same-name pairs that a nightly rebuild cannot fix. Then linking at write
  time is unsafe; go to A plus search.
- **Architect:** C within 3 points of B. Then drop the linking.
- **Researcher:** C within 3 points of B, which he puts at about 40% likely (a plain file-search agent scored 74.0% on
  LoCoMo, ahead of Mem0's graph variant).
- **Founder:** the 4B with thinking off ≥ 90% on the whole notebook. Then A goes first in D for the first year or two.

---

## 7. Risks nobody had named before round 2

| Risk | Raised by |
|---|---|
| **Memory is mostly events and time, not people's attributes.** "When did I last see Dad?", "the restaurant Mira mentioned in Lisbon". A person/slot index has no place for events, and the timing questions are exactly the ones LongMemEval's budget cut. The index must key on time and episode too. | neuroscientist, architect |
| **Real chat never announces "use memory now".** "Her teacher cancelled again" must quietly bring up Mira's notes. That check runs on every message, and it is where both "it didn't notice" and "creepy injection" come from. | founder |
| **Designer's advantage.** Every pass so far was on a world the team wrote. The index's slots can quietly mirror the generator's 9 attributes. | researcher |
| **4-bit weights at long context hit road A hardest.** One 4-bit whole-notebook arm would cost under an hour. | researcher |
| **Phones ration background compute.** "Write the index at night on the charger" needs checking on a real phone (iOS defers background work). | founder |
| **A malicious instruction stored in a note** would sit inside road A's always-loaded context for good. | architect |

---

## 8. What this changes

- **The research target is sharper.** Reasoning over memories works at phone size (99.9% with the right pages). The open
  part is filing messy experience so that a vague cue reaches it, which is what human memory does with association.
- **The owner's idea becomes the next test, built carefully:**
  - pointers, not rewritten summaries;
  - indexed by description as well as by name;
  - every link traceable to its source note;
  - tested against a fair search opponent.
- **Nothing here touches continual learning.** It is the memory step only, and the panel did not claim otherwise.
