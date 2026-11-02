# Notebook reading at shipping size (November 2026)

Test 1 of the expert panel's plan (`reports/Expert panel 2026.md`, section 5). This file was written and committed
before any model call. The design, pass lines and stop lines below are fixed; the result goes in section 6 and nothing
above it changes after the run.

---

## 1. Why this test

The team's case that "a notebook plus the model reading it" is not enough rests on three measurements, and the
panel showed each was taken on a model too small or with thinking switched off:

| Earlier conclusion | Measured on |
|---|---|
| Pasting facts into the prompt collapses: 10 / 100 / 1,000 facts gave 43 / 27 / 13% | Qwen2.5-0.5B |
| Stored facts combine poorly: multi-hop 9.4%, use ratio 0.40 | GPT-J-6B (its own multi-hop rate is 23.4%) |
| (b) "LLMs can't check their own conclusions": 37% (27B) and 96% (4B) of guesses contradicted the evidence shown | Qwen3.5 with thinking off, sampled at temperature 0.8 |

The app would ship Qwen3.5-27B in the cloud and Qwen3.5-4B on the phone, both able to think. If the 27B model reads a
5,000-note notebook from its prompt well enough, version 1 of the companion's memory is "keep the notebook, put it in
the prompt" (or select pages from it). That then becomes the bar every memory method has to beat, including the
keyed memory and anything the long-shot tests produce. If it doesn't, the test shows where reading breaks: number of
notes, rewording, combining two notes, updates, or belief conflict.

## 2. What is already known

- **Rewording hurts long-context reading.** NoLiMa (arXiv 2502.05167): at 32K tokens, 10 of 12 models fall below half
  their short-context score when the question shares no words with the fact; GPT-4o goes from 99.3 to 69.7.
- **Long-context chat memory loses 30–60% relative to being handed only the evidence** (LongMemEval-S, about 115K
  tokens; arXiv 2410.10813).
- **No published number exists for a 27B-class model reading 3–10K personal facts.** This is the panel researcher's
  search result; the moderator did not repeat the search.
- **Panel predictions, recorded as given.**
  - Architect: fine for direct recall when the question shares words with the note; falls off when it shares none,
    or when two steps are needed.
  - Researcher: lookups sharing words with the fact hold; reworded questions, combining facts and latest-value
    updates degrade.
- **The moderator makes no forecast.** This project's record is 0 for 5.

---

## 3. Design

### 3.1 Models and serving

- **Models:** Qwen3.5-27B-FP8 (the cloud model) and Qwen3.5-4B (the phone-sized model), each behind its own Modal
  endpoint serving the OpenAI-compatible chat API (SGLang).
- **Context:** the endpoints must accept at least 65,536 tokens. The 5,000-note pile is 195,077 characters, about
  50K tokens, plus up to 6,144 tokens of thinking.
- **If the largest pile does not fit:** the run waits for an endpoint that takes it. There is no fallback to a
  smaller top level.
- **Thinking on:**
  - `enable_thinking: true`, temperature 0.6, top_p 0.95, top_k 20 (Qwen's thinking-mode sampling);
  - up to 6,144 tokens;
  - the answer is what follows `</think>`;
  - a reply cut off at the cap counts as wrong, and the share cut off is reported.
- **Thinking off:** `enable_thinking: false`, greedy (temperature 0), up to 48 tokens.
- **Seeds:** each request carries a fixed seed derived from its condition and question.
- **Server non-determinism:** this project has measured that identical requests can differ on these servers. One
  answer per question per condition; no reruns.

### 3.2 Part A: personal notebook (`brain/notebook_read.py`)

**The world.** Built with seed 0, fully determined by the script.
- **People:** 1,080 fictional people in 540 groups. Names come from lists filtered so that no name is a prefix of
  another, and no name word is an attribute value or relation word.
- **Each group** is:
  - person A with 4 facts;
  - one note giving A's relation to person B;
  - person B with 4 facts.
- **Facts** come from 9 attributes, each with a closed list of 16–30 values: job, city, pet, hobby, favourite food,
  instrument, car, language being learned, allergy.
- **Relations** are 10 gender-neutral ones: neighbour, best friend, manager, cousin, flatmate, landlord, colleague,
  mentor, business partner, personal trainer.
- **Updates:** 135 groups carry a later note that changes A's city ("has moved to") or job ("has changed careers and
  now works as").
- **Size:** 4,995 notes in all.

**Piles.** Every question is asked inside a pile of notes. The pile is introduced as a user's notebook "in the order
they were written; later notes may update earlier ones".

| Level | Groups per pile | Notes per pile | Piles |
|---|---|---|---|
| 10 | 1 (the question's own group: the "selected pages" condition) | 9–10 | 540 |
| 100 | 11–12 | 99–111 | 49 |
| 1,000 | 108 | 993–1,007 | 5 |
| 5,000 | all 540 | 4,995 | 1 |

- **Order:** notes are shuffled within a pile, with each update placed at a random point after the note it changes.
- **Same questions at every level:** each question is asked inside the pile that holds its group, so levels are
  compared on identical questions.

**Questions** (fixed by seed 2):

| Kind | n | What it asks |
|---|---|---|
| reword | 150 | One fact of A or B, asked with no content word in common with its note. The script asserts this: the 4-letter stems of the question and note template share nothing outside a stopword list. Example: note "Paula Kaplan drives a Nissan.", question "Which vehicle brand does Paula Kaplan own?" |
| same | 100 | The first 100 of those facts, asked in the note's own words ("What car does Paula Kaplan drive?"). Paired with reword. |
| two | 120 | One of B's facts through A: "What pet does Signe Larsen's neighbour have?" Needs the relation note and B's fact note; A's own value for that attribute, if A has one, is in the pile as a distractor. |
| upd | 60 | The latest value of an updated fact: "What does Matteo Lopes work as now?" |

### 3.3 Part A2: conflicting world edits

- **Notes:** MQuAKE-CF-3k-v2 (sha256 `f82091cb…ddc71e`, checked by the script), all 2,764 unique edits written as
  notes ("Hans Delbrück is employed by Pasteur Institute."), introduced as "changes in the world… treat them as true,
  even where they differ from what you know".
- **Questions** (seed 3):
  - 150 single-hop questions on edited facts (MQuAKE's own question for each edit);
  - 120 two-hop cases (first question of each), each with all its edits in the pile.
- **Exclusions:** questions whose new and old answers share an alias are excluded.
- **Piles:**
  - 10, 100 and 1,000 notes: the questions' needed edits plus random other edits;
  - "all": all 2,764 edits.
- **Own facts:** the same 120 two-hop questions asked with no notes, scored against the original answer. This is the
  model's own two-hop rate, the denominator the GPT-J use ratio used.

### 3.4 Part B: the scientist agent with thinking on

`brain/scientist.py --thinking` reruns the scientist agent's random1 streams (12 tasks, 8 rounds, 4 proposals per
round), with:
- the same seeds as the original run;
- thinking on, with the sampling above, up to 6,144 tokens;
- the rule taken from the text after `</think>`.

It reports the share of guesses that contradict the cases shown, against the original random1 streams: 35.7% of 314
guesses on 27B, 96.5% of 288 on 4B.

### 3.5 Scoring

- **Normalization:** answers are lowercased, accents and punctuation stripped, and matched as whole words.
- **Part A:** an answer is right if it contains the gold value and no other value from that attribute's list. So
  listing two candidates is wrong.
- **Update questions:** an answer containing the old value and not the new one is also counted as **stale**.
- **Part A2:** right if it contains the new answer or one of its aliases and not the old answer or its aliases. An
  answer with the old answer and not the new is counted as **old**.
- **Own facts:** right if it contains the original answer and not the new one.

### 3.6 Run order and budget

Each endpoint runs out of a two-hour window, so conditions run in this fixed order and the primary conditions finish
first:

1. Personal, thinking on: 5,000, then 10, then 1,000, then 100 notes.
2. Personal, thinking off: 5,000, 10, 1,000, 100.
3. World, thinking on: all, 10, 1,000, 100, then own facts.
4. World, thinking off: all, 10, 1,000, 100, then own facts.
5. Part B.

Further details:
- **Large piles:** at 1,000 notes and above, one unscored call first puts the pile's prefix in the server's cache;
  then its questions run.
- **Resumability:** answers are appended to `results/notebook_read_<model>/answers.jsonl` as they arrive, and a
  restart skips what is done.
- **Preflight:** a call to `--preflight` checks first that the 5,000-note pile fits, whether its prefix is cached, and
  that thinking-mode answers can be read. These calls are not scored.
- **If the window closes:** whatever conditions have finished are reported, and unfinished ones are listed as not
  run.

### 3.7 Pass and stop lines

**The primary lines decide version 1 of the app.** They are read on Qwen3.5-27B, personal notebook, thinking on.
Decisions use point estimates; 95% bootstrap intervals over questions (10,000 resamples) are reported.

| Measure | Pass | Stop |
|---|---|---|
| Reworded single-hop accuracy at 5,000 notes | ≥ 85% | < 70% |
| Two-hop ratio: accuracy at 5,000 notes / accuracy at 10 notes, same 120 questions | ≥ 0.8 | < 0.5 |

- **If two-hop accuracy at 10 notes is below 50%,** the ratio is reported as not measurable and only the first line
  decides.
- **PASS** = both lines pass. The whole notebook in the prompt works at 5,000 notes; it is version 1, and the bar
  every memory method has to beat.
- **STOP** = either line stops. Reading the whole notebook from the prompt is not version 1 at 5,000 notes; pages
  must be selected first.
  - If the 10-note condition (selected pages) is high, selection plus reading is the candidate. The keyed memory's
    lookup, or a retriever, is the selector.
- **IN BETWEEN** = otherwise. Report where it breaks, and decide with the owner.

**The same lines are applied to Qwen3.5-4B** and reported as the phone-sized reading. They do not decide version 1,
which would run the 27B model in the cloud.

**Part B lines** apply to each model separately:

| Outcome | Condition |
|---|---|
| WITHDRAWN | ≤ 10% of guesses contradict the shown cases. Conclusion (b) does not hold in thinking mode. |
| STANDS | ≥ 30%. Conclusion (b) holds in thinking mode too. |
| WEAKENED | Between 10% and 30%. |
| UNREADABLE | Fewer than 50% of proposals give a readable rule; no judgement on (b). |

**Reported without lines:**
- same-wording vs reworded, paired;
- thinking on vs off;
- stale answers after updates;
- conflicting edits: new and old answers by pile size;
- the conflict use ratio (two-hop new-answer accuracy / own-fact two-hop accuracy);
- truncation and token use.

### 3.8 Where this test departs from the panel's wording

1. **The two-hop ratio's denominator.** The panel wrote "two-hop ≥ 0.8× its own-fact rate". Fictional people have
   no own-fact rate, so the personal denominator is the same questions with only the group's own notes (10 notes).
   The pile is then the only difference. The own-fact rate is measured on the world edits instead (Part A2).
2. **No separate top-k retrieval arm.** The 10-note level is perfect page selection, an upper bound for any selector.
   A real retriever's misses are not measured here.
3. **Storing facts under the model's own wordings** (neuroscientist) concerns the keyed memory, not prompt reading.
   It is not part of this test.

### 3.9 Limits fixed in advance

- **Notes are clean template sentences with full names.** Real chat has pronouns ("she"), nicknames, "my boss",
  corrections and small talk. A pass here means reading works on clean notes, not that the app's lookup problem is
  solved. A fail is not softened by this: clean notes are the easy case.
- **One model family (Qwen3.5), one answer per question, server non-determinism.**
- **The world-edit arm instructs the model to accept the notes,** as MeLLo-style prompts do. Without that instruction
  conflict would likely be larger.

### 3.10 Cost

- About 430 personal questions × 4 levels × 2 modes, plus 390 world calls × 4 levels × 2 modes, plus Part B (384
  proposals) per model.
- On the endpoints' dedicated H100s, expected to fit within one two-hour window per model.
- Throughput at 50K-token prompts is unknown until the preflight, and the run order protects the primary lines.
- No API credit.

---

## 4. Commands

```
python brain/notebook_read.py --stub                                       # done before commit: bookkeeping only
python brain/notebook_read.py --base-url URL27 --model Qwen/Qwen3.5-27B-FP8 --preflight
python brain/notebook_read.py --base-url URL27 --model Qwen/Qwen3.5-27B-FP8
python brain/notebook_read.py --base-url URL4  --model Qwen/Qwen3.5-4B
python brain/scientist.py     --base-url URL27 --model Qwen/Qwen3.5-27B-FP8 --thinking
python brain/scientist.py     --base-url URL4  --model Qwen/Qwen3.5-4B --thinking
```

## 5. Stub check before commit

The stub reader is right about 97% of the time minus 4 points per tenfold growth of the pile. It ran all 18
conditions end to end and produced the tables and a verdict. It confirmed:
- 4,995 notes; pile sizes 9–10, 99–111, 993–1,007 and 4,995;
- 430 personal and 270 world questions;
- 2,764 world edits, none clashing;
- Part B produced its table and verdict.

The stub numbers mean nothing about any model.

## 6. Result

(Not yet run.)
