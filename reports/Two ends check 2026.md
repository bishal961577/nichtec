# The "two ends" check: can the phone model read without thinking, and write without the memory? (pre-registration)

Written and committed **before any model call** for this check. It is the first step of the plan all five members of the
breakthrough panel voted for (`reports/Breakthrough panel 2026.md`, section 6). The owner approved it on 16 November
2026 with a budget of about $3 of GPU time.

## The question

The panel's candidate design is **"describe, don't decide"**:
- the phone model describes each message without looking at its memory;
- code works out who everyone is and what is true now;
- the phone model answers from pages that code has already bound.

Before anything is built, two unknowns must be measured. The code in the middle is cheap and already simulated.

1. **Reading end.** With names, current values and links already worked out by code, can Qwen3.5-4B answer the hard
   questions **with thinking off**, in a few dozen tokens? If not, thinking is doing more than bookkeeping, and the
   phone-speed case dies for every design.
2. **Writing end.** Reading one message at a time, with only the previous message for context and no memory in view,
   can Qwen3.5-4B (thinking off) describe who is mentioned and attach each claim to the right mention?

## Data (development world only)

- Test 2's dev world (`brain/messy_world.py`, seed 101), first 1,500 messages, with the same 35 deletions as the dev run
  in `reports/Messy chat index test 2026.md` §16.7.
- **Reading end:** the 288 hard-set questions of that slice (the same questions as test 2's dev results).
- **Writing end:** 300 messages from that slice, drawn by a fixed rule with `random.Random(20261116)`:
  - all hard fact messages, up to 100 (hard = a fact message with a confusable name, a "through someone" or second-hand
    mention, or a pronoun pointing to the previous message);
  - 60 lure messages;
  - all definition messages up to 60 (introductions with a relation to me or a group, nicknames);
  - 40 relation messages;
  - the rest from other fact and event messages.
- **No test world, no sealed material, no training.**

## Arms

**Reading end.** The 4B, thinking off, greedy, at most 256 tokens. Answer prompt 4 (the current `ANSWER`). The pages
arm's messages (`pages()`), built four ways:

| Version | What the model sees |
|---|---|
| V0 | The messages as they are (the baseline; dev result 66.7% on the hard set) |
| V1 | V0 + **names**: each message carries the full names of everyone it mentions, with the words used ("Ana Varga, as 'Ana from choir'; Kofi Mensah, as 'he'") |
| V2 | V1 + a **"true now" header** for every person mentioned in the shown messages, built by code from all messages so far (not only the shown ones). It gives other names, how I know them, the current value of each attribute with the date it started, earlier values, things that stopped, and times with me. |
| V3 | V2 + **links**: every relation between people, in both directions ("Kofi Mensah's flatmate is Ana Varga") |

The header is built from the generator's truth. **So the reading end measures a ceiling:** what the phone model does
once the bookkeeping is perfect. It does not show that code can do the bookkeeping from real descriptions; that is the
writing end and step 2.

**Writing end.** The 4B, thinking off, greedy. Its output is forced into a fixed JSON form (structured outputs) with:
- **mentions:** the words used, the kind of mention, the name part, the person "through" whom someone is named, and
  the relation;
- **claims:** which mention the claim is about, a label, a value, and a status (true now / no longer / correction /
  plan or wish / tried or temporary / past / about me);
- **links** between mentions;
- **who told me**, for second-hand news.

Input: the message, its date and the previous message. Nothing from the memory.

## Lines

**Reading end, judged on V3** (panel §6):
- **PASS:** hard set ≥ 88%; each bookkeeping kind ≥ 75% (two-hop, retractions, roles, last meeting, nicknames);
  median answer ≤ 64 tokens.
- **STOP:** hard set < 80%.
- **Between:** reported as is.
- V1 and V2 are reported to show which part helps.

**Writing end**, scored by code against the generator's truth:

| Measure | Line |
|---|---|
| Claims attached to the right mention, easy fact messages | ≥ 98% |
| Claims attached to the right mention, hard fact messages | ≥ 93% |
| Relation direction (one-way relations: landlord, mentor, godparent) | ≥ 95% |
| Lures tagged (no "true now" claim carries the lure's value) | ≥ 90% |
| Definitions captured (relation to me, group, nickname: on the right mention with the right value) | ≥ 98% |
| Fact recall (each true fact's value appears in some claim) | reported; the neuroscientist's line was ≥ 90% |

- **STOP:** hard-message attachment < 85% **after a week of prompt work on dev.** This first run is the first
  measurement. Prompt changes are allowed on dev, and every version is logged here with its result.
- **How a claim is matched:** to a true fact by its value (the scorer's answer keys). Its mention is matched to a true
  mention by the words used. "Right mention" means the true mention of the fact's subject: for "Shirin's colleague is
  allergic to mustard", the mention "Shirin's colleague", not "Shirin".

**Not in this run:**
- the 27B writer arm, which needs a separate server;
- the owner's 300 labelled messages, which need the owner.

Both stay in the plan.

## Cost

- Reading end: 4 versions × 288 questions, thinking off. Writing end: 300 calls.
- Well under 1 L40S hour including start-up: about $1–3.

## Code

`brain/two_ends.py`. Results go to `results/two_ends/`.

---

## Results (16 November 2026)

All runs: Qwen3.5-4B on the project's Modal L40S, thinking off. Code and outputs: `brain/two_ends.py` and
`results/two_ends/` (`reading.jsonl`; `writing_v1.jsonl`, `writing_v2.jsonl`, `writing.jsonl` = v3; one summary per
version). GPU use was well under an hour, about $1–2 at the assumed rate. The exact bill is on the owner's Modal page.

### Reading end: BETWEEN (as registered), and a large cost result

| Version | Hard set | two-hop | retractions | roles | last meeting | nicknames | desc | who | Median answer tokens |
|---|---|---|---|---|---|---|---|---|---|
| V0 plain pages | 66.3% | 23.5 | 26.7 | 33.3 | 33.3 | 52.9 | 73.1 | 87.5 | 11 |
| V1 + names | 72.9% | 29.4 | 60.0 | 50.0 | 33.3 | 70.6 | 84.6 | 87.5 | 12 |
| V2 + "true now" header | 90.3% | 58.8 | 73.3 | 100 | 55.6 | 91.2 | 96.2 | 90.6 | 17 |
| **V3 + links** | **91.7%** | **94.1** | **80.0** | **100** | **44.4** | **85.3** | 88.5 | 90.6 | **18** |

**V3 against the lines.** The hard set (≥ 88) and the median tokens (≤ 64) are met. One kind misses its ≥ 75 line: last
meeting, 44.4% on only 9 questions. Its failures give the date a note was written instead of the meeting date shown in
the header.

**Verdict: BETWEEN.**

**What it means.**
- Once code has done the bookkeeping, the phone model answers 91.7% of the hard questions with thinking off, in a
  median of 18 tokens.
- The same model needed about 4,200 thinking tokens for 95.5% on plain pages (test 2 dev, §16.7). That is about
  1/200 of the tokens for 4 points less.
- Names alone help little (+6.6 points). The "true now" header does most of the work (+17.4). Links add two-hop
  questions (58.8% to 94.1%).
- **This is a ceiling.** The header was built from the generator's truth. Whether code can build it from real
  descriptions is the writing end, then step 2.
- Some V3 failures are answers that give the right value with its history ("she previously had a labrador"). Test 2's
  frozen scorer marks those wrong in every arm alike. It was not changed.

### Writing end: below its lines after one day of prompt work; above the stop line

Three describer versions on the same 300 messages, each logged:
- **v1:** the registered prompt.
- **v2:** an introduction is one mention; "Name from group" is one mention; someone named through another person is
  not that person; relations go in links; trips are temporary; a second made-up example.
- **v3:** relations as "who is_the X of whom"; never describe the previous message; every message about a person gets
  a claim; things said in passing count.

**A corrected scorer is reported beside the registered one.** Reading run 1's failures showed four scorer false
negatives:
- a definition value with an article ("a party" vs "party");
- a mention given without its group ("Rosalind" for "Rosalind from work") falling just below the word-overlap
  threshold;
- a claim attached to no mention counted as a lure leak, though code would drop it;
- a relation written in its natural direction ("rents from") read backwards.

The corrected scorer also accepts the generator's own surface words for each group ("a colleague of mine" for work).
Both scorers give 100% on descriptions built from the truth, and both catch all 19 anchor misfilings ("Uma's colleague"
filed under Uma).

| Measure (line) | v1 registered / corrected | v2 registered / corrected | **v3 registered / corrected** |
|---|---|---|---|
| Right mention, easy (≥ 98) | 87.5 / 100 (n 24) | 87.0 / 91.3 (n 23) | **96.2 / 96.2 (n 26)** |
| Right mention, hard (≥ 93; stop < 85 after a week) | 79.4 / 91.8 (n 97) | 92.0 / 92.0 (n 88) | **88.8 / 89.8 (n 98)** |
| Relation direction, one-way (≥ 95) | 40.0 / 85.7 (n 17) | 45.5 / 46.7 | **78.6 / 64.7** |
| Lures tagged (≥ 90) | 78.3 / 83.3 | 93.3 / 95.0 | **91.7 / 93.3 ✓** |
| Definitions (≥ 98) | 55.4 / 78.5 | 52.3 / 72.3 | **70.8 / 92.3** |
| Fact recall (reported) | 93.8 | 86.0 | **96.1** |
| Tokens per message, out / in | 178 / 924 | 165 / 1,324 | 168 / 1,453 |

**What it means.**
- **The best version (v3) recalls 96% of facts.** It puts about 89–90% of claims in hard messages on the right
  mention, against B's writer at 78.2% of facts filed right (different measure, same direction). It tags 93% of lures
  and captures 92% of definitions (corrected).
- **Relation direction is the weak spot.** On 17 one-way relations, "Maya rents from Nuno" often came out as "Maya is
  Nuno's landlord".
- **Definitions are below their line.** Both matter more than their counts suggest: one misread definition or relation
  misfiles many later facts (the architect's fan-out finding).
- **The writing end has not reached its stop line** (hard attachment 88.8–89.8%, against the stop at < 85 after a
  week). The registered week of prompt work is not used up. The three versions moved the numbers by several points in
  both directions, so a 4B prompted this way is noisy to steer.
- **Small samples:** 26 easy facts, 17 relations and 9 last-meeting questions. Each could move by 10 points with a
  different sample.

### Not done in this run

- The 27B writer arm. It settles whether the root is the 4B's size or the procedure, and needs a separate server.
- The owner's 300 labelled messages.
- Step 2, the code resolver run on the describer's real output: the measure that decides wrong-person filing end to
  end.

### Next, in order (for the owner to decide)

1. **Step 2 on v3's descriptions** (CPU only): run them through a resolver, then measure wrong-person filing,
   "Kofi's flatmate" filing, forgetting and exact deletion against B's 6.27%, 11.1% and 62.2%.
2. **The describer's two weak spots,** with the rest of the prompt-work week, or a small LoRA trained only on
   generated lives:
   - relation direction;
   - definitions.
3. **The 27B arm, and the owner's 300 messages.**
