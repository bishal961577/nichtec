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
