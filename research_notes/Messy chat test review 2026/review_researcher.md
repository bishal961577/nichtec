# Test 2 review: AI research scientist

### 1. Verdict: run with changes, not as is
Two flaws on their own are enough to block the run. First, one relation family has inverted gold answers. Second, the guard against designer's advantage protects the wrong layer. Most fixes are cheap, and section 8 already requires them before the freeze.

### 2. Flaws, ranked (scripts are in the session scratchpad)

**CRITICAL 1. Asymmetric relations are inverted [checked, a1.py].**
- For mentor, landlord and godparent, the two-hop questions ask about "p's mentor" but use p's *mentee* as gold (messy_world.py:713). Example: "What car does Agnes Pereira's landlord drive?" has gold Rosa Kiss, but the note says "Rosa Kiss rents from Agnes Pereira." Agnes is the landlord, not the tenant.
- Size: 23 of 180 two-hop questions in dev, and 23–39 of 180 across 13 test seeds (a6.py). That is 5–8% of the 480-question pool B − C is judged on, which is bigger than the 5-point margin.
- The gold rewards readers that ignore direction. A careful reader is marked wrong.
- The same bug affects "X's mentor" mentions (line 426 checks `link_known[(b,rel,pid)]`). There are 41 such mentions in dev, so a writer that files them correctly gets charged wrong merges against a 1% line.
- **Fix:** at line 713 skip `way=="out"` for asymmetric relations; at line 426 use `(pid,rel,b)`.

**CRITICAL 2. The held-out guard covers vocabulary only.**
- Held-out types change slot values only. Every question, mention, lure, retraction and event template is the same in dev and test. B's read side is coded to those templates: `" from "` at messy_index.py:385, the event_who branch at :457, and the dev fix that took event_who from 38% to 99%.
- The writing prompt (lines 83–86) quotes the generator's lure and retraction templates almost word for word ("is thinking about getting a ferret", "her mother is a nurse", "gave the pet away").
- C gets no comparable engineering, so B − C mixes "keeps an index" with "has code tuned to the templates".
- The mess is also shallow [checked, a2.py]. All 353 pronouns have their antecedent in the same message. 46% of fact subjects are full names.
- **Fix:**
  - Have someone other than the reader's author write new templates for each question kind and mention style. Commit their hash before the dev phase and reveal them at the freeze.
  - Add a no-LLM "code index" arm (a regex filer) to show how much of B's result is just parsing templates.
  - Give C structured tools (a date filter, name grep) or the same development budget.

**HIGH 3. The scorer has both false negatives and false positives [checked, a3.py].**
- False negatives (scored wrong, actually right):
  - short or singular forms: "egg", "Peanut allergy", "The drum kit", "A cat/dog/snake", "fire fighter", "Dust", "She paints";
  - an old value or lure value mentioned as context on any non-update kind ("labrador (before that, iguana)"), because history is only allowed for update questions (line 613);
  - key collisions: "running late" matches marathon running, "garage" matches mechanic;
  - the date format "28/07/2026";
  - activity wordings "walk up the ridge trail", "the lido", "Games night";
  - "Your mum." on who-did-I questions. 35 of the 70 event_who golds are close people. This one hits C harder, because B's reader pulls in the naming rows automatically (line 465).
- False positives (scored right, actually wrong):
  - updates: "bass guitar, previously flute" (old value presented as current) and "flute or bass guitar";
  - who: "Pia Mensah, or maybe Mira or Kofi" and "Could be Pia";
  - event_last: any list of dates that includes the right one;
  - retractions: "Still Tesla – she sold her old one";
  - ambiguity: any answer containing "which";
  - deleted: an empty, cut-off answer.
- An answerer that always says "not in my notes" gets 214 of 1,531 hard questions (14.0%) for free.

**HIGH 4. The deletion line can only fail by chance.**
- B's reader sees only raw messages that were not deleted (line 502). A design leak is therefore impossible, and "any answer names a deleted value means STOP" (line 953) can only be tripped by hallucination or a misfiled row hitting a 1-in-10 to 1-in-24 value.
- **Fix:** STOP only when the deleted value is in the reader's prompt or `rows_left>0`. Add a "never stated" control kind to measure the chance rate.

**HIGH 5. The decision logic is lopsided.**
- The C verdict ("choose search") requires B to pass every absolute bar. If B fails and C scores 90%, the verdict is STOP, which points to a bigger writer and ignores C. Add an absolute line for C.
- The writing-cost line cannot fail: the cap is 1,500 tokens per 20 messages, so at most 75 tokens per message (lines 45 and 48).
- The hand audit reads only B's wrong answers. It can only raise B, it never looks at C, and it cannot find false positives. Make it blind, cover both arms, and sample answers scored right as well as wrong.

**HIGH 6. The seed protocol cannot be enforced.**
- Whoever makes the freeze commit can retry commit hashes. A world builds in about 4 seconds and the stub screens it offline. Worlds do differ: asymmetric two-hop questions range 23–39 and the hard set 1,408–1,569.
- **Fix:** seed = hash(freeze commit + a drand or NIST beacon round published after the push).

**MEDIUM**
- **Statistics:**
  - There is one greedy index, and the 480 pooled questions fall on 223 people (up to 5 each). One merge of a close person spreads to many questions. Use a bootstrap clustered by person.
  - Rough power: the standard error of B − C is about 2.7 points, so the margin is decidable only if the true difference is at least 10 or at most 0.
  - A single kind at n = 150 has a standard error of about 3.5 points, so three "each ≥ 75%" conditions are largely noise near 78%.
- **write_quality():**
  - The denominator is diluted. Adding one row per small-talk message drops an adversarial writer from 10.4% to 7.4% wrong merges (a4.py).
  - Link targets are never checked (line 661): a link to the wrong Mira, the panel's main named danger, is invisible.
  - "Facts filed" ignores the kind and matches substrings.
  - **Fix:** score each mention on messages that state a fact, by mention style, and count link-target errors.
- **Kin lures contradict the truth:** there are 40 lures such as "My sister's mother works as a pharmacist" and "My brother's brother is allergic to latex", which are statements about mum or the user.
- **Ambiguous event gold:** 11 of 170 event_last questions have a later "picked me up / ran into / saw … earlier" message, and 18 event_what questions have a second meeting in the same month.
- **C handicaps:**
  - The third follow-up search is wasted once 60 messages are found (line 550).
  - The BM25 re-scoring uses raw `.split()` tokens (line 516). The effect is small: 44.7% → 45.1% of first searches find every needed message (a5.py).
- **Held-out reporting:** in the test world, 133 who/desc questions use held-out descriptors but only 30 are flagged.

**LOW**
- 23 two-hop questions still give the target's pronoun, which contradicts section 9 of the report.
- Grammar slip for "they": "They switched careers and now works".

**Departures from the panel's design:**
- Dropping A is justified.
- Dropping the paraphrased slice removes the only real check on whether this measures messy chat. Combined with critical 2, a PASS would not transfer.
- C's token budget is not matched to B's, so the founder's cost dissent cannot be judged.

### 3. Closest published work
- **[recall]** LongMemEval's fact-augmented keys (fact keys pointing to raw rounds) is essentially B. Also close: ReadAgent (gists plus page lookup), entity resolution at write time in Zep/Graphiti, HippoRAG 2, A-MEM, and Letta's file-search agent (74.0% on LoCoMo). NoLiMa's no-shared-words design is the right model for the desc and who questions.
- **[search]** [Mem-α (2509.25911)](https://arxiv.org/abs/2509.25911) trains Qwen3-4B with reinforcement learning to build its memory chunk by chunk.
- **[search, abstract only]** [Deng & Sinchaisri, Oct 2026](https://www.roboticscenter.ai/research/papers/stale-misattributed-or-late-where-personal-memory-fails-before-generation-2610) already scores identity resolution before generation, separates missed from false merges, and compares LLM against rule-based key assignment.
- **[search, unverified]** There is related 2026 group-chat entity-resolution work ([NORA 2026](https://aclanthology.org/2026.nora-1.10.pdf), [SocialMemBench](https://benchmarklist.com/benchmarks/socialmembench/)).

What a result would add: a modest data point. It would be the first time a phone-size model both files at write time and reads, scored at the level of single mentions and paired against agentic search. It is not a new method, and on templates it does not generalize.

### 4. My forecast (test world, code as it is)

| Measure | Forecast |
|---|---|
| Wrong merges | 2–6% |
| B, hard set | 62–78% (median 70) |
| C, hard set | 55–75% (median 64) |
| B − C on desc + who + two | +5 to +25 points (median +12); C is weak at intersecting several facts |
| Verdict | STOP 55%, BETWEEN 38%, PASS 4%, C 3% |

If the asymmetric relations, the scorer and the leak rule are fixed, STOP drops to about 45%.
