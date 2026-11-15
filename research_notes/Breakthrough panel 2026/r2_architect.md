# Round 2: machine-learning architect

Labels: **[checked]** verified today, source given; **[recall]** my knowledge; **[speculation]** my idea.

## 1. Corrections

**"Subtract or Replay?": the number exists, but my citation was bad.**
- arXiv 2607.27539 has four versions. v2 (13 Aug 2026, "Subtract, Transport, or Replay?") says in its abstract, for the 48B Kimi Linear hybrid: "the corpus-pooled raw recurrent contribution changes by 12-49% with the suffix" [checked: arxiv.org/abs/2607.27539v2]. v3 and v4 (September) are a rewrite retitled "Can an AI Assistant Really Forget?", on Gemma 3 4B. Their comments say the delta-rule work moved to "a separate companion paper" [checked: …v3]. The moderator read v4, which does not contain my claim.
- The companion paper is arXiv 2609.06872 (v3, 25 Sep 2026; same single author, a small company). On the 48B Kimi Linear [checked: full text, arxiv.org/html/2609.06872]:
  - "Adding the delta rule produced 12–49% movement" under two equal-length suffixes, "while additive writes alone showed none";
  - subtracting the record's saved footprint left 0.7–32.9% (median 12.6%) of the state;
  - replay from a checkpoint matched the never-stored state in all 80 contexts.
- I made three mistakes:
  - I cited an ID without a version, and that ID now shows different content.
  - My paraphrase ("12–49% of a record's contribution depended on later writes") was stronger than the text ("changes by 12–49% with the suffix").
  - I labelled a search summary "[checked]".
- **What changes:** the direction of my argument stands, but its weight drops. "Exact deletion depends on how the store was written" now rests on three things, and not on a 48B result:
  - a single-author preprint on synthetic contexts;
  - plain algebra: an addition can be subtracted exactly, but a delta-rule write (one that corrects what the store already holds) depends on what it found [recall];
  - the engineer's reproduced simulation (rebuilding from the message set is exact).

**The same mistake in a second citation.** My Block-Attention numbers (HotpotQA 68.5 vs 69.5) come from v3 of 2409.15355 [checked: Table 1]. The current v5 (Tulu3-8B) gives 72.3 vs 74.8 on HotpotQA and 72.2 vs 73.2 on 2Wiki [checked]. The gap is 1–2.5 points, not 0.1–1.

**Wrong in substance:**
- I called test 2's "Kofi's flatmate" failure the place where locality should hurt. The engineer showed test 2 never uses a reference before its definition [moderator confirmed]. With a perfect writer, the right person was on the 4B's list 95.4% of the time [checked: report §15.1]. So the 11% was a failed lookup, and test 2 cannot test locality.
- "Train the interface" was too coarse as the single assumption whose removal unblocks most. The engineer's rule is sharper: the writer must not see the memory. Training is the fallback.
- Blaming the notebook's 20% on the fast-weight state put a whole-model failure on one part.

## 2. What moved me

- **The engineer's simulation, reproduced by the moderator.** Code works out who is who from per-message descriptions. It rebuilds in 0.04 s, deletes exactly and ignores arrival order. That gives my candidate's accountability without a new network. The only distinctive part left in my proposal is a *trained* reader that binds in one pass, and that matters only for C4. My from-scratch experiment drops from "first" to "long bet".
- **The founder's breakdown:** 42 of 49 wrong rows are binding across messages [moderator confirmed].
- **DISCERN** [moderator confirmed] already states my principle for systems: "preserve at write, resolve at read".
- **The neuroscientist:** brains don't do strong C4 either. I accept the engineer's measurable C4: the fact is cheap to reach, shapes answers that never mention it, and combines with other facts in a few tokens. That makes C4 and C7 one target.

## 3. Is the convergence real?

Five copies of one model agreeing count as about one opinion, and the brief seeded it: A4, A5 and the verdict "WRITER-LIMITED". Two things do not come from us: the 4B's measured failures, and human authors who reached the same rule (DISCERN; collective entity resolution, 2007). So the diagnosis holds **for this world and this model size**.

**Strongest argument against: the root may be the generator's shape, not the problem's.**
- `messy_world.py` makes every reference by rule: definitions come before use, and bare names appear only when they are unique. A code binder simply inverts the generator.
- Real life has references no rule settles: two Anas active at once, "she" after a change of topic, people never named. A model must still judge those, only later.

I tested one piece of this today [checked: CPU only, `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/architect/definition_errors.py`, output in `definition_errors_out.txt` beside it].
- **What the engineer's simulation leaves out.** Its injected errors never touch definitions, the messages with the most downstream uses. Introductions ("my brother, Gaia Okafor") and nickname messages have one mention each, so the "wrong mention" error cannot fire there, and their nouns are never changed.
- **Why it matters.** In the dev world, 12 role definitions carry 952 "my mum"-style mentions.
- **One misread definition.** I misread exactly one role definition, trying all 108 noun swaps. Code then files on average 13 extra facts under the wrong person (median 9, worst 53) and leaves 18 unfiled. A misread ordinary fact costs one.
- **At a 10% misreading rate.** With role definitions misread 10% of the time and everything else perfect, wrong filing averages 0.70%. The 90th percentile is 2.0% and the worst 4.3%.
- **What follows.** A code binder's errors are rarer but bunched: two or three messages can decide a 1.5% pass line.
- **The upside of the same fan-out.** One correction repairs dozens of facts. So the system should ask the owner to confirm the few definitions ("Is Gaia your brother?"), the founder's M4 [speculation].

**Second argument against.** Systems built on larger models resolve identity at write time (Zep/Graphiti, Hindsight) [checked by founder and engineer], and they work acceptably on their own benchmarks [recall]. The 4B failed lookups it had the means to make. So "decided at write time" may be a 4B capacity problem that training fixes, not an architectural root.

**Verdict:** real here, unproven for life. The test that tells the two apart is the owner's own messages (below).

## 4. Vote

The founder's reader and binder, the engineer's describer and the researcher's stage 1 are **one experiment**: describe each message alone, then bind in code. The researcher's stage 0 and my pre-joined check are its first arm. I vote to run this merged experiment first.

- **Day 1 (under 1 L40S-hour).**
  - Built: true descriptions go through the same resolver and evidence card, and the 4B answers the 288 dev hard questions with thinking off.
  - Pass: ≥ 88%, at a median of ≤ 64 tokens.
  - Stop: < 80%. That would mean thinking is not mostly bookkeeping, and the phone-speed case dies for every design on the table.
- **Describer gate (300 dev messages):**
  - mention attachment and relation direction ≥ 95%;
  - lures tagged ≥ 90%;
  - **definition messages ≥ 98%**, because of the fan-out above.
- **Main pass, on a fresh, never-viewed seed with the lines committed first; all of:**
  - wrong merges ≤ 1.5%;
  - "Kofi's flatmate"-style facts filed right ≥ 75%;
  - worst-person forgetting ≥ 90%;
  - 100 of 100 deletions identical to never-seen;
  - reversed order gives an identical memory;
  - hard set with thinking off ≥ B with thinking + 5 points, with the people-resampled interval above 0;
  - median answer ≤ 64 tokens.
- **Stop, any of:**
  - day 1 < 80%;
  - wrong merges > 3%;
  - thinking off < 70%;
  - a fall of more than 10 points from dev to the fresh seed.
- **Transfer check.** The owner marks who is meant in 300 of his own messages. If the wrong-person rate there is more than 3× the synthetic rate, the synthetic pass does not count.
- **Cost:** 5–8 L40S-hours (about $10–20) and about 3 weeks. The resolver already exists.

**Why first:**
- It attacks the measured blocker.
- Its first day can kill the phone-speed claim for all five designs.
- The late-connection join needs settled identities.

The neuroscientist's 1-hour gate can run the same week. Mine waits as a laptop-scale long bet.

**Odds [speculation]:**
- day 1 passes: about 65%;
- the full pass line: about 20%;
- the headline (thinking off beats B with thinking, and wrong merges ≤ 1.5%): about 35%.

## 5. To the owner

- **No 2017-class breakthrough is here.** All five of us proposed careful rearrangements of known parts. The 2017-class question, new experience used inside the model's computation as if always known, is untouched by everything on the table, mine included.
- **The honest best case for the month:**
  - the phone model describes each message without looking at its memory, and plain code works out who everyone is;
  - answers take about 5 seconds instead of 3.5 minutes (the engineer's estimate);
  - wrong-person memories fall from about 6 in 100 to about 1 in 100;
  - "forget Kofi" removes all of Kofi, provably;
  - you get the first number from your own messages.

  That would be the project's first memory one person could trust daily. It would not be a new science.
- **What to do now:**
  1. Run test 2's sealed scored run as registered.
  2. Start the merged experiment, day 1 first.
  3. Spend two hours marking who is meant in 300 of your own messages.
  4. Keep one day in five for one long bet, chosen by what breaks.

The 2017 shift came from the bottleneck of a working system, seen on a shared test. You are close to having the working system. Watch where it breaks.

**Sources:** [2607.27539v2](https://arxiv.org/abs/2607.27539v2) · [2607.27539v3](https://arxiv.org/abs/2607.27539v3) · [2609.06872](https://arxiv.org/abs/2609.06872) · [Block-Attention v3](https://arxiv.org/html/2409.15355v3) / [v5](https://arxiv.org/html/2409.15355) · [DISCERN](https://arxiv.org/abs/2611.15729)
