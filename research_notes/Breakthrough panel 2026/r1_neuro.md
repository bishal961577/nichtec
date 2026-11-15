# Breakthrough panel, round 1: senior neuroscientist (memory)

**Labels:** [checked] = verified today (repository or web search; "abstract" or "secondary" marks a partial check); [recall] = my own knowledge; [speculation] = my idea.

## 1. The assumptions list

**What is wrong.**
- **C4 ("as if always known") is set higher than any brain reaches.** Brains do not silently combine two fresh, separately learned facts either.
  - A fact that fits an existing schema (a well-learned framework) is absorbed within a day or two (Tse et al. 2007, rats) [recall].
  - Anything else must first be active together with what it must combine with. This is panel 1's finding, and the brain shows it too (below).
  - So "C4 at once" asks for something no known brain does.
- **A3 is half right.** In brains, most connecting happens *when the second thing arrives*: it reminds you of the first, and the two are stored together.
  - Reactivating the first memory while learning an overlapping second one predicts later inference (Zeithamova, Dominick & Preston 2012) [recall; paper checked to exist].
  - The same lab's Molitor et al. 2021 found the same [checked].
- **A5 is neither new nor how brains work.**
  - "A mention is a variable, bound later" is the discourse referent of Discourse Representation Theory (Kamp 1981) [recall].
  - People don't hold unbound variables for weeks. They commit early, revise when they recall, and lose track of where they learned what (source monitoring, Johnson et al. 1993) [recall].
- **A6 should be anti-brain on purpose.**
  - Source memory is human memory's weakest part [recall], so receipts are where the project can beat the brain.
  - Time-stamped graph memory that invalidates outdated facts is already published (Zep/Graphiti 2025) [checked, secondary].

**What is missing.**
- **M1. Memory is goal-neutral.** C5 says "matter together", but matter to whom? 5,000 notes make 12.5 million pairs, and a language model can find a "connection" in most of them.
  - **Holds up:**
    - Unsolved problems make people pick up relevant hints met later in unrelated tasks (Moss, Kotovsky & Cagan 2007) [checked, abstract].
    - Replay (the brain re-running experience offline) looks prioritised by expected usefulness (Mattar & Daw 2018) [recall].
  - **Fails:**
    - Unfinished tasks are not remembered better (the Zeigarnik effect; ratio 0.99 over 37 studies), though people do tend to resume them (Ghibellini & Meier 2025) [checked, secondary].
    - "Sleep strengthens what you'll need" did not replicate (Ashton & Cairney 2021) [checked].
  - Goals act on attention and reminding, not by magic at night.
- **M2. A memory's meaning is fixed when it is written.**
  - Test 2 files each message once. Its 6.3% misfiled facts, and "Kofi's flatmate" filed right only 11–22% of the time, stay wrong unless something re-reads them [checked, brief].
  - Brains rewrite memories when they are reactivated (reconsolidation) [recall].
  - In mice, a salient new experience links back to a memory from two days earlier by offline co-reactivation. The link runs backwards only, and forms more in waking rest than in sleep (Zaki et al., Nature 2025) [checked].
- **M3. Memories are found through their own words.** The owner's two papers rarely share words, and people fail here too.
  - Given a story containing the solution, roughly 20–30% use it on an analogous problem without a hint, against about 75% with one (Gick & Holyoak 1980) [checked; secondary sources differ on the exact numbers].
  - What reminds people of an old case is surface similarity, not structure (Gentner, Rattermann & Forbus 1993) [checked].
  - Comparing two cases while learning roughly triples later transfer (Loewenstein, Thompson & Gentner 1999) [checked, abstract].

**The single assumption whose removal unblocks most: M2.**
- It is the common cause of three problems:
  - the measured blocker: filing errors that are never repaired (WRITER-LIMITED);
  - C5: a late connection is an old message re-read in the light of a new one;
  - clean retraction of derived beliefs.
- A5 and the useful half of A3 are special cases of it.
- It is cheap to drop, because unlike a brain the project keeps every raw message.

## 2. Candidate: memory that re-reads itself, keyed by what things are *for*

**The shower moment, by the evidence.**

| Folk claim | Evidence | Reliability |
|---|---|---|
| A break helps ("incubation") | 117 studies show a small-to-medium effect (d ≈ 0.3). It is larger for open-ended tasks, after long preparation, and with an undemanding filler task (Sio & Ormerod 2009) [checked, secondary]. | moderate |
| Sleep creates insight | Wagner 2004: over twice as many people found a hidden rule after sleep [checked, secondary]. But which sleep stage helps flips between studies. Lacaux 2021: light sleep (N1) 83%, deeper sleep (N2) 14%, awake 31%. Löwe 2025, preregistered: N2 86%, N1 64%, awake 56% [checked]. Groups had 14–49 people. Sleep gave no benefit on classic insight problems (Schönauer 2018, n=77) [checked]. | weak to moderate |
| Mind-wandering does it | Baird 2012 did not replicate (Smeekens & Kane 2016; Steindorf 2021) [checked]. | weak |
| Brains link memories across time unaided | In mice, shared neurons link memories formed about 5 hours apart, but not 7 days apart (Cai 2016) [checked]. Memories days apart link only through reactivation (Zaki 2025) [checked]. | strong, over hours to days |
| The "aha" means it's right | Usually, but false ahas are common and can be induced: in Grimmer 2022, 44.6% of people had false ahas on at least half the trials. An aha also makes false claims feel true (Laukkonen 2020) [checked, secondary]. | a noisy gate |

**My reading [speculation].** The shower gets the credit, but four things do the work:
1. how the first item was stored: by its surface or by its role;
2. a concern that stays open;
3. a reminding when something overlapping arrives;
4. a fast, fallible "this fits" check.

We remember the hits, not the misses. Sleep adds a modest, poorly understood boost.

**The minimal mechanism for an AI:**
1. **Role keys at writing.** Beside what a message says, the 4B writes in free text what it could be *for*: a need, an offer, a risk, a constraint or an open problem.
2. **Open concerns as standing memories.** Needs, plans and unanswered questions are kept until they are resolved.
3. **Reminding on arrival.** A new message or a new concern triggers a cheap screen over role keys and concerns, using embedding search (comparing meaning vectors). The 4B then re-reads the few old *raw* messages it found, beside the new one.
4. **A false-alarm budget and receipts.** Every connection is a derived note that cites both messages. Deleting either message deletes the note.
5. **An optional night sweep,** prioritised by the open concerns, kept only if the test shows it adds something.

**What to copy from the brain:**
- prioritisation;
- reminding on arrival;
- backward linking;
- co-activation before integration.

**What not to copy:**
- lossy gist;
- lost sources;
- sleep-stage stories;
- random recombination ("dreaming" as noise): nobody has shown it beats directed search at equal compute.

**Why it could be a shift, and why I doubt it.**
- Memories are keyed by *purpose* rather than by content (entities and words, as in test 2 and the moderator's engine).
- Connections are sought when a message arrives, not when a question is asked.
- One mechanism that repaired filing, surfaced late connections and kept deletion exact would be a new organising principle. I expect the field to call it a re-arrangement [speculation].

**Closest prior art** [checked, abstracts]:
- **A-MEM** (NeurIPS 2025): a new note rewrites the descriptions of its nearest old notes.
- **ProMemAgent** (2610.27400): links earlier events to later developments, to help without being asked.
- **ProAct** (2605.25971): uses idle time to anticipate needs.
- **Sleep-time compute** (Letta, 2504.13171): about 5× less compute at question time.
- **Anthropic "Dreaming"** (2026): surfaces "new insights" [checked, repository notes].
- **REALM** (2609.16053): rewrites memories each time they are retrieved.
- **Older roots:** MAC/FAC (Forbus, Gentner & Law 1995: a cheap screen, then a structural check) [checked]; literature-based discovery (Swanson 1986: fish oil and Raynaud's disease) [recall].

No abstract I found measures planted late connections against a false-alarm budget. None separates linking on arrival, overnight linking and backward linking. The full papers might.

**Runner-up** (shift-shaped, but beyond these resources): relational structure learned separately from content.
- The Tolman–Eichenbaum Machine (Whittington et al. 2020) learns a relation like "flatmate of" as a reusable structure. It is formally close to a transformer (Whittington et al. 2022) [recall].
- It would replace the hand-typed memory language with a learned one.
- It has been shown only on toy worlds.

## 3. The smallest decisive experiment: the late-connection test

**What is built.** A layer on `brain/messy_world.py`, with a new seed and not the sealed world. It plants 200 connections across 22 simulated months, 50 of each kind:

| Kind | Example |
|---|---|
| K1, shared words | Mira wants a cello; ten weeks later Ugo sells his cello. |
| K2, different words | "something for her daughter's string lessons" / "clearing out his old instruments" |
| K3, a risk that needs knowledge | Ana's sesame allergy / a Lebanese restaurant booked for her birthday |
| K4, backward | Two old, unremarkable messages that matter only once a later concern arrives. |

The world also has:
- 200 lures (surface overlap, no fit);
- two connection kinds that exist only in the test world.

The prompts are frozen on the dev world before the test world is generated.

**Arms.** All use Qwen3.5-4B, with thinking under a hard token cap (test 2's uncapped thinking writer never finished [checked]). Every arm gets the same number of tokens per simulated day.

| Arm | What it does |
|---|---|
| E | A-MEM-style reminding from the nearest old messages (the strongest cheap baseline) |
| E+ | Role keys, open concerns and reminding (items 1–4 of the mechanism) |
| N | A night sweep over random pairs of a new and an old message |
| N+ | A prioritised night sweep, plus a re-scan whenever a new concern arrives |

**Gate (under 1 GPU hour).**
- Shown a pair directly, the 4B must say "fits" for at least 80% of planted pairs and at most 20% of lures.
- If it cannot, no retrieval scheme can help.

**Measured by code.** This is possible because every connection cites its two messages.
- The share of planted pairs surfaced within 7 simulated days, by kind.
- False alarms per week: lure hits, plus a two-rater audit of 100 unplanted connections.
- Tokens per message.

**Pass:** E+ or N+ meets all of these:
- it surfaces at least 60% of K2–K4;
- it makes no more than 1 false alarm per simulated week;
- it beats E by at least 15 points on K2–K4, and a paired resampling check shows the gain is not luck.

**Night-shift line:**
- If N+ is within 5 points of E+ on K1–K3, the phone needs no general night sweep.
- Night compute then goes only to re-scans triggered by new concerns, and only if N+ beats E+ by at least 10 points on K4.

**Stop:** either of these:
- the gate fails;
- every arm finds under 30% of K2–K4 at no more than 1 false alarm per week.

Then C5 is out of reach at phone size: give it to a server model, or drop it.

**Cost:**
- **GPU:** about 10–20 L40S hours, roughly $20–40 at the road panel's implied rate of about $2 per hour [checked].
- **Tokens:** about 1,000 per message, against 154 for test 2's writer [checked].
- **Work:** about two weeks for one person: 4 days on the world, 5 on the arms, 3 on scoring and the audit.

## 4. The moderator's living belief system

**Good engineering, not a breakthrough,** as the moderator says. Every part is already published:
- Datalog (a rule language for databases);
- truth maintenance (Doyle 1979);
- discourse referents (Kamp 1981);
- time-stamped agent memory (Zep 2025);
- memory evolution (A-MEM).

**Two warnings from neuroscience:**
- **It inherits the old knowledge-base problem.** This is a symbolic knowledge base in the style of Cyc (a decades-long hand-built one), with a language model as its perceiver.
  - That lineage failed at getting knowledge in, and test 2 measures exactly that bottleneck (WRITER-LIMITED).
  - A typed language turns bookkeeping slips into errors that code catches, but it cannot tell who "she" is.
- **Hand-written types will run out.** Brains learn their relational structure. Hand-written types will meet relations nobody typed.

**It would be more than engineering if:**
- (a) derived beliefs changed answers to questions that never mention them, with deletion still exact;
- (b) the types were learned from data and carried to a new domain (papers, tasks) without a new schema;
- (c) it did C5 at a low false-alarm rate.

I would call (a) and (b) together a shift. My odds for that here are under 2% [speculation].

## 5. Odds

**(a) A measured win (E+ or N+ passes against E): about 30%.**
- The gate passes about 65% of the time. Given a pass, the pass line holds about 45% of the time.
- **Against:**
  - nearest-neighbour reminding already catches K1 and part of K2;
  - language-model judges over-connect;
  - in test 2, the 4B failed to link identities with thinking off.
- **For:** K3 and K4 need world knowledge and the owner's concerns, which plain nearest neighbours lack.
- About 75% that the test reaches a clean decision (pass, stop or the night-shift line).

**(b) A breakthrough: about 1%.**
- Even a clean pass yields two things:
  - a benchmark nobody has;
  - a design rule: "key memories by role and concern; spend night compute only on backward links".
- The field would call that a good systems paper.
- The 2017-class problem, learning new structure into the computation itself, stays untouched.

These are opinions. The project's forecasting record is 0 for 5, and every panel member runs on the same underlying model.
