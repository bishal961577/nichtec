# Round 2: senior neuroscientist (memory)

**Labels:** [checked] verified today, source given; [recall] my own knowledge; [speculation] my idea. New arithmetic (no model calls) is in my scratch folder, `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/neuro/r2_arith.py`, from `results/messy_index_dev_dev1500/summary.md`.

## 1. What I now think was wrong in my round 1

- **My fix for misfiling.** I proposed a memory that "re-reads itself": the 4B re-reads old messages when something reminds it of them. For identity, that is the costly, noisy version of what code does exactly. The 4B writer does not even repeat itself: identical code gave 5.5% and 6.3% wrong merges, and two runs on the same 400 messages differed by 10 points [checked: report §16.3–16.4]. A code re-solve is exact and takes 0.04 s [checked: moderator re-ran `resolver_sim.py`].
- **My reading of "Kofi's flatmate".** I called it a commitment made too early. That was wrong. The world uses that phrase only after the relation has been stated [checked: moderator, `messy_world.py`], and the right person was on the writer's list 95.4% of the time [checked: report §15.1]. It was a failed lookup. Test 2 cannot measure deferred reference (a mention left open and bound to a person later).
- **My late-connection gate contradicted my pass line.** The gate let the judge accept 20% of lures (pairs that look related but are not), while the pass line allowed only 1 false alarm a week. At 25 candidate pairs a week, 20% means about 5 false alarms. The judge needs to stay near 2% [checked: arithmetic].
- **I missed the biggest writer loss.** Of about 641 true facts in the dev world, about 104 (16%) appear in no index row from their own message. Only 36 (5.6%) were filed under the wrong person [checked: arithmetic; "no row" includes values the scorer did not match]. All five of us built on the smaller number.

## 2. What moved me, and the brain evidence on "describe, don't decide"

Two checks moved me, both confirmed by the moderator: the engineer's check of the world generator and the founder's sorting of the 49 wrong rows. Together they show the damage comes from the 4B doing joins in its head (linking facts across messages), not from too little re-reading.

- **The brain decides who is who, with memory in view.** Listeners work out who "she" is word by word, from what they already know (Tanenhaus et al. 1995) [recall]. Single neurons in the human temporal lobe fire for one person whether shown a photo or the written name (Quiroga et al. 2005) [recall]. So the engineer's describer works unlike the brain.
- **The brain also shows the price of deciding that way.** Its typical errors are the project's errors: the fact is kept, but the person or source is wrong. The brain does not store a memory's source as a label; it works out the source at recall from blurred features (Johnson, Hashtroudi & Lindsay 1993) [recall]. Later information rewrites earlier memories (the misinformation effect, Loftus) [recall]. No brain can delete one episode. On receipts and deletion the brain is a warning, not a model. That supports the engineer.
- **Where the brain agrees with him.** The hippocampus keeps similar episodes apart instead of merging them ("pattern separation") [recall]. A leading model has the brain store episodes separately and draw conclusions that span them only at recall (REMERGE, Kumaran & McClelland 2012) [checked: abstract]. But the brain also combines memories while learning (Zeithamova et al. 2012) [recall], and Schlichting et al. 2015 propose that the two alternate, event by event [checked: abstract]. The brain does not settle the question.
- **Where the brain warns him.** A passage read without its context is poorly understood and poorly remembered (Bransford & Johnson 1972) [recall]. A describer that sees one message and the one before it will describe "she finally said yes" only vaguely. Test 2's generator has little of this; real chat is full of it. I expect missed facts, not wrong binding, to break first.

**So:** the brain should not decide the design, and I concede that point to the engineer. Its lesson is "keep the episode intact and work things out again later". The project can do that exactly, because it keeps every raw message. My purpose keys survive as one field of the description: need, offer, plan or risk. That field depends on one message only, and it is close to the engineer's plan and wish tags and his standing items.

**Where I don't move.** The engineer's late-connection join links facts only through the same person. It cannot find the owner's own example: two papers read ten weeks apart share no person. People are reminded by surface similarity, and a deeper link needs the first item to have been stored by its role (Gentner et al. 1993; Loewenstein et al. 1999) [checked, round 1].

## 3. The convergence: partly real, mostly inherited

The strongest argument against it:
1. **Same model, same priming.** One model read one brief headed WRITER-LIMITED. Its assumption list already proposed deferred reference (A5) and a code engine for the bookkeeping (A8). The owner asked for a 2017-shaped idea, "remove recurrence" (stop carrying state from step to step) is the most available 2017 story, and three of us told it. That is an attractor, not five independent confirmations.
2. **The evidence supports a narrower root.** What it shows is that the 4B cannot join facts across messages in one pass without thinking. It fails when writing: "Kofi's flatmate"-style mentions were filed right 11.1% of the time, with the answer on its list. It fails when reading: with perfect pages and thinking off, two-hop questions scored 23.5% [checked]. Nothing shows that deciding at write time, or making the decision final, is the problem.
3. **The shared fix misses most of the loss.** Missed facts (16%) are three times wrong-person facts (5.6%). B's writer put lure values into 51.9% of the messages that contained a lure [checked]. Both are errors in reading a single message. Also, the index built from the truth scores 86.1% against 95.5% for perfect pages, and 65.6% on who-questions [checked]. Perfect binding would still leave a gap on the reading side.
4. **No intervention yet.** Every supporting number is either a post-hoc reading of one run or a simulation by someone who had read the generator.

The field is moving the same way (DISCERN, MemReader-4B [checked by the researcher]). That makes the direction more likely right, and less likely new.

**A cheap control separates the two roots** [speculation: about 1 GPU hour]. Re-run B's writer with the *final* index in view. If wrong merges fall to about 1%, the order of writing and the finality of decisions were the problem. If not, the problem is the joins in the 4B's head, and only the describer route removes it.

## 4. Vote

**I vote for the memory-blind describer with a code binder, run as one experiment.** The engineer's describer arm (D), the founder's order-free reader plus binder, and the researcher's stage 1 are the same build. Its first two days are the ceiling check that three proposals share: the researcher's stage 0, the architect's pre-joined check and the founder's fast path. The 4B, thinking off, reads perfect pages that code has already annotated. If that check hits its stop line (below 80%), the three weeks are not spent.

**Three additions:**
- a fact-recall line in gate 1: at least 90% of true facts appear in some description (B: about 84%);
- the final-index control from section 3;
- the founder's 300 hand-marked messages from the owner, which would give the only number from a real life.

**Why it goes first:** it attacks the measured failures (facts filed under the wrong person, 3.5-minute answers, incomplete forgetting) for 4–8 GPU hours. Any late-connection join also needs exact identity first.

**Why not my own test:** besides the gate error, on a generated world we decide what "fits", so the test would measure our own templates. Late connection should be tested on material we did not write: the researcher's stream of papers, or the owner's diary (section 5).

The architect's free laptop prototype is the only proposal aimed at the 2017-shaped question. It is worth running as a side bet only.

**My probabilities [speculation]:**
- Stage 0 passes its line (at least 88% overall, each bookkeeping kind at least 75%): about 60%.
- The merged core pass line holds: about 25%. It requires all of:
  - D with thinking off beats B with thinking on by at least 5 points, with the interval above 0;
  - wrong merges at most 1.5%;
  - every person loses at least 90% of their messages when forgotten;
  - after a deletion, the memory is identical to one that never saw the message.
- The engineer's full PASS: about 15%. It also needs who-questions at 75% or more, where even the index built from the truth reached 65.6%.

## 5. To the owner

**Is anything here a 2017-class breakthrough?** No. Nothing on this table is one, and all five of us put the chance at about 1%. The useful idea is this: the model describes each message, and code decides who is who from the whole record. It is good engineering, and the field is converging on it.

**Your shower moment.** The experience is real. In a survey of 1,114 people, 80% reported insights, often in the shower, at night or in quiet moments (Ovington et al. 2018) [checked: abstract]. But the shower mostly gets the credit:
- a break helps, modestly;
- what sleep does is disputed;
- connections often happen when the second thing arrives and reminds you of the first, because a question was still open in your mind;
- the "aha" feeling is sometimes wrong.

[checked in round 1; "often" is my reading]

So keep the goal and drop the mechanism. An AI needs no dreaming phase. It needs three things:
- to know what you are working on;
- to check each new item against those open questions as it arrives;
- to interrupt you rarely, and show both sources when it does.

**The honest best case for the month (about 1 in 4):**
- facts filed under the wrong person fall from about 6% to about 1%;
- answers fall from about 3.5 minutes of thinking to a few seconds;
- "forget Kofi" becomes complete and checkable;
- you get a first score on your own messages.

The more likely outcome: binding is fixed, but the model still drops facts while reading. The next step would then be a few GPU hours of training the describer.

**What to do now:**
1. Run the sealed test 2 as registered. It is your honest baseline.
2. Run the two-day ceiling check, then the describer.
3. Start a shower-moment diary today. Each time two things click, write down the two items, their dates, what you were working on and what set it off. Thirty entries are the only fair test of your idea, because nobody on this panel designed them.
