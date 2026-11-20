# Growing Brain Test (November 2026)

**The one line every test in this line of work must answer to (the anti-drift rule):** does knowledge from the
agent's own experience, kept in its weights (a growing brain on a frozen head start), make later learning faster,
last, and carry over better than the same knowledge kept as notes that the same frozen model reads? A proposed test
that cannot say in one line how it answers this is not run.

Pre-registered: this document and the code (`brain/growing_brain.py`, `brain/modal_brain4b.py`) are committed before
any run of the real model. Lines, measures and the exam worlds are fixed here. Development on the two dev worlds may
change prompts and parsing only; every change is logged in section 10 before the FREEZE commit, and the eight exam
worlds run once after it.

## 1. Why this test

The owner's idea, in their words: give a frozen model "the understanding of how to understand things" plus basic
knowledge, the way children are born with what DNA gives them, and let each person's own brain grow from experience.
Companies' frozen models are like genomes, mostly the same; the brain is what makes each of us different.

Two corrections were agreed before this design:
- **A genome holds almost no knowledge.** It is about 750 MB of instructions for building and wiring a brain (Zador
  and colleagues' "genomic bottleneck", PNAS 2024), while the brain learns nearly everything. A frozen language model
  is the opposite: nearly all its knowledge is already in the frozen part. So "frozen model + personal memory of
  facts" is an educated adult with a diary, not a newborn with a growing brain. That is what this project's steps 1-8
  of the two-ends check were building, and it drifted away from this idea.
- **A brain, unlike a notebook, makes the next learning easier.** If a personal brain does not speed up later
  learning, carry what it learned to new cases, and keep it, it is only memory, and the idea reduces to a diary.

This test asks exactly that, on the smallest setting that can answer it.

## 2. What it stands on

- **Two memory systems:** complementary learning systems (McClelland, McNaughton and O'Reilly 1995). A fast store
  (hippocampus; here, the notes) and a slow one consolidated in sleep (neocortex; here, the nightly training of the
  weights). The arms below separate them.
- **Chunks:** working memory holds a few chunks (Miller 1956), and experts see larger chunks (Chase and Simon 1973).
  Here the combining step holds at most two questions, so a law of three or four features can only be learned
  quickly through an earlier concept used as one chunk.
- **Growing a library of concepts in wake-sleep cycles:** DreamCoder (Ellis and colleagues 2021). The notes arm is a
  plain version of a written library; the brain arm puts the library in the weights.
- **Notes against training for new knowledge:** retrieval beat fine-tuning both for facts seen in training and for
  new facts (Ovadia and colleagues, EMNLP 2024). This is the main reason to expect the notes to win here.
- **This project's own record:** the ingredients design (the model names yes/no questions, code combines them)
  passed on the 4B: 87.4% against the 27B reading its history at 86.2% (`reports/Learning on the job from first
  principles 2026.md`, section 9). Each day here uses that machine.

The question here is narrower than any of these: for one frozen small model and the same machine, does knowledge in
its weights beat the same knowledge in notes, on creatures it has never met?

## 3. The world

**Creatures.** Each world has 300 invented creatures (names like "tavo" or "mirek"), each with a written description
of 14 independent features: scales, fur, horns, wings, long tail, glows in the dark, lives near water, active at
night, eats only plants, lays eggs, sings, larger than a horse, stripes, more than four legs.
- Every feature is stated as true or false in the description ("It has scales." / "It has no scales."), in a random
  order.
- Seven features are true for 70% of creatures and seven for 30%, by the seed, so laws of two parts can be balanced.
- 200 creatures are met during the days; **100 are never met** and test what was kept.

**Events and laws.** Each day brings a new event (from 16 like "the gate opens for it", "the bell rings for it")
that happens or not by a hidden law. Each world cares about 6 of the 14 features.
- **Odd days (1, 3, 5, 7, 9, 11): a base law.** Two of the world's six features, each possibly negated, joined by
  "and" or "or". 30-70% of creatures say yes; no single feature agrees with it on more than 85% of creatures.
- **Even days (2, 4, ..., 12): a composed law.** An earlier event (possibly negated) joined with one of the six
  features or another earlier event. 25-75% yes; no single feature or earlier event agrees on more than 88%.
- Every law differs from every earlier one (at most 90% agreement, either way).
- **Composed laws are out of reach without memory.** No rule of at most two raw features agrees with a composed law
  on more than 90% of creatures. On 200 worlds that no stage uses (seeds 1000-1199), the best such rule's agreement
  is a median of 86.0% (10th percentile 81.0%, 90th 88.0%, maximum 90.0%). An agent with memory can reach 100%.

**A day.** 99 of the 200 creatures are met (a fresh draw each day). 35 of them are probes, balanced to the day's law,
and are never tested that day. The agent tests 24 of the other 64 in 6 rounds of 4 and sees each answer.

## 4. The agent and its three memories

**One machine, the same for all arms, each round:**
1. **Acting.** Code chooses the next 4 creatures: those on which its 8 cheapest explanations disagree most (each
   weighted by e to the minus its extra cost). In round 1, the same 4 random creatures for every arm.
2. **Naming.** Qwen3.5-4B, thinking off, sees the day's tested creatures (yes or no, each with its description) and
   lists up to 8 yes/no questions about a creature that might explain them. That is 4 samples at temperature 0.8
   (top-p 0.95, top-k 50, up to 160 new tokens). Up to 24 questions are kept per day, in order of appearance.
3. **Reading.** The model answers each question for each of the day's 99 creatures, from P(yes) against P(no) for the
   first answer token.
4. **Combining (code).** Every rule of one or two questions (each possibly negated, two joined by "and" or "or"),
   plus "always yes" and "always no". Cost in nats: errors on the tested creatures × ln 19 + questions × ln(2K) + ln 2
   for a joining word. The cheapest is the agent's explanation and is scored on the 35 probes.

**Three memories, the only difference:**
- **none:** the frozen 4B; nothing is kept between days.
- **notes:** the frozen 4B plus a notebook.
  - At the end of each day it writes its cheapest explanation as that event's law.
  - The next days' naming prompt lists every known event with its written law, and says a known event may be used
    as a question.
  - When a question names a known event, code works the event out for each creature from its written law, reading
    the plain questions with the frozen model.
- **brain:** the 4B plus one LoRA (rank 16, alpha 32, dropout 0.05, on the language model's linear layers, as in
  `brain/who_train.py`), zero at the start.
  - Each night it continues training (AdamW, learning rate 1e-4, 2 epochs, loss on the answer only) on:
    - **its own lessons from the day:** for each of the day's 99 creatures, its own yes or no for the event (the
      tested ones as observed, the rest from its explanation), in two wordings;
    - **the law it found, in words,** asked three ways;
    - **the list of events met so far;**
    - **replay:** everything from earlier nights;
    - **a little of its head start:** the frozen model's own answers to the 14 feature questions for 2 of the day's
      creatures (28 examples), so training does not pull the head start away.
  - During the day the prompt lists the known events' names only, never their laws. When a question names a known
    event, the model with its LoRA answers it directly. All its reading and naming use the LoRA.
- A question naming today's event, or an event the arm has not met, is dropped (circular or meaningless). The none
  arm is never shown events.

**References (reported, no lines):**
- the **oracle**: the true features and the true earlier events, with the same combining and acting;
- **notes+all** and **brain+all**: on the same tested creatures, every known event is added as a question. This
  separates bringing the right concept forward from reading it.

## 5. Measures and lines (fixed now)

**Measures, per world and arm:**
- **Learning curve:** the probe accuracy after each of the day's 6 rounds. **AUC** is its mean (higher is faster);
  **final** is after round 6.
- **Transfer:** the AUC on the six composed days.
- **Faster:** the AUC on late base days (7, 9, 11) minus early base days (1, 3, 5). This asks whether learning a new
  base law speeds up with experience. Law difficulty varies by day, so it is read only as brain minus notes.
- **Kept:** after the last night, every event is worked out for the 100 never-met creatures (notes by code from the
  written laws; brain from the LoRA), against the truth, averaged over all 12 events. The same is recorded after every
  night, giving a forgetting curve.
- **Head-start damage:** accuracy on the 14 feature questions for the 100 never-met creatures, for the frozen model
  and for the brain after the last night.

**Validity checks.** If any fails, the verdict is INCONCLUSIVE and the world is fixed on dev:
1. The world is learnable: the oracle's final accuracy is at least 95% on both day types.
2. Memory is needed: the none arm's final accuracy on composed days is at least 10 points below the oracle's.
3. The head start reads descriptions: its feature reading is at least 90%.

**Lines,** on the 8 exam worlds, paired by world. The 95% interval is the mean ± t(7) × standard error over the
8 worlds:
- **BRAIN:**
  - brain minus notes is at least +5 points with the 95% interval above zero, on transfer or on faster;
  - **and** the brain keeps at least 80% on never-met creatures;
  - **and** head-start damage is at most 3 points.
- **NOTEBOOK ENOUGH:** brain minus notes is at most +2 on both transfer and faster.
- **BETWEEN:** anything else, for example a brain that wins on transfer but does not keep, or damages its head start.

**Reported without lines:**
- brain minus none ("did the weights carry anything": more than +2);
- notes minus none;
- the references;
- final accuracy on composed days for probe creatures never met before, which separates remembering individual
  creatures from knowing the law;
- experiments to reach 90%;
- the forgetting curves;
- the questions named;
- the calls made.

## 6. What each result would mean (decided now)

- **BRAIN:** knowledge in the weights carried over and lasted better than the same knowledge in notes, on creatures
  never met. That is the first evidence for the owner's idea. Next: many more days, the brain bringing a concept
  forward without being shown its name, and the same on a phone.
- **NOTEBOOK ENOUGH:** at this scale, notes plus a frozen head start do everything a growing brain does. The idea's
  value, if any, has to show where notes fail:
  - when there are too many concepts to read (this project's 4B fell to about 20-25% reading a whole 5,000-note
    notebook);
  - or for skills that cannot be written down.

  The next test grows the number of concepts until the notebook overflows. If the brain loses there too, a "brain"
  for this idea should be notes plus code that grows a library, not weights trained at night.
- **BETWEEN:** diagnose with the references:
  - bringing the right concept forward: the +all references;
  - reading it: kept on never-met creatures;
  - forgetting: the curve over nights.

  Then decide with the owner.
- **INCONCLUSIVE:** the world is wrong; fix it on dev and rerun.

## 7. Forecast (mine, written before any run)

My guess: NOTEBOOK ENOUGH about 60%, BETWEEN about 30%, BRAIN about 10%.
- The notes are exact, and code works them out.
- The brain must turn 99 self-labelled examples a night into a law that holds for creatures it has never seen, and
  LoRA training on few examples tends to learn surface patterns.
- The published comparison (section 2) went the same way.

The interesting outcome would be the brain winning on transfer, by bringing a concept forward that the notes only
list.

## 8. Limits, stated before the run

- **The world is synthetic,** with features stated outright in each description, so reading is easy and the test is
  about learning laws, not perceiving.
- **12 days and 12 concepts are small.** The notes cannot overflow here, which favours them (see section 6).
- **A LoRA trained each night is one stand-in for a growing brain,** not the only one.
- **Code does the acting and the combining.** That is part of the head start in every arm, the same for all.
- **Designer's advantage.** The world, prompts and code are mine. Guards:
  - the exam seeds (101-108) are fixed now;
  - the exam worlds' laws are never printed or run with the stand-in before the exam;
  - the stand-in check and the CPU check run on dev seeds only;
  - the generator statistics use seeds 1000-1199.

## 9. Procedure and cost

1. **Stand-in check (done, no model, `--stub --stage dev`):**
   - every validity check passed;
   - the stand-in "brain", which only memorises, kept 50.2% on never-met creatures against the notes' 94.0%;
   - the bookkeeping matches.
2. **CPU check (Qwen3.5-0.8B on a CPU, two tiny days):**
   - Left-padded batches give the same margins as one prompt at a time (largest gap 0.0000).
   - Day 1 (naming, reading, combining), night 1 (98 examples, 10 steps, loss 0.24) and the "kept" check ran.
   - Night 2 was killed for lack of memory: the container has 15 GB, and the tiny model was run in 32-bit floats
     with 8,192-token training batches. It is being rerun with 1,024-token batches on the CPU only; the H100 keeps
     8,192, as in the who-model job.
3. **GPU check** (`modal run brain/modal_brain4b.py --stage check`): one dev world, two days, on an H100.
4. **Dev** (`--stage dev`): worlds 1 and 2. Prompts and parsing may change after this; every change goes in section
   10.
5. **FREEZE commit.**
6. **Exam** (`--stage exam`): worlds 101-108, once. The verdict is computed by `growing_brain.summarise` exactly as
   committed.

**Cost:** an H100 is roughly $4 an hour and a world takes about 30-50 minutes, so roughly $3 a world: about $1 for
the check, $6 for dev and $25 for the exam, roughly $30-45 in all.

## 10. Changes after dev (logged before FREEZE)

None yet.
