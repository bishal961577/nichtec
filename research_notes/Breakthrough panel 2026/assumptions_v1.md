# The assumptions under today's AI memory, and which one is blocking (moderator's draft, v1)

Written by the moderator from this project's record, before the panel. It is a starting point to attack, not a
conclusion. Every "evidence" line points at a measured result in this repository; everything else is argument and is
marked as such.

## The target, restated as capabilities

What the owner wants, broken into capabilities that can be tested one by one:

- **C1. Exact memory with receipts.** It never states something it was not told, and it can show the message behind
  every answer.
- **C2. Real deletion.** "Forget Kofi" removes Kofi and everything derived only from what was said about him.
- **C3. Association.** "You know Mira?" brings up everything about Mira, and about the people and things linked to
  her, including through others ("Kofi's flatmate").
- **C4. Integration ("as if always known").** A new fact changes answers that never mention it, and combines with
  other facts without being spelled out.
- **C5. Late connection (the shower moment).** Two things learned weeks apart that matter together get noticed, without
  being asked.
- **C6. Learning skills from experience.** It gets better at things it does, not only at remembering facts.
- **C7. Phone cost.** All of this runs on a phone, with answers in seconds and the heavy work at night on the charger.

## The assumptions, one by one

### A1. Knowledge lives in only two places: in the model's weights, or in text it re-reads

**Why it matters.** Weights integrate knowledge (C4) but cannot show receipts or delete (C1, C2). Text gives receipts and
deletion but does not integrate: the model must re-read it and reason over it every time.

**Evidence here:**
- GPT-J with keyed memory held 96% of single facts after 2,764 edits, but combined two stored facts 9.4% of the time.
- Fine-tuning does not compose either (Balesni et al., cited by the first panel).
- Text: the 4B reads the right messages at 95–99.9% (tests 1 and 2), but has to be handed them.

**Possible replacements (argument):**
- **A third form of knowledge,** structured and addressable, that the model can operate on directly. Candidates:
  - a typed store the model queries like a database;
  - per-fact compressed states (saved attention cache, "cartridges");
  - facts as small programs.
- **The fast-weight state the 4B already has.** 24 of its 32 layers (Gated DeltaNet) update a fixed-size state by a
  learning rule as they read. That state is a learned memory that integrates by construction, but it is small (about
  25 MB) and is thrown away after each conversation.

### A2. Learning happens in a separate training phase by gradient descent on one shared set of weights; afterwards the model is frozen

**Evidence here:**
- The founding argument (`AGI_BOTTLENECK.md`): gradient updates do not commute, so ordered experience overwrites.
- The project's from-scratch learners stopped at their lines: SPARC on language, the counting language model, and
  additive feature learning (70.72% vs 71.16%).
- The order-gap test: a 27B model learning from its full history beat the ledger.

**Possible replacements (argument):**
- **Complementary learning systems.** Fast exact episodic storage, plus slow consolidation into weights only for skills
  and regularities, during "sleep" (offline replay). Facts never go into weights, so deletion stays real.
- **Test-time training** on the person's own data.
- **Local, additive updates** for the memory part only.

### A3. Recall is triggered only by the current input; nothing happens between inputs

**Evidence here:**
- Test 1's search arm: the "is this enough?" step said DONE on 117 of 120 described-person questions. It fails as a
  feeling of knowing.
- Test 2: the search agent invented an answer to never-stated facts 37.5% of the time.
- No system in this project ever acts between questions.

**Possible replacement (argument):** memory as an ongoing process.
- Open questions and goals stay active ("who is Kofi's flatmate?", "is Mira OK with cats?").
- An offline process (the night shift) matches new material against them, reorganises, and flags connections worth
  telling the owner (C5).

### A4. Natural language is the universal interface: the model writes its memory as text and reads text back

**Evidence here (test 2, development phase, section 16):**
- Most of the 4B writer's errors were bookkeeping, not understanding:
  - wrong person numbers;
  - "NEW: He" written as a person;
  - "boss of | me" instead of "relation to me";
  - the fact put in the label;
  - one line per message.
- The clue writer put things in the name slot ("name: axolotl").
- Each was fixed by parsing code or an example. Each fix was a patch on an interface that is free text in, free text
  out.

**Possible replacement (argument): a small typed memory language between the model and the store.**
- The model only perceives. It turns a message into typed claims, and code checks the types.
- Code does identity, time, bookkeeping, deletion and merging, perfectly and auditably.
- This is "a new programming language" in a literal sense: what the model emits is a program run by a memory engine.

### A5. Who a mention refers to must be decided when the message is written

**Evidence here:**
- Test 2: "Kofi's flatmate adopted a ferret" was filed under the right person 11–22% of the time.
- 6.3% of all facts were filed under the wrong person. Every one of those is a false memory (breaks C1).

**Possible replacement (argument): deferred reference.**
- A mention becomes a variable with constraints: "the person who is Kofi's flatmate at date D".
- It is resolved when evidence arrives, the way a logic program binds a variable by unification.
- Facts attached to the variable flow to the person once it is bound. Until then the system says "Kofi's flatmate
  (unknown) has a ferret", which is true. The owner raised this himself: "this is how the human brain does it".

### A6. Memory is storage; recall returns stored items

**Evidence here:** the "oracle" index (facts as typed rows built from the truth) gives 86.1% on the hard set, against
70.1% for the 4B's own rows and 62.8% for search. A maintained structured world-state, built by perception, is better
than retrieval over text.

**Possible replacement (argument): memory as a maintained belief state.**
- What is believed now is kept: who is whom, what is true now, what changed when.
- Every belief has justifications (receipts).
- Deleting a message retracts every belief that depended only on it. This is what a truth-maintenance system does
  (Doyle, 1979), and it gives C1 and C2 by construction.

### A7. Time is just more text: "what is true now" is worked out from dated messages at question time

**Evidence here:**
- Updates were 82–88% right even with perfect pages.
- Retractions went from 47% to 100% only after the answer prompt was changed.
- Last-meeting dates went from 44% to 89% after a prompt change.
- The model is doing date arithmetic and "latest value wins" in its head, in thousands of thinking tokens.

**Possible replacement (argument):** valid-time facts. Every fact holds over an interval, "now" is kept up to date
automatically, and history is kept.

### A8. Reasoning means generating thinking text, token by token

**Evidence here (test 2):**
- Thinking-off answers fail even with perfect pages: 66.7% on the hard set.
- Thinking-on answers take a median of about 4,200 tokens, roughly 3.5 minutes on a phone (C7).
- The thinking writer ran past 8,192 tokens on 20-message and even 5-message batches and wrote nothing.
- Much of what that thinking does is bookkeeping a database does instantly: joins (who is Kofi's flatmate), latest
  value, date arithmetic, identity.

**Possible replacement (argument):**
- Move the bookkeeping into the engine (A4–A7), so the model reads a short, already-joined answer and only puts it
  into words.
- For real reasoning, consider latent (non-verbal) reasoning or a trained small reader.

### A9. Success is measured by answering questions put to the system

**Evidence here:** none of the project's tests measures C5 (noticing without being asked). Test 2 measures false
memories and deletion; few public benchmarks do.

**Possible replacement:** proactive tests. Plant fact pairs weeks apart that matter only together. Measure what is
noticed, and the false-alarm rate, against search and against reading everything.

### A10. The hardware: memory and compute are separate, so everything goes through a processor reading RAM

Speculative and far off for a small team. Content-addressable and associative memories (Hopfield networks, in-memory
compute) do recall-by-content in hardware. Listed so the engineer can judge it, not as a plan.

## Which assumption is blocking, by the evidence

| Capability | Blocked by (moderator's reading) | Strength of evidence |
|---|---|---|
| C1, C2 (receipts, deletion) | A4, A5, A6 | strong (test 2: 6.3% false filing, forgetting 89% average, worst 62%) |
| C3 (association) | A5 | strong ("Kofi's flatmate" 11–22% filed right) |
| C4 (as if always known) | A1, A2 | strong (9.4% composition, first panel) |
| C5 (late connection) | A3, A9 | none measured: an open gap |
| C6 (skills) | A2 | medium (order-gap and MovieLens tests) |
| C7 (phone cost) | A8 | strong (4,200-token answers, thinking writer cannot finish) |

## The moderator's candidate: a living belief system

An argument, not a result:

- **The small model only perceives and speaks.** It turns each message into claims in a small typed memory language,
  and turns query results back into words.
- **A memory engine maintains a belief state:**
  - typed facts with valid time;
  - people as identities, with mentions as variables bound when evidence arrives (deferred reference);
  - every belief justified by messages (truth maintenance), so deletion retracts exactly what depended on the message.
- **A night shift runs inference over the state while the phone charges:**
  - binds open variables;
  - detects conflicts and changes;
  - matches open questions and goals against new material;
  - queues connections worth telling the owner.
- **Skills only, never facts, are consolidated into the model's weights**, so deletion stays real.

**Honest status.**
- Every part is old: Prolog and Datalog, truth maintenance, knowledge graphs, complementary learning systems.
- What would be new is the whole, measured: a phone-sized perceiver plus this engine beating "read everything" and
  "search" on accuracy, false memories, deletion, late connections and phone cost.
- That would be an engineering advance. **It would not be a 2017-class shift.** The 2017-class shift would be C4: new
  experience used inside the model's thinking as if always known, which this candidate side-steps rather than solves.

## Questions for the panel

1. Is this the right list? What assumption is missing that, once dropped, unblocks something?
2. Which single replacement is the most likely to be a real shift, not a re-arrangement of known parts? Why?
3. What is the smallest experiment, runnable with this project's resources, that would show it (pass line) or kill it
   (stop line)?
4. Is the moderator's candidate a breakthrough, or just good engineering? What would have to be true for it to be more?
