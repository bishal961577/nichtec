# Round 1 — Neuroscientist

Labels: **[checked]** means I verified it today in the repo or by web search (papers only at the abstract level). **[recall]** means it comes from my own knowledge.

## 1. Audit: is the work done well?

**The process is good.** The lines were set before the run. Amendments were written before any model call. A recall check came before the run. The hand audit caught 30 scorer errors. Failed predictions stay on the record, including the earlier neuroscientist's 60–80% for described people (Haiku scored 100%).

**My confidence.** I trust the cloud verdict. I do not trust what the 4B's 50.2% says about search.

**Weak points:**

- **a) The search arm was handicapped at its deciding step** [checked, `brain/notebook_read.py`].
  - The step "are these notes enough, or search again?" ran with thinking off: greedy, 64 tokens, one extra round. Only the final answer got thinking (median 975 tokens).
  - The 4B replied DONE on 117 of 120 described-person questions and 59 of 60 "who" questions.
  - Then, thinking during its answer, it correctly declined to answer. 117 of its 119 described-person errors and 111 of its 148 two-hop errors are "no value" answers, not invented ones.
  - So 50.2% measures a fast "I know enough" judgement that was wrong. That is a feeling-of-knowing failure, the kind seen after frontal damage. It does not measure road C: meaning-based (embedding) search and a thinking, multi-round search agent were never tested.
  - The phone line fails fairly as registered. It must not be read as "search can't work".
- **b) No thinking text was saved** [checked]. `think_chars` is 0 for all 10,305 Haiku answers and all 2,607 Qwen answers. The likely cause is a field-name mismatch: newer vLLM returns the reasoning in a field named "reasoning" [recall]. Nobody can audit why the 4B failed or whether it looped.
- **c) "Perfect pages" is the question's own two-person group.** For described and "who" questions that means choosing between two people. So 99.9% is a ceiling for reasoning, not for picking the right person among similar ones.
- **d) The samples are small.** One sample per question at temperature 1.0, and n = 60 per kind, which gives about ±12 points.
- **e) The design scales the number of people, not the number of facts per person.**
  - In human memory, interference grows with how many facts hang on one cue (Anderson's fan effect) [recall].
  - Here each attribute value is shared by 7–34 people (median 17) [checked], but each person has only 9–44 notes.
  - Real close people will have thousands of notes and many updates. That stress is untested.
- **f) The pending 4B whole-notebook run has no pre-set line.** The owner will lean on that result, so fix the line now (section 4).

**What the 4B result means for human-style recall.** With the right episodes in front of it, inference is almost perfect. Every failure is at the step from cue to stored trace (Tulving's "ecphory").

- **Where word search works:** when the cue repeats how the note was written (encoding specificity). Names, updates at 99%, and close people with lures at 100%.
- **Where it fails:**
  - joint cues, such as "Krakow + parrot", where each cue alone matches about 17 people;
  - two-hop recall, where the middle person's name is not in the question.
- **How humans solve both:** iterative recall. Generate a candidate, check it, and use the answer as the next cue. The 4B did not iterate because its fast check said DONE.
- **Intrusions:** 28 two-hop answers gave the asker's own value instead of the other person's, a classic intrusion error [checked].
- **What the test does not show:** messy chat turned into notes, years of facts piling onto the same people, phone speed, or anything about roads B, C or E.

## 2. The token worry

A brain does not hold its life in working memory.

- It keeps about 4 chunks active, plus recently used items held in a silent state that a cue revives quickly (Stokes; Wolff) [recall].
- Everything else sits in synapses and comes back on demand. Recall revives only a small set of cells, and storage is superimposed, so it does not take more space with every memory.

**Where the worry misfires: re-reading.** The prefix cache pays for reading the notebook once: 4.7 s for the first read, 0.55 s for a later question [checked, preflight]. The model's cache is less like working memory and more like a stored record searched by content.

**Where the worry is right, in two un-brainlike ways:**

- **Storage grows with every note, forever.** About 32 KB per token, so 500K tokens is about 16 GB at 16-bit (the moderator's estimate; I checked the arithmetic).
- **Every generated token reads the whole cache, including about 1,000 thinking tokens per answer.** On a phone's memory bandwidth that is roughly a minute or more per thought-through answer (my estimate, not measured).

**What the brain suggests instead:**

- **Sparse reactivation:** for each question, attend only to the retrieved blocks of the cache (Quest, RetrievalAttention, EM-LLM) [recall].
- **Offline consolidation:** compress older periods. Cartridges (ICLR 2026) trains compact caches offline and reports about 38.6× less memory [checked].
- **On the model's design:** the Gated DeltaNet state is a fixed-capacity associative memory. It holds gist, not receipts. Exact recall must come from the 8 full-attention layers [recall/inference].

## 3. The association idea (road B)

**What the instinct gets right** [recall]:

- **Concept cells** (Quiroga 2005): one sparse unit per person, responding to her name, face or voice alike.
- **Indexing theory** (Teyler & DiScenna): the hippocampus stores pointers to memories, not their content.
- **Integrative encoding** (Zeithamova; Schlichting): new information that overlaps an old memory reactivates it and binds to it at the time of writing, which helps later inference.
- **Spreading activation** (Collins & Loftus): when two cues are given, activation meets where they intersect.
- **Built properly, an index would solve the hard kinds.** Make attribute values nodes too: "Krakow" ∩ "parrot" leads to one person, then follow one link for two-hop questions.

**What would mislead an engineer:**

- **i) Cues are often descriptions, not names.** The owner's example is cued by a name; the failures were cued by descriptions. Pages keyed by name don't help unless values are indexed too. Values shared by 17–34 people need weighting by how specific they are, plus inhibition.
- **ii) Near-identical names need pattern separation when a note is written.** A wrong merge (Joao into Joaquin) becomes a permanent false memory that every later recall inherits. A mistake at read time stays local to one answer.
- **iii) Updates and retractions.** Brains don't overwrite: extinction is new learning (Bouton). An overwritten profile loses the history and silently goes wrong on updates where only the order shows which value is current. Keep the raw notes as the truth and profiles as rebuildable caches.
- **iv) Linking creates false memories.** Associative recall produces confident false memories (the DRM lure paradigm) and source confusion. Every link needs to record which note it came from, and whether it was stated or inferred. Otherwise the "receipts" are fabricated.
- **v) "Everything comes to mind" is a cost, not a feature.** Unchecked spreading floods the prompt.

**Prior art:**

- **HippoRAG (NeurIPS 2024) and HippoRAG 2 (ICML 2025)** [checked]: they report a 7% gain on associative tasks over a strong embedding model. They capture the index and spreading activation (personalized PageRank, weighting by specificity). They miss three things:
  - pattern separation: their synonym edges do the opposite, completion, which would merge near-identical names;
  - time, meaning which value is current;
  - retraction.
- **Zep/Graphiti** stores validity intervals on facts, which covers time [recall].
- **SYNAPSE (2026)** adds lateral inhibition and decay [checked].
- **The sobering part** [checked]: AMA-Bench (ICML 2026) and EngramaBench (2026) find that putting the whole history in context beats structured memory systems overall. EngramaBench's graph system won only on queries that link across areas, and on just 30 such queries.

## 4. Vote

**(i) Next test:** B as an index (pointers to notes, not rewritten profiles), run together with a fair road-C control. Then A, C, D, E. Confidence 65%.

**(ii) Long-term phone memory:** D in this form:
- raw notes are the truth;
- an entity/value index built as notes arrive;
- a recent window kept in the cache;
- derived profiles rebuilt every night.

Then B, A, C, E. Confidence 55%.

**Conditional vote on the pending 4B whole-notebook run (hard set):**
- **≥ 90%:** reading is not the bottleneck at 60K tokens. Phone v1 is A for recent months, and the research becomes sparse reactivation and consolidation for years.
- **≤ 70%:** the small model cannot search the whole notebook by content. A selector is required, and the index test is the critical path.
- **Between 70% and 90%:** decide question kind by question kind.

**My prediction:** about 80% on the hard set, described people about 65%, and 15–30% stale answers on plain-restatement updates.

## 5. Next test: "index, then read" (hippocampal indexing plus intersection search)

**Write time:**
- The 4B, thinking off, turns each note into rows: (person, slot, value, note id, position).
- First-name mentions are linked to a person with the candidates shown.
- Rows are never overwritten.

**Question time:**
- The 4B, thinking on, names the cues in the question.
- Code intersects them, follows at most one relation link, and returns the notes they point to (60 at most).
- The same 4B reads those notes with thinking.

**Control arm:** a BM25 search agent with thinking on, up to 3 rounds, with the same token budget.

**Messy slice:** add 1,000 paraphrased notes using first names and pronouns, written by the 27B on Modal.

**Pass line:** all of these must hold:
- hard set ≥ 80%;
- described people, "who" and two-hop each ≥ 75%;
- wrong links ≤ 1% overall and ≤ 3% on near-name pairs, scored against the generator's own answer key.

**Stop line:** hard set < 65%, or wrong links > 3%.

**Cost:** about 2 L40S-hours (about $5) and 1–2 days of code.

**What each outcome decides:**
- **Pass:** B as an index becomes the long-term road.
- **Stop:** linking at write time is unsafe at 4B size; go with A plus sparse reactivation.
- **The control matches the index:** C is enough.

**Sources:**
- [HippoRAG 2](https://arxiv.org/abs/2502.14802)
- [Cartridges](https://arxiv.org/abs/2506.06266)
- [AMA-Bench](https://arxiv.org/abs/2602.22769)
- [EngramaBench](https://www.alphaxiv.org/abs/2604.21229)
- [SYNAPSE](https://arxiv.org/abs/2601.02744)
- [Spreading activation in knowledge-graph RAG](https://arxiv.org/abs/2512.15922)

No files were modified.
