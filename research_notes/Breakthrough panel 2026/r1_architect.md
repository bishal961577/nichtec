# Round 1: machine-learning architect

Labels: **[checked]** = verified today, source given; **[recall]** = my own knowledge; **[speculation]** = my idea.

## 1. The assumptions list

**What is wrong**

1. **A1: "the fast-weight state integrates by construction" is wrong where it matters.**
   - **Small and lossy.** 24 layers × 32 heads × 128×128 numbers: about 25 MB at 16-bit, 50 MB at the 32-bit precision the model stores it in [checked: Memory road panel brief, line 46, and that panel's config check]. Such a linear memory cleanly holds roughly as many items per head as its key width (128) [recall: Schlag et al. 2021]. A life does not fit.
   - **The least accountable store there is.** The delta rule (Gated DeltaNet's learning rule: each write corrects what the state already holds) makes a message's trace depend on later messages. In a 48B hybrid, 12–49% of a record's contribution depended on later writes, so deletion needs re-reading [checked: "Subtract or Replay?", arXiv 2607.27539, abstract via search; single-author preprint].
   - **It did not integrate in practice.** It was running when the 4B read the 5,000-note notebook and scored 20% [checked: brief].
2. **C4's evidence is stale.** The 9.4% figure is GPT-J (2021) without thinking. In test 2 the 4B with thinking scored 95.5% on perfect pages, including identity joins and two-hop questions [checked: test 2 report, §16.7]. Integration over text works; doing it cheaply does not (66.7% thinking off; median 4,200 tokens thinking on). **So C4 and C7 are one blocker: the cost of integration at use time.** "As if always known" means integration already paid for.
3. **A4–A7 are one assumption counted four times:** the stored form is free text, written at write time by an untrained model. Splitting it inflates the case for the moderator's candidate.
4. **A2's evidence is the wrong evidence.** The project's from-scratch learners losing shows they were weak, not that gradient descent blocks a phone memory. No failure measured in tests 1–2 involves weights.
5. **Drop A10.** Nothing here is hardware-limited at this scale.

**What is missing**

- **M1. "The model is an off-the-shelf part used through prompts; the memory interface is never trained."**
  - No language model's weights have ever been trained here; the only gradient code trains CIFAR baselines and per-fact value vectors [checked: `brain/step1.py`, `reallm/memtest*.py`].
  - Every test-2 fix was a prompt or parser patch [checked: §16.2–16.7]; the thinking writer never finished [checked: §16.8–16.9].
  - The first panel's rule: a new fact joins thinking only if co-active, or if the network was trained to read its memory.
  - The world generator knows who every "she" and "Kofi's flatmate" is: a free, unlimited, perfectly labelled training signal, never used.
- **M2. "Deletion and receipts can be added afterwards."** They cannot: exactness of deletion is a property of how the store was written [checked: the thesis of arXiv 2607.27539]. Test 2's rows were filed by a 4B whose prompt held everything filed before, so a row depends on messages that are not its receipt. "Forget X" at 89% (worst 62%) is a symptom [numbers checked; causal reading is speculation].
- **M3. "Reasoning is paid when the question arrives."** Precomputing between questions cut test-time compute about 5× (sleep-time compute, Lin et al. 2025) [checked: arXiv 2504.13171].

**The single removal that unblocks most: M1.** Training the writer and reader on generated lives attacks the measured blockers: wrong filing (C1, C3), thinking-off joins (C7), the writer that thinks forever. It is a precondition for any C4 route. It is boring and not a shift. Its risk is designer's advantage: train on one generator family, test on another and on human chats (LoCoMo). Do not train on `messy_world.py` while test 2's scored run is pending.

## 2. My candidate: write locally, read globally

**In plain words.** Each message is turned, alone, into a few additions to a fixed-size memory table. No stored piece depends on two messages. All connecting (who "she" is, who Kofi's flatmate is, what two facts imply together) happens when reading, by a network trained in advance on millions of simulated lives to read the table in one pass, without thinking text.

**By construction:**
- **deletion** = subtract the message's additions: exact, one message's cost;
- **receipts:** the read is a sum, so every answer splits exactly into per-message shares;
- **order-free:** a sum ignores order (the founding SPARC thesis);
- **constant read cost** however long the life: sparse "product-key" addressing, millions of slots, a few dozen per message [recall: Lample et al. 2019];
- **a hook for C5:** every write is also a read; a strong hit on an old message not linked by name is a candidate connection, Schank's "reminding" (1982) [recall].

It joins three of this project's findings: commutative memory (`AGI_BOTTLENECK.md`), fast learning needs a representation learned in advance (MovieLens), and facts join thinking only via a trained reader (first panel).

**Why it could be a shift** [speculation]. Today's belief (the moderator's A1): accountability and integration trade off. The hypothesis: **integration never needed writes that mix memories; a trained read over a summed store is enough.** That is the 2017 move: transformers dropped recurrence (state mixed at write time) and put all interaction into attention reads. If it holds, accountability becomes structural, not a bookkeeping layer. If it fails, the gap is the measured price of accountability, a number nobody has.

**Prior art (searched today).** The parts are old; some combinations are published:
- Memory networks (2015): sentences encoded separately, multi-hop reads [recall].
- TOME (ICLR 2022): a vector per entity mention, attended inside a transformer [checked: arXiv 2110.06176].
- Larimar (ICML 2024): least-squares memory, forgetting by reverse writes; 1–3K single-fact edits, no multi-hop test [checked: arXiv 2403.11901].
- MAC (NeurIPS 2024): documents encoded alone, combined at question time; every question comes from one document [checked: arXiv 2403.04317].
- **Block-Attention (ICLR 2025):** passages encoded alone; after fine-tuning it matches full attention on multi-hop QA (Llama3 HotpotQA 68.5 vs 69.5; 2Wiki 73.3 vs 73.4) [checked: arXiv 2409.15355]. The strongest support for the hypothesis and the biggest threat to its novelty.
- PICASO (ICLR 2025): composing separate states, 91% of concatenation's gain [checked: arXiv 2502.17605]; MesaNet (ICLR 2026): least-squares fast weights, reset per sequence [checked: arXiv 2506.05233].

**Not found** [checked: two searches]: a test where records are not self-contained (pronouns, "my mum", "Kofi's flatmate", corrections) and binding must happen at read time in one pass. Wikipedia passages name their subject; chat does not. That is where locality should hurt, and it is this project's exact failure.

**Runner-up: sleep that can be undone.** Nightly, the 4B thinking over the right pages teaches a small trained cache per person (a "cartridge"; Cartridges, ICLR 2026, matches in-context reading at 38.6× less memory [checked: arXiv 2506.06266]) to answer without thinking. Deleting a message re-trains only the cartridges whose teacher prompts contained it: an exact rebuild, as in SISA [recall]. It targets C4 and C7 on the real 4B; a rearrangement, with better odds of a usable win and lower odds of a shift.

## 3. The smallest decisive experiment

**Built** (from scratch, PyTorch, about 40M parameters per arm, equal sizes):
- **A new life generator,** disjoint from `messy_world.py` (own templates, names, relation types; some held out for testing): pronouns, nicknames, "my X", second-hand facts, corrections, deletions; 300–2,000 messages per life.
- **G, read everything:** a standard transformer over all messages plus the question. Upper reference; deletion = re-read.
- **L-KV:** the same layers, each message encoded alone (a second version also sees the 2 previous messages; deletion then re-encodes 2). The question attends to everything stored.
- **L-SUM:** each message written alone as sparse additions into about a million slots; the question reads in 2–3 hops.
- **R:** retrieval of the top 8 messages by name and meaning, plus the same reader.
- All answer in one forward pass, no thinking.

**Measured** on held-out lives:
- accuracy by kind, with binding kinds pooled (pronoun, nickname and "my X" referents; second-hand "X's flatmate" facts; two-hop; latest value after a correction);
- deletion error: largest output-score difference against a store that never saw the message;
- receipts: how often the largest share is the true source;
- L-SUM at 2,000 vs 500 messages;
- reminding precision for C5 (reported, no line).

**Pass (all):** an L arm ≥ 0.9× G on binding kinds at 500 messages; beats R by ≥ 5 points; deletion error ≤ 1e-3 (32-bit); receipts ≥ 90% on single-source facts; L-SUM at 2,000 ≥ 0.85× its score at 500.
**Stop:** G ≥ 70% on binding kinds but both L arms < 0.75× G after a fixed budget (3 settings × 2 seeds). Then structural accountability costs integration at this scale, and binding belongs in a symbolic engine or in undoable sleep. If G < 70%: one scale-up to about 150M, then stop.
**Between:** report, claim nothing. **Only on a pass:** graft L-KV onto a pretrained 0.6–1.7B full-attention model with LoRA (a small trainable add-on to frozen weights); test on test 2's dev world only.

**Cost:** a free laptop prototype at about 10M parameters first (kills the idea early if G cannot learn binding); then about 30–40 L40S-hours (roughly $60–80 at Modal list price [recall]); about 3 weeks for one person.

## 4. The moderator's candidate: good engineering, not a breakthrough

I agree with the moderator's honest status, and add:
1. **Closer prior art.** Zep/Graphiti keeps time-windowed facts traced to source episodes [checked: arXiv 2501.13956]; ChatDB (2023) put an LLM in front of SQL [recall]; deferred reference is "identity uncertainty" (Pasula et al. 2003) and BLOG [recall]; the ATMS (de Kleer 1986) tracks alternative assumption sets [recall].
2. **The engine is the easy half.** A typed language makes the perceiver's errors checkable, not absent. "Kofi's flatmate" filed right 11–22% of the time is perception; deferred reference moves it into the engine, which then needs a learned matcher for who "she" probably is, whose decisions need justifications too.
3. **Truth maintenance is only as honest as its justifications.** A 4B's claim depends on its whole prompt. Either the perceiver writes locally (one message in, claims with open variables out: M2), or deletion over-retracts or needs re-reading. The design must say which.
4. **Its cost claim (A8) is testable this week.** Give the 4B, thinking off, the oracle index's pre-joined answer per hard dev question ("Kofi's flatmate = Ana Varga [#812]; pet: ferret [#1033], since 2025-03-02"). Pass: ≥ 90% at ≤ 300 tokens (the engine route solves C7). Stop: < 80% (the 4B's reading, not bookkeeping, is the bottleneck). Under 1 L40S-hour, 1–2 days.
5. **More than engineering would need:** a learned typed language (predicates invented from experience, still auditable); night-shift derived beliefs that change answers nobody asked about, at near-zero measured false-belief rate; C5 with a precision figure. The first is symbolic AI's old unsolved problem; making it work on messy chat would be noticed. Without it, this is a well-built personal knowledge graph.

## 5. Odds

The project's forecasting record is 0 for 5 [checked: first panel report]. All numbers below are [speculation].

- **(a) Measured win within resources.** About 30% that an L arm passes against G and R: Block-Attention says locality costs little when records name their subjects; chat records do not, and that is unmeasured. About 15% for fixed-size L-SUM alone. About 10% that this route then beats the project's current best within its resources: the 4B's own index, 70.1% thinking on, beaten at thinking-off cost.
- **(b) A breakthrough as the field means it: about 1–2%.** A clean pass is a 40M-parameter result on synthetic lives amid dense prior art. It becomes a shift only if it holds at scale on real chat and shows that integration never needed mixing writes: a claim about all memory, not one experiment.
- **For comparison, M1 (training the interface):** about 40% to beat the 4B's own index on the dev world with a disjoint training generator; breakthrough about 0%.

**Sources searched today:** [Subtract or Replay?](https://arxiv.org/abs/2607.27539) · [Block-Attention](https://arxiv.org/abs/2409.15355) · [MAC](https://arxiv.org/abs/2403.04317) · [Larimar](https://arxiv.org/abs/2403.11901) · [PICASO](https://arxiv.org/abs/2502.17605) · [MesaNet](https://arxiv.org/abs/2506.05233) · [TOME](https://arxiv.org/abs/2110.06176) · [Cartridges](https://arxiv.org/abs/2506.06266) · [Sleep-time compute](https://arxiv.org/abs/2504.13171) · [Zep](https://arxiv.org/abs/2501.13956)
