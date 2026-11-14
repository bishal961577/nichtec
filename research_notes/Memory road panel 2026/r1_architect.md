**Architecture expert, round 1.** [C] means I checked it this session, in the repo, the Qwen3.5-4B config, a scratch script or a source. [M] means from memory, not re-checked. I changed no repo files.

## 1. Audit
The process is sound: the test was pre-registered, predictions were recorded and the scorer was audited. My recount of the hard set matches: 99.9% for perfect pages and 50.2% for search2, n=695 [C]. The conclusions drawn from it are weaker than the test design.

- **Search2 on the 4B mostly never searched.** The model writes its follow-up query with thinking off (greedy, 64 tokens).
  - It replied DONE to 97.5% of described-person questions, 98% of "who is" questions and 52.5% of two-hop questions [C].
  - On two-hop, it scored 53.7% when it did search and 1% when it said DONE [C].
  - So 50.2% measures an overconfident 4B with thinking off, plus an index of one-fact notes. It does not measure "search on a phone".
- **"Word search can't reach a description" is true only for that index.** I grouped the same notes into one page per person and ran the same BM25 over the pages [C, scratch script]. This is trivial here because every note carries the full name.
  - The right person came first for 97.5–100% of described, who-is, first-name and two-hop questions, and was in the top 3 every time.
  - The asker's page named the hidden middle person in 100% of two-hop questions.
  - The index failed, not search.
- **Perfect pages hold only two people (the question's group).** A described person is a choice between two, and the two-hop middle person is simply "the other one".
  - 99.9% shows the 4B reasons flawlessly over about 200 clean tokens: updates told only by order, retractions, lures and two-step links. Reasoning is not the bottleneck.
  - It does not show that the 4B can pick one person out of 840, read 60K tokens, handle real chat (pronouns, "my sister"), or run quantized on a phone.
  - The world is 420 separate pairs, so road B and the perfect-pages oracle are the same thing in it. This test cannot judge B.
- **The deciding run (4B reading the whole notebook) has no pass line.** Fix it before the result arrives (section 4).
- **Cost was not scored.** The 4B's median output was 688 tokens on perfect pages and 1,454 on search2 [C].

**Am I confident?** In the measurements, yes. In "search failed, therefore road A", no.

## 2. The token worry
The moderator's numbers hold [C, config]:
- Per-token cache: 8 attention layers × 4 KV heads × 256 dims = 32 KB per token. The 60K-token notebook is about 2 GB at 16-bit and about 1 GB at 8-bit.
- The fixed state of the 24 DeltaNet layers is about 25 MB at 16-bit (50 MB if kept at 32-bit).
- About 4.2B parameters, so about 2.5 GB at 4-bit.

What each part costs:
- **Done once: reading the notebook.** It took 4.7 s on the L40S [C]. On a phone NPU it is perhaps 1–3 minutes [M, unmeasured]. The result is saved to flash storage and reloads in about a second.
- **Each new note:** only its ~12 tokens, as long as it is added at the end. Deleting or editing an old note means re-reading everything after it. That is fine as a nightly batch, and it makes deletion exact.
- **Each message:** the notebook is not re-read. But every word the model writes has to pull the weights plus the whole cache through memory.
  - At roughly 50–80 GB/s of phone memory bandwidth [M], that is about 10–15 words per second instead of 20–25.
  - The real bill is thinking: 700–1,500 tokens per answer, so 1–2 minutes per answer on a phone.
  - Thinking can't simply be turned off: without it, Haiku fell to 28.8% on the full notebook.
  - So the owner is right that it is expensive, but for the wrong reason. The cost is thinking length times cache size, not re-reading.
- **Hard ceilings:**
  - 5–6 years of notes is about 480K tokens (the moderator's arithmetic is right). That is past the model's 262K native context.
  - The cache would be about 16 GB at 16-bit (4 GB even at 4-bit), and all of it is read for every word written.
  - Exact recall happens only in the 8 attention layers. The fixed DeltaNet state is a lossy summary, not a store.
  - Road A works as a working set of 1–2 years, not as a lifetime memory.

## 3. The association idea (road B)
**Mechanism.** When a note arrives, work out who it is about, update that person's page (current value, history, which notes it came from) and add typed links to other people. When asked, find the pages by name or description, add the pages one link away, and answer from a few thousand tokens.

**Prior art:**
- HippoRAG 2 spreads activation (PageRank) out from the question's people and things. About 7 points above strong embedding search on multi-hop questions [C, secondary source].
- A-MEM is closest to the owner's idea: the model writes the links and updates neighbouring notes. LoCoMo multi-hop F1 45.9 vs 25.5 for MemGPT [C, secondary].
- Zep/Graphiti marks old links as no longer valid instead of overwriting them. LongMemEval 71.2% vs 60.2% for full context with gpt-4o [C, vendor's own number].
- Mem0's graph mode: LoCoMo 68.4 vs 72.9 for full context, so below full context, but with about 90% fewer tokens [C, secondary].
- GraphRAG is costly to update [M]. Letta's core memory is in effect an editable profile page [M].
- ProGraph (June 2026) uses profiles plus following names between them. It claims parity with full context, but on its own benchmark, and is single-author [C, abstract only].

Graph memory's accuracy edge came from weak 2024 long-context models. Here Haiku reading the full history equalled reading only the evidence (90.2%). B's real advantage is cost.

**Where B breaks:**
- **Ambiguous names:** merging two Miras by mistake when a note is written is permanent, and the answering model trusts the page.
- **Updates and deletions:** they must also reach text on *other* pages, such as "Kofi's sister, the piano teacher". Without a source note recorded for every sentence, a deleted fact survives as a paraphrase, and there is no deletion receipt.
- **Described people:** descriptions that use different words need embedding search.
- **Two-hop:** relations written from the other person's side are exactly what extraction misses.
- **Years:** a partner's page will hold thousands of facts and link to everyone, so the search problem comes back inside the pages.
- **Writing cost on a phone:** one 4B call per note at 500–1,500 tokens is 15–30 minutes a day, run overnight. B moves the token bill from asking to writing; it does not remove it.

**Rule:** the raw notes stay the source of truth, and the pages are an index that can be rebuilt from them.

## 4. Vote
- **(i) Next test:** B, with pages written by the 4B itself, against A on a messier notebook. Confidence 60%. After that, a fixed C (thinking on, several rounds), then E.
- **(ii) Long-term phone memory:** D, confidence 65%.
  - A raw log is the truth, and B's pages are the index.
  - An A-style cache holds recent notes and the pages of the people in the current chat.
  - E only for style and skills, never for facts that must be deletable.

**Conditional vote on the pending 4B whole-notebook run:**
- If the hard set is **at least 90% (X)** and median thinking is at most 1,500 tokens: A becomes the phone's working-set engine, up to about 60K tokens.
- If it is **at most 75% (Y)**: A is out at 5,000 notes on a phone, and B is the road. The pending 100-note runs then show how large a working set A can hold.
- In between: A only as a small working set.
- My forecast: about 75%.

## 5. Next test: the Mira test
**Idea.** A messier notebook generated by code:
- about 300 people, each linked to 3–6 others;
- nicknames, pronouns and "my sister";
- people with the same name inside one circle;
- updates written as plain restatements, plus 50 deletions;
- about 5,000 notes.

**Arms on the 4B:**
- A: the whole notebook;
- B: pages the 4B writes itself, each sentence tagged with its source note;
- oracle: pages built by code.

**Measures:** accuracy, wrong merges of two people, deleted facts that still leak into answers, tokens spent writing, and tokens spent answering.

**Pass line:** B reaches at least 0.9 × oracle, with zero leaks and at most 1,000 writing tokens per note. That decides the road: B on the phone, plus an A working set.

**Stop line:** B is below 0.75 × oracle, or wrong merges exceed 3%. The 4B can't be its own librarian. Then either a bigger model builds the pages overnight, or we rely on A.

**Cost:** about 4 L40S-hours, under $10 of Modal credit [M on the price]. Scoring is by code, so no paid API is needed.

**Sources:**
- https://blog.getzep.com/state-of-the-art-agent-memory/
- https://arxiv.org/abs/2504.19413
- https://arxiv.org/abs/2502.14802
- https://arxiv.org/pdf/2502.12110
- https://arxiv.org/abs/2607.19359
- https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json
