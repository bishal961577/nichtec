## Research scientist, Round 1

Labels: **[checked]** means I verified it this session, in the repo or a primary source. **[recall]** means it comes from memory.

### 1. Audit: is the work done well?

The process is better than most published memory papers: lines registered before the run, amendments written before any model call, a recall check, a hand audit, and recorded forecasts. The construct, the baseline and a few claims have real problems.

**What the 4B numbers show and don't show**
- 99.9% on perfect pages shows the 4B with thinking on can compose, resolve updates and ignore lures inside 9–44 notes. Reading is not the bottleneck at that size.
- It does not show that a selector could reach that ceiling. A "perfect page" is the group's own notes, so a described-person or "who" question has about two candidates. The oracle removes the distractors, not just the search. A fairer ceiling would be the needed notes plus about 20 hard neighbours.
- The search2 score of 50.2% is mostly BM25's recall, which was known before any model call: 0% for described people and 0.5% for two-hop. 440 of the 695 hard questions are kinds where BM25 recall is 45% or less. The mix of question kinds the team chose produces the headline number.
- The phone line was therefore close to decided by construction. It tests word search, not search.

**Biggest mistakes, mine to name, including the assistant that ran it**
1. **A strawman selector.** [checked, `brain/notebook_read.py` and `answers.jsonl`] The follow-up query was written with thinking OFF, greedy, 64 tokens, behind a "DONE" gate, while the answer had thinking with a 16K cap. The 4B said DONE on 117 of 120 described-person questions and 59 of 60 "who" questions, so search2 there was plain BM25.
   - Not tested: dense or hybrid search, a reranker, multi-round search with thinking, an attribute filter, an entity index.
   - The verdict labels "SELECTION" and "phone line FAILS" are broader than what was measured.
2. **"Flat with size" compares different question sets.** [checked] The 95.9 / 95.7 / 95.3% figures put 635 name-keyed questions at 100 and 1,000 notes against 995 questions at 5,000. On the same 635 questions the Haiku scores are 97.5 (pages) / 95.9 / 95.7 / 93.4. The hard kinds were never run at the middle sizes.
3. **A token estimate off by about 2x** (53K vs 100.5K Claude tokens; LongMemEval histories 181K, not 115K). The budget ran out, so the 28 missing full-history questions are not random (21 of 25 timing questions). "Full = oracle 1.00" rests on a biased subset, judged by Haiku grading its own answers.
4. **No 4B reasoning saved.** [checked] `think_chars` is 0 in all 2,718 answers. The code reads `reasoning_content`; my guess is that vLLM 0.33 returns `reasoning` [recall]. Scores are unaffected, but no 4B reasoning can be audited.
5. **Smaller issues:** the retraction scorer bug (4.3%, caught by the audit); the Part B blanking bug; a latency line checked with thinking off while accuracy was scored with thinking on.
6. **Construct validity.** The world is a 9-column table with closed value lists, written as 12-token template sentences. That is the best case for reading the whole notebook and the worst case for word search.
   - It is no test of road B at all: writing profiles from templates is trivial.
   - Road A's 60K-token premise already assumes something at write time has condensed chat into notes. So "A vs B" is partly a false choice. The real question is how much work happens at write time.

**Statistics.** The gaps are about 50 points, so the bootstrap intervals don't matter. Per-kind n of 60 gives about ±12 points. There was one sample per question at temperature 1.0.

### 2. The token worry (road A)

- **Re-reading is not the cost.** Prefix caching (vLLM, SGLang; prompt caching billed at 0.1x [recall]) means the notebook is not re-read every turn. The preflight measured 4.7 s cold vs 0.55 s warm [checked].
- **The real costs on a phone:**
  - **Decode.** Every generated token reads the whole cache. I confirmed 32 KB per token [checked, HF config: 8 full-attention layers, 4 KV heads, 256 dims]. At 60K tokens that is about 2 GB of cache plus about 2.5 GB of weights read per token. On LPDDR5X (60–85 GB/s [recall]) the ceiling is roughly 15–20 tokens/s, so a thinking answer of 600–1,500 tokens takes 30–100 s.
  - **Memory.** 500K tokens is 16 GB at 16-bit, and it is beyond the 262K native context anyway. (The linear-layer state is about 50 MB at the config's float32, not 25 MB; still trivial.)
  - **Deletion.** The delta-rule state cannot subtract a note. Deleting it means replaying everything after a checkpoint. Chunk-reuse methods (CacheBlend, EPIC [recall]) only help the attention layers.
- **Compressing the cache is risky.** SCBench [checked, arXiv 2412.10319] finds that sub-O(n) memory methods degrade in multi-turn shared-context use, which is exactly road A's use. Layer-level sparsity of the kind Qwen3.5 uses holds up.
- **Long contexts degrade, though less in newer models.** RULER, BABILong and NoLiMa [recall]; LongMemEval reports a 30% drop [checked]. Haiku showed no drop here. The pending 4B run is the measurement that matters.
- **A published middle path.** Mastra's "observational memory" [checked, vendor-reported] keeps a compressed, append-only observation log in context, which keeps the cache valid. It reports 84.2% on LongMemEval with gpt-4o, against 60.2% full context and 82.4% oracle.

### 3. The association idea (road B): what is published

- **HippoRAG 2** [checked, 2502.14802, F1 with Llama-3.3-70B]: average 59.8 vs 57.0 for the NV-Embed-v2 dense retriever.
  - GraphRAG 49.6, RAPTOR 48.8 and LightRAG 6.6 all fall below plain dense retrieval.
  - Indexing about 11.7K passages costs 9.2M input tokens, against 115.5M for GraphRAG.
- **LoCoMo** [checked, Mem0 paper, LLM-judge %]: full context 72.9, Mem0 graph variant 68.4, Mem0 66.9, Zep 66.0, A-Mem 48.4. Full context wins; its "90% token savings" is measured against full context without caching.
- **Zep** [checked] on LongMemEval-S: 71.2% vs 60.2% full context with gpt-4o. Its single-session-assistant score falls from 94.6 to 80.4, which is detail lost at extraction.
- **Letta** [search summary, vendor blog]: a plain filesystem-search agent scored 74.0% on LoCoMo, ahead of Mem0's graph variant.
- **MemoryAgentBench** [checked, 2507.05257]:
  - accurate retrieval: HippoRAG-v2 65.1, BM25 60.5, GPT-4o long-context 58.1, Zep 37.5, Mem0 32.6;
  - long context wins test-time learning and long-range understanding;
  - multi-hop conflict resolution is 7% or less for every method.
- **A-MEM** [checked, abstract] is almost exactly the owner's idea: links written at write time, and old notes revised as new ones arrive.

**Known failures of B:** detail lost at extraction is permanent; entity-resolution errors (Joao vs Joaquin) become permanent wrong links; updates and conflicts; questions nobody anticipated at write time (a description needs an attribute index, not person-to-person links).

**What would be new if this team did it:**
- a 4B model doing both the linking and the reading on the phone, measured on the same questions against its own whole-cache reading and against a fair search agent;
- verifiable deletion that reaches every profile or summary derived from a deleted note;
- a less-templated benchmark of described people and near-identical names.

### 4. Vote

**(i) Next test:** a fair selector bake-off, B vs C, with A as the reference. Ranking: B ≈ C > A-at-scale > D > E. Confidence 60%.

**(ii) Long-term phone memory:** D > B > C > A > E. Confidence 65% for D: a recent working set held in the cache, plus write-time pages that link back to the raw notes, read by search with thinking on. I am about 90% confident that A alone cannot carry 5–6 years: the 262K limit, decode cost growing with size, and deletion.

**Conditional on the pending 4B whole-notebook result.** My forecast is about 72%, with an 80% interval of 50–90.
- **If ≥ 85%:** A is viable as the working set for roughly the first year. Add arms that find where it breaks (15K notes, about 180K tokens) and measure on-device latency.
- **If ≤ 65%:** A on the phone shrinks to a small recent window, and selection decides the phone road.
- **In between:** find the knee in the curve with 1K and 2K-note notebooks.

### 5. The one next test

**Idea.** Same Qwen3.5-4B, thinking on, same hard set, on two notebooks: 5K notes and 15K notes. Run it on the clean notes and on a 4B-paraphrased chat version, where each line is checked automatically to keep its value.

**Arms:**
- A: whole cached notebook;
- C1: BM25 + embedding search fused, then reranked, top 20;
- C2: a search agent with thinking on, up to 4 rounds;
- B: the 4B builds person pages, links and an attribute index as notes arrive, with provenance back to the raw notes;
- B without links (to isolate the association effect);
- HippoRAG 2 run locally.

**Baselines it must beat:** perfect pages (99.9%), search2 (50.2%), C1 and A.

**Pass line:** the best non-oracle arm reaches ≥ 90% on the hard set at 5K and is no more than 3 points below A. B counts as "association adds value" only if it beats C2 by ≥ 5 points on two-hop, described and "who" questions together (paired test, n = 380).

**Stop line:** B and C2 both below 70%. Then the 4B cannot drive its own retrieval: stop building B on the 4B and move to a bigger model in the loop or a trained retriever.

**Cost.** About 10–25 L40S-hours, roughly $20–50 at about $1.95/h [recall].

**What each outcome decides:**
- B passes and beats C2: build the owner's idea as D's long-term layer.
- C2 passes and B adds nothing: drop the links and build simpler search.
- A holds at 15K while the selectors fail: A plus consolidation for the early years.
- Everything fails: the phone cannot do this alone yet.

Also measure deletion: delete 50 notes and check no value survives in B's pages, and time A's replay.

Sources:
- [Mem0, arXiv 2504.19413](https://arxiv.org/html/2504.19413)
- [Zep, arXiv 2501.13956](https://arxiv.org/html/2501.13956)
- [HippoRAG 2, arXiv 2502.14802](https://arxiv.org/html/2502.14802)
- [MemoryAgentBench, arXiv 2507.05257](https://arxiv.org/html/2507.05257v2)
- [SCBench, arXiv 2412.10319](https://arxiv.org/abs/2412.10319)
- [LongMemEval, arXiv 2410.10813](https://arxiv.org/abs/2410.10813)
- [A-MEM, arXiv 2502.12110](https://arxiv.org/abs/2502.12110)
- [Letta blog](https://www.letta.com/blog/benchmarking-ai-agent-memory)
- [Mastra observational memory](https://mastra.ai/research/observational-memory)
- [Qwen3.5-4B config](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json)

Files checked (none modified): `/home/user/nichtec/brain/notebook_read.py`, `/home/user/nichtec/results/notebook_read_Qwen3.5-4B/answers.jsonl`, `/home/user/nichtec/results/notebook_read_claude-haiku-5-5/answers.jsonl`, `/home/user/nichtec/reports/Notebook reading at shipping size 2026.md`.
