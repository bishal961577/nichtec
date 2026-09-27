# Parametric memory modules for LLMs (writable after deployment)

Scope: architectures where new knowledge goes into dedicated memory parameters or state (not the context window, not a text database) and can be written after deployment. Titans is out of scope and covered elsewhere.

Evidence tags used on every bullet:
- **[P]**: read directly in the primary paper's full text. arXiv itself is blocked by the network proxy, so the full text came from the Hugging Face paper mirror (hf://papers/<id>/paper.md, which is the arXiv HTML). The links below point to huggingface.co/papers/<id>.
- **[S]**: secondary. It comes from a search-engine summary of the arXiv abstract/PDF or a vendor page that I could not open directly.
- **[I]**: my own arithmetic or inference from cited numbers. These are not claims made by the papers.

Access caveat: arxiv.org, openreview.net, semanticscholar, aclanthology, alphaxiv, bytez, awesomepapers.io, lamini.ai and GitHub repos outside this session were all blocked. For some older papers the HF mirror dropped table contents (MemoryLLM Table 1, M+ Tables 1–2, PEER Table 1, Memory-Layers-at-Scale Table 2), so those numbers are missing or secondary.

---

## Q1. MemoryLLM (Wang et al., ICML 2024) and M+ (2025): memory pool size, forward-pass injection, retention after N updates, model size

### Takeaway
MemoryLLM adds a ~1B-parameter latent memory pool to Llama-2-7B: 32 layers × 7,680 memory tokens × 4,096 dims. Each write is a single forward pass with no backprop, and the model stayed functional after 650k writes. However, forgetting is exponential by design: each write randomly drops K/N = 256/7,680 of each layer's pool. Knowledge is detectably retained for about 20 updates and in practice fades beyond roughly 20k injected tokens. M+ (Llama-3.1-8B) moves the dropped tokens into a CPU-side long-term store of up to 150k tokens per layer with a co-trained retriever, which extends retention to more than 160k tokens. That store is effectively a latent-vector database, not a growing set of weights.

### Cited Findings
**MemoryLLM: architecture and write mechanism**
- Memory pool θ = hidden-state vectors ("memory tokens") in every transformer layer. Backbone φ = Llama2-7B (32 layers, hidden 4,096). Pool per layer = 7,680 tokens, so θ ∈ ℝ^{32×7680×4096}, which the paper states as "1.066B parameters". The authors call θ the "self-updatable parameters". — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)
  - [I] 32×7,680×4,096 = 1.007B, so the paper's "1.066B" looks like a small arithmetic or rounding slip. Either way it is about 2 GB in fp16.
- Self-update (write), per layer l:
  1. Take the last K memory tokens of θ_l.
  2. Concatenate them with the new text's hidden states h_l and run the frozen/shared layer φ_l.
  3. The last K outputs become the new memory tokens.
  4. Randomly drop K old tokens from θ_l and append the new K.
  Main model: K=256. The update uses only the last K tokens, so write cost does not grow with pool size N. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)
- The paper lists "Efficiency: ... potentially eliminating the need for back-propagation" as a design goal. Gradients are used only during the offline training of the update/read behaviour. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)
- Read path: during generation every token attends to all N memory tokens of each layer, giving an attention map of n_x×(n_x+N). Cost is linear in N, and the authors name this attention as the main scalability constraint. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)

**MemoryLLM: forgetting and retention**
- Forgetting is designed to be exponential. Retention of knowledge injected N/K steps earlier is (1−K/N)^{N/K}, which tends to 1/e. The authors cite the Ebbinghaus curve. Old knowledge is deliberately phased out. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)
  - [I] With N=7,680 and K=256, the upper-bound retention curve is 0.71 after 10 updates, 0.51 after 20, 0.36 after 30, 0.18 after 50 and 0.03 after 100.
- Retention experiments on SQuAD (2,250 samples) and NaturalQA (1,004 samples): "the model retains knowledge even after 20 updates", but the measured curve "falls short of the exponential decay curve" upper bound. Step-1 accuracy (the fact was injected in the latest update) was 0.46 on NaturalQA and 0.39 on SQuAD. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)
- Ablations:
  - Larger N/K gives less forgetting (tested N ∈ {10,20,30}×256, K ∈ {256,512}).
  - K=128 gave much worse step-1 accuracy (0.34 NQA, 0.25 SQuAD).
  - Memory in only 1 layer gave "almost zero improvement".
  - Memory only in the last half of layers gave 0.39 NQA / 0.22 SQuAD. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)

**MemoryLLM: robustness after many updates ("~1M updates")**
- The abstract says "without any sign of performance degradation even after nearly a million memory updates". The body reports "up to 650,000 steps for 3 days" with no decline in accuracy on the most recently injected context (smoothed EMA). This tests that the model keeps working (integrity), not that it retains old facts. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)

**MemoryLLM: training, compute and benchmarks**
- Training took 3 days on 8×A100-80GB on RedPajama C4.
- Inference needs "one 48 GB GPU or two 40GB GPUs ... regardless of the input length".
- Model editing (ZsRE, first 10k records; CounterFact, first 2k) was claimed best overall versus FT, FT-L, IKE and ROME. The table values were not recoverable from the mirror. — [P] [MemoryLLM](https://huggingface.co/papers/2402.04624)

**M+ (Wang et al. 2025, follow-up)**
- M+ states that MemoryLLM "struggles to retain knowledge beyond 20k tokens". M+ extends retention "from under 20k to over 160k tokens with similar GPU memory overhead". — [P] [M+](https://huggingface.co/papers/2502.00592)
- Architecture:
  - Backbone Llama-3.1-8B.
  - Short-term memory N=10,240 tokens per layer plus K₀=2,560 tokens retrieved per layer from long-term memory, for 12,800 memory tokens in total. K=256.
  - Dropped tokens go to long-term memory Θ_l, which is capped at M=150k tokens; the oldest are evicted when full.
  - Retriever = two 2-layer MLP projectors with output dimension d/20, co-trained with the LM. Retrieval happens once per layer for all heads.
  - Two LoRA sets are used, one for writing and one for reading.
  - Long-term memory is stored on CPU. — [P] [M+](https://huggingface.co/papers/2502.00592)
- Training cost:
  - Stage 1: 1,200,000 steps over about 4 weeks on 8 A100s (fineweb-edu).
  - Stage 2: about 1 week on 4k–64k-token SlimPajama documents.
  - Stage 3 adds long-term memory. — [P] [M+](https://huggingface.co/papers/2502.00592)
- Peak GPU memory (MB) on the long-book tasks: M+ 21,177.76; M+ with offload 17,973.34; Llama-3.1-8B-16k 19,239.21; Llama-3.2-3B-128k 30,422.70; Llama-3.1-8B-SnapKV 32,574.49. — [S] (search summary of M+ Table 1) [M+ PDF](https://arxiv.org/pdf/2502.00592)
- Lossy compression and short-context cost:
  - An 8k input is split into 512-token chunks, and the first 6k tokens are compressed into about 2,755.6 memory tokens, with about 316.4 dropped.
  - For a 16k input, 14k tokens are compressed to about 5,530, with about 1,638 dropped.
  - On LongBench (8k/16k), M+ matches Llama-3.1-8B on 4 of 6 datasets but trails on **hotpotqa and musique**, which are both multi-hop. — [P] [M+](https://huggingface.co/papers/2502.00592)
- M+ outperforms MemoryLLM-7B and Llama-3.1-8B-SnapKV (48k prompt, more than 70 GB of GPU memory) on SQuAD/NQ retention with distractor contexts. SnapKV "struggles to recall information injected more than 30k tokens earlier". — [P] [M+](https://huggingface.co/papers/2502.00592)

### Inferences
- [I] MemoryLLM writes knowledge without backprop, but it does not keep it: retention is designed to fall off exponentially, and N/K = 30 means old knowledge is mostly gone after a few dozen writes. That contradicts the "never forget" requirement. M+ keeps more by appending to an ever-growing latent store with retrieval, so its "permanent" part is effectively an external vector store of hidden states, not weights.
- [I] Phone cost: each write is roughly one forward pass of a 7–8B model over one chunk (≤512 tokens) plus 256 memory tokens per layer, with no backward pass. Each generated token must attend to 7,680 (MemoryLLM) or 12,800 (M+) extra memory tokens in every layer, which is comparable to permanently running with an 8–13k-token context. With a 7–8B backbone plus about 2 GB of memory state, this exceeds typical phone RAM budgets. The authors' own inference needed 48 GB of GPU memory.

### Gaps
- MemoryLLM Table 1 editing numbers and M+ LongBench Table 2 numbers were not recoverable because arXiv is blocked and the mirror dropped the tables.
- No MemoryLLM-family result on multi-hop use of injected knowledge beyond LongBench's hotpotqa/musique (where M+ underperforms).
- No 2026 successor from this group (e.g. a phone-scale MemoryLLM) was found.

---

## Q2. Larimar (Das et al., ICML 2024, IBM): one-shot episodic memory; speed vs ROME/MEMIT/GRACE; sequential editing; capacity

### Takeaway
Larimar writes facts in one shot, gradient-free, into a tiny Kanerva-style memory matrix. The main models use 512×768, about 393k floats. The write is a closed-form least-squares (pseudo-inverse) update, and the memory conditions a frozen-ish decoder as a KV prefix. Editing is several times faster than ROME/GRACE on an A100, with ERR 0.97 after 1,000 sequential ZsRE edits. Capacity is effectively about K facts: recall stays near 100% up to 512 facts in a 512-slot memory and falls to 82% at 1,024. It also supports exact deletion of a written fact.

### Cited Findings
**Architecture and write mechanism**
- Larimar-1.3B = BERT-large encoder + GPT2-large decoder. Larimar-6B = BERT-large + GPT-J-6B. Memory matrix = 512×768.
- Training data: 7.6M WikiText chunks of 64 tokens. Larimar-6B was trained for 10 epochs on 8×A100-80GB.
- The readout is projected and "broadcasted to act as a KV cache across all layers" of the decoder.
- WikiText perplexity: 14.6 (1.3B) and 15.9 (6B) on 1,000 samples; the paper describes this as memory "barely affects performance". — [P] [Larimar](https://huggingface.co/papers/2403.11901)
- Write/update: M₀ is a learned prior. A write computes addressing weights W₀ = Z_ξ·M₀† and then posterior M = W₀†·Z_ξ, i.e. a least-squares fit via pseudo-inverse. The update is described as one-shot "without the need for computationally expensive re-training or fine-tuning". — [P] [Larimar](https://huggingface.co/papers/2403.11901)
- Sequential writes and deletes: C_i = C_{i−1} + α_i·W_iᵀW_i and M_i = M_{i−1} + α_i·C_i⁻¹W_iᵀ(Z_i − W_iM_{i−1}). α=+1 writes; α=−1 exactly removes a previously written encoding, keeping M the least-squares solution over the remaining data. — [P] [Larimar](https://huggingface.co/papers/2403.11901)
- Optional scope detector: external MiniLM (384-dim) nearest-neighbour detector with an EER of 2.9% and F1 of 0.974 on 3,800 EasyEdit samples. — [P] [Larimar](https://huggingface.co/papers/2403.11901)

**Speed**
- Wall-clock time "across 10 edits" on one A100 in EasyEdit:

  | Editor | GPT-2 | GPT-J |
  |---|---|---|
  | Larimar | 1.1 s | 1.7 s |
  | ROME | 4.8 s | 13.9 s |
  | GRACE | 13.9 s | 19.3 s |

  The paper summarises this as "4-10x" faster in the text and "8-10x" in the abstract. — [P] [Larimar](https://huggingface.co/papers/2403.11901)

**Single-fact editing (CounterFact, first 2,000 records; values are % S-score)**

| Model | Edit success | Paraphrase | Neighborhood |
|---|---|---|---|
| Larimar-6B | 99.6 | 88.4 | 80.4 |
| ROME on GPT-J | 99.9 | 99.1 | 78.9 |
| FT on GPT-J | 100.0 | 96.6 | 10.3 |
| Larimar-1.3B | 100.0 | 83.5 | 74.7 |

- Writing 2 extra paraphrases into memory lifts Larimar-6B paraphrase generalisation from 88.4 to 93.6.
- On ZsRE, paraphrase generalisation rises from 70.4% to 82.2% with 2 rephrases. — [P] [Larimar](https://huggingface.co/papers/2403.11901)

**Sequential editing (ZsRE, 200 facts × 5 rephrasings)**
- Edit retention rate (mean F1 after 1,000 sequential edits):

  | Method | ERR |
  |---|---|
  | Larimar-1.3B | 0.97 |
  | Larimar-6B | 0.92 |
  | GRACE | 0.93 |
  | MEND | 0.27 |

  These runs used K=1000 (memory size proportional to the number of facts). Larimar surpasses GRACE's held-out generalisation after about 600 edits in a 3,000-edit stream. — [P] [Larimar](https://huggingface.co/papers/2403.11901)

**Capacity (batch editing, CounterFact)**
- Rewrite accuracy is "near 100% for up to 512 edits (eqv. to the memory size K) and then drops to 82% for 1024 edits". This is better than MEND/ROME but worse than MEMIT for very large batches. — [P] [Larimar](https://huggingface.co/papers/2403.11901)
- Selective forgetting (K=512), recall of retained facts after deleting one:

  | Model | N = K (CF / ZsRE) | N = 2K (CF / ZsRE) |
  |---|---|---|
  | Larimar-1.3B | 0.997 / 0.95 | 0.79 / 0.52 |
  | Larimar-6B | 0.993 / 0.86 | 0.71 / 0.50 |

  Recall of the forgotten fact itself falls to 0.0–0.04.
- Input-rephrasing attack success: Larimar 17.6% (single) / 21.5% (batch) versus ROME 29.0% and MEMIT 49.3%. — [P] [Larimar](https://huggingface.co/papers/2403.11901)

**Long context and limitations**
- Recursive memory hierarchy on CNN FastFacts: recall 0.88 (1.3B) at 128 facts versus Mistral-7B 3-shot 0.57. Memory read time 0.36 s versus 1.44 s for Mistral-7B. — [P] [Larimar](https://huggingface.co/papers/2403.11901)
- Stated limitation: "only handle shorter-length facts"; training is "limited to sentence completion tasks". — [P] [Larimar](https://huggingface.co/papers/2403.11901)

### Inferences
- [I] Larimar is the cheapest writer in this survey. A write is one BERT-large encoding plus small matrix algebra on a 512×768 matrix (about 0.8 MB in fp16), and there is no backprop, so it is plausible on a phone NPU/CPU. The memory is a genuine persistent parameter-like state (a matrix M), written by a closed-form update.
- [I] Capacity is roughly linear in K: about K facts at high recall, degrading beyond that. A "never forget, ever-growing" memory would need K to grow, or a hierarchy of many memories, which the paper only uses for long context. Facts are short sentences.
- [I] New knowledge is injected as a decoder KV prefix from the readout, so multi-hop composition of stored facts is not demonstrated.

### Gaps
- No multi-hop (e.g. MQuAKE) result for Larimar was found.
- No test with more than about 3,000 sequential writes.
- No on-device timing.

---

## Q3. Memory³ (Yang et al., 2024): explicit memory as sparse KV; costs and results

### Takeaway
Memory³ is not weight-writing. Its "explicit memory" is sparsified attention key-values precomputed from 128-token reference chunks, stored on disk and retrieved per 64-token chunk. Writing is a single forward pass per chunk, with no training. The data is heavily compressed: 7.17 PB becomes 45.9 TB, or 4.02 TB with vector quantisation, for 1.1×10⁸ references. The trade-off is about 35% lower decode throughput and dependence on an external vector index.

### Cited Findings
**Model and memory format**
- Model: 2.4B non-embedding parameters. Knowledge base: 1.1×10⁸ text chunks of ≤128 tokens. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
- Each explicit memory has tensor shape (memory layers, 2, KV heads, sparse tokens, head dim) = (22, 2, 8, 8, 80).
  - Only the first half of attention layers are memory layers.
  - Only 8 of 128 tokens are kept per KV head, selected by attention weight.
  - Optional residual vector quantisation gives an 80/7 ≈ 11.4× compression rate.
  - Total sparsity is 160× without vector compression and 1,830× with it. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
  - [I] (22·2·8·8·80) = 225,280 values, about 450 KB per 128-token reference in bf16. With vector compression this becomes about 37 KB per reference (4.02 TB / 1.1×10⁸), or roughly 290 B per source token.
- Storage for the full base: 7.17 PB raw becomes 45.9 TB after sparsification and 4.02 TB with vector compression. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
- Write: "the write process involves no training"; each reference is encoded independently. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
- Read: 5 memories are retrieved per 64-token chunk. Retrieval uses BGE-M3 embeddings and a FAISS index, with parallel position encoding. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
- Proposed phone/laptop deployment: the memory bank and vector index stay on a cloud server, and the device stores only model parameters plus the VQ decoder. — [P] [Memory³](https://huggingface.co/papers/2407.01178)

**Results**
- Explicit memory "boosts the average score by 2.51%" (Memory³-SFT 63.31 vs 60.80 without memory). The authors equate this to about 49–51% more "effective model size". With vector compression the score is 63.33, i.e. no degradation at 8.75% of the original KV size. — [P] [Memory³](https://huggingface.co/papers/2407.01178)
- Throughput (tokens/s, with vs without retrieval):

  | Setting | Memory³-2B | MiniCPM-2B + RAG |
  |---|---|---|
  | A800 GPU | 733.0 vs 1,131 (≈35.2% slower) | 501.5 vs 974.0 |
  | Jetson AGX Orin (end-side, remote KB) | 27.6 vs 44.36 | not recorded |

  Reading memories costs 2.884×10⁻³ TFLOPs versus 1.264 TFLOPs for the rest; the authors attribute the slowdown to implementation. — [P] [Memory³](https://huggingface.co/papers/2407.01178)

### Inferences
- [I] For the phone brief this is a KV-cache database, which the brief puts outside "parameters". It is still relevant as a cheap forward-pass writer with a strong compression ratio (about 37 KB per 128 tokens). At that rate 1 GB of flash holds about 27k chunks, or about 3.5M tokens of read text.
- [I] Nothing is ever forgotten because the store is append-only, but retrieval quality (BGE-M3) becomes the bottleneck. There is no mechanism for the model to consolidate facts into weights, although the paper discusses "memory consolidation" as future work.

### Gaps
- No on-device measurement with a local (not remote) knowledge base.
- No multi-hop evaluation of retrieved explicit memories.

---

## Q4. Product-key memory (Lample et al., 2019) and Memory Layers at Scale (Berges et al., Meta 2024): parameter counts, lookup cost, factual QA gains

### Takeaway
Product-key memory layers add up to 128B trainable key-value parameters at negligible FLOPs. Each token touches only top-k of about 1M–64M slots, using O(√N) key comparisons. They give large factual-QA gains: a 1.3B model with 64M keys scores NQ 20.78 / TQA 62.14, versus 7.76 / 32.64 for dense, approaching Llama2-7B. These are true parameters but are written only by gradient descent, which is the substrate that sparse memory finetuning (Q6) exploits.

### Cited Findings
**Large Memory Layers with Product Keys (Lample et al., 2019)**
- The memory adds "up to a billion parameters with a negligible computational overhead".
- Keys are the Cartesian product of two sub-key sets, so exact top-k search costs O(√|K|) comparisons.
- "A memory augmented model with only 12 layers outperforms a baseline transformer model with 24 layers, while being twice faster at inference time" (dataset of up to 30B words). — [P] [PKM](https://huggingface.co/papers/1907.05242)

**Memory Layers at Scale (Berges et al., 2024): design**
- One or more FFN layers are replaced by memory lookups. Memory+ uses 3 memory layers that share one pool, so the parameter count stays the same.
  - More than about 3 memory layers degrades performance.
  - Memory+ adds input-dependent silu gating ("swilu") and qk-norm.
- Base models range from 134M to 8B parameters, with up to 128B memory parameters. Scaling-law models were trained to 1T tokens. — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)
- Lookup is memory-bandwidth-bound. Custom CUDA EmbeddingBag kernels reach 3 TB/s on an H100 (spec 3.35 TB/s), versus less than 400 GB/s with PyTorch. — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)

**Results (Table 1; 2²⁰ ≈ 1M values unless noted; NQ = accuracy, TQA/HotpotQA = F1)**

| Model | Total params | NQ | TQA | HotpotQA |
|---|---|---|---|---|
| 1.3B dense | 1.3B | 7.76 | 32.64 | 13.92 |
| 1.3B MoE | 3.545B | 8.14 | 31.46 | 15.15 |
| 1.3B PEER | 3.646B | 12.33 | 42.46 | 15.39 |
| 1.3B Memory | 3.377B | 9.83 | 39.47 | 15.46 |
| 1.3B Memory+ | 3.377B | 13.68 | 42.89 | 16.72 |
| 1.3B Memory+ 4M keys | 9.823B | 14.43 | 51.18 | 18.59 |
| 1.3B Memory+ 16M keys | 35.618B | 20.14 | 58.67 | 20.65 |
| 1.3B Memory+ 64M keys | 138.748B | 20.78 | 62.14 | 20.47 |
| Llama2-7B (2T tokens) | 7B | 25.10 | 64.00 | 25.00 |

— [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)

- Memory models "generally match the performance of models with twice the number of dense parameters on QA tasks". Memory+ falls between dense models with 2×–4× the compute. — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)
- 8B scale: an 8B base with 4096² = 16M memory values (64B memory parameters) trained to 1T tokens "approaches the performance of Llama3.1 8B, which was trained on 15 trillion tokens". Gains are larger early in training (200B tokens), "suggesting that memory helps models learn facts faster". — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)
- Stated shortcoming: "a substantial engineering task to make them efficient enough for large scale production uses". — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)

### Inferences
- [I] Per-token cost for a 1M×1,024 memory with 4 heads × k=32 (the SMF config) is 128 value rows × 1,024 = 131k parameters read per token, about 256 KB in fp16, plus about 2×1,024 sub-key dot products. FLOPs are trivial. The cost is random-access bandwidth into a pool of about 1.07B parameters (about 2.1 GB fp16, about 1.1 GB int8). On a phone this favours keeping the pool in DRAM or fast flash with caching (compare MeKi's ROM offload in Q7).
- [I] HotpotQA F1 rises with memory size (13.92 to about 20.5). This shows memory-stored pretraining knowledge is usable in multi-hop QA, but it says nothing about facts written after deployment.

### Gaps
- Table 2 (8B) numbers were not recoverable.
- No latency figures on mobile SoCs for product-key lookups were found in these papers.

---

## Q5. PEER (He, 2024, "Mixture of A Million Experts"): mechanism and results

### Takeaway
PEER is a product-key router over more than 1M single-neuron experts, each a (u_i, v_i) rank-1 pair. Per token it assembles a dynamic MLP from h×k retrieved neurons. It beats dense FFN, coarse MoE and PKM at equal FLOPs on C4 perplexity. In Meta's comparison it matches plain memory layers on factual QA. Like other memory layers it is written only by gradient descent.

### Cited Findings
- PEER layer: a pool of N experts, each a single-hidden-neuron MLP with d_expert=1, plus N product keys and h query heads. Each head retrieves top-k experts, and outputs are softmax/sigmoid-weighted and summed. Granularity G = h·k. — [P] [PEER](https://huggingface.co/papers/2407.04153)
- Main config: 1024² ≈ 1.05M experts, h=8 heads, top k=16 per head, query BatchNorm. The PEER layer replaces the middle-block FFN. It was compared in isoFLOP sweeps at 6e18 and 2e19 FLOPs on C4 (batch 128, sequence length 2,048). Given the same compute budget, "a PEER model achieves the lowest compute-optimal perplexity" versus dense, 128-expert expert-choice MoE, and PKM (1024² memories, h=8, k=32). — [P] [PEER](https://huggingface.co/papers/2407.04153)
- Ablations:
  - More total experts (128² to 1024²) at fixed h·k=128 improves perplexity.
  - More active experts (h·k from 32 to 512) helps but saturates and raises memory use.
  - Expert usage is close to 100% at 1M experts; BN improves balance. — [P] [PEER](https://huggingface.co/papers/2407.04153)
- The paper motivates many tiny experts partly by lifelong learning ("adding new experts ... can adapt to continuous data streams"), but it has no continual-learning experiments. — [P] [PEER](https://huggingface.co/papers/2407.04153)
- Independent comparison: at 1.3B base, PEER (3.646B total) scores NQ 12.33 / TQA 42.46 versus Memory+ 13.68 / 42.89. PEER "performs similarly to Memory for the same number of parameters, while lagging behind Memory+". — [P] [Memory Layers at Scale](https://huggingface.co/papers/2412.09764)

### Inferences
- [I] PEER's experts are two d_model-vectors each (down and up embeddings), so a write via sparse finetuning would touch rank-1 slices. That is similar in spirit to SMF's value-row updates, but no one has published sparse continual writes into PEER that I found.

### Gaps
- PEER Table 1 perplexity values were not recoverable.
- No continual-learning or knowledge-injection results with PEER were found.

---

## Q6. Sparse memory finetuning (Lin et al., Meta FAIR / UC Berkeley, arXiv 2510.15103) and 2026 follow-ups

### Takeaway
SMF writes new facts by gradient into only the top-t memory-layer value rows that the new batch uses unusually often compared with pretraining usage (TF-IDF). On a 1.3B model with a 1M-slot memory it learns as much as full finetuning or LoRA while NaturalQuestions F1 drops only 11%, versus 89% for full finetuning and 71% for LoRA. It is the closest existing match to "the model's own parameters change and it answers later without re-reading". However, it is gradient-based, has only been shown at 0.5–1.3B scale for about 1k facts or about 1.8k document chunks, and has not been tested on multi-hop use. The 2026 follow-ups retrofit Qwen-2.5-0.5B and replace TF-IDF with KL-divergence selection. They show small but low-forgetting gains; their exact figures are secondary here.

### Cited Findings
**Mechanism (primary)**
- Base model: 1.3B, 22 layers. The FFN at layer 12 is replaced by a memory lookup:
  - pool of 1M slots, k=32 accesses per token per head, 4 heads, value dimension 1,024;
  - 32×1,024 = 32,768 active parameters versus 50M in the original FFN (model dim 2,048).
  - Memory values are the only parameters trained; the rest of the model is frozen. — [P] [SMF](https://huggingface.co/papers/2510.15103)
- Slot selection:
  1. Count memory-index accesses on the current batch.
  2. Score each index i by TF-IDF: c(i)/Σc(j) · log((|B|+1)/(Σ_b 1[c_b(i)>0]+1)), where the background B is memory accesses on 1,000 random DCLM pretraining batches, stored statically in the checkpoint.
  3. Update only the top-t indices. Gradients are stopped on all other memory parameters; all accessed indices still contribute to the forward pass. — [P] [SMF](https://huggingface.co/papers/2510.15103)
- Scale of sparsity:
  - Each forward pass accesses 10³–10⁶ indices, but t can be much smaller.
  - t=500 for facts; t=10,000 for documents.
  - Typical "core set" per fact is about 100–500 indices, versus 1,024 indices accessed per token (32 × 4 heads). — [P] [SMF](https://huggingface.co/papers/2510.15103)
- Optimiser: SGD for SMF (lr 5 = "high", lr 2 = "low"); AdamW with weight decay 0.1 for baselines. The authors found AdamW's adaptivity, momentum and weight decay "interact with sparsity in unexpected ways". — [P] [SMF](https://huggingface.co/papers/2510.15103)

**Fact learning**
- Stream of 1,000 TriviaQA facts, rewritten as statements, one fact per step, batch = 64 paraphrases, sequence length 64.
- "NaturalQuestions F1 drops by 89% after full finetuning on new facts and 71% with LoRA, sparse memory finetuning yields only an 11% drop with the same level of new knowledge acquisition." SMF also forgets less on HellaSwag and GSM8K NLL. — [P] [SMF](https://huggingface.co/papers/2510.15103)

**Document learning**
- 100 Wikipedia-grounded SimpleQA questions whose source articles were split into 1,824 paragraph chunks.
- One gradient step per chunk, in sequence (not iid), using N Active-Reading synthetic augmentations of the chunk per batch; sequence length 512.
- SMF reaches the same target performance as full finetuning and LoRA "with much less forgetting". — [P] [SMF](https://huggingface.co/papers/2510.15103)

**Ablations**
- The Pareto sweep covered t ∈ {25…1,000} and lr ∈ {0.1, 2} for SMF, versus LoRA ranks 32–256 and full-finetuning learning rates 2e-6 to 5e-5. SMF "Pareto dominates".
- TF-only ranking forgets more, and the gap widens at small t (t=50).
- Naive full memory finetuning forgets badly (GSM8K NLL 3.87).
- Using the learning set as the background corpus increases forgetting. — [P] [SMF](https://huggingface.co/papers/2510.15103)
- Stated limits: only fact-learning tasks, and it needs "to scale ... to more complex tasks beyond fact learning, as well as larger models". — [P] [SMF](https://huggingface.co/papers/2510.15103)

**"Improving Sparse Memory Finetuning" (Goyal, Kanchi, Shah, Gupta; arXiv 2604.05248, 6 Apr 2026)**
- Open-source pipeline that retrofits pretrained Qwen-2.5-0.5B by replacing FFNs with sparse key-value memory layers, followed by a "healing" stage to recover general capability. — [S] [arXiv 2604.05248](https://arxiv.org/abs/2604.05248)
- Proposes KL-divergence slot scoring: the information gain of the current batch's slot-usage distribution relative to background usage, prioritising "surprising" slots. — [S] [arXiv 2604.05248](https://arxiv.org/abs/2604.05248)
- Learns TriviaQA via sparse updates with "higher stability on held-out benchmarks (GSM8k, NaturalQuestions) compared to dense finetuning". The summary describes it as "continual learning on consumer hardware". — [S] [arXiv 2604.05248](https://arxiv.org/abs/2604.05248)

**"Sparse Memory Finetuning as a Low-Forgetting Alternative to LoRA and Full Finetuning" (Gupta, Shah, Goyal, Kanchi; arXiv 2605.03229, 4 May 2026, revised 8 Jun 2026)**
- Setup: Qwen-2.5-0.5B-Instruct; task = MedMCQA (4-choice); forgetting probes = WikiText perplexity and TriviaQA accuracy. — [S] [arXiv 2605.03229](https://arxiv.org/abs/2605.03229)
- Results:
  - SMF improves MedMCQA by 2.5 points while keeping both forgetting probes within about 1 point of the base model.
  - LoRA and full finetuning gain more but "with clear drift on both" probes.
  - KL selection better preserves WikiText fluency; TF-IDF is slightly better at retaining TriviaQA facts. — [S] [arXiv 2605.03229](https://arxiv.org/abs/2605.03229)

### Inferences
- [I] SMF is a permanent write into real weights (memory value rows). It is sparse: t=500 rows × 1,024 dims = about 0.5M parameters (about 1 MB fp16) changed per fact, or about 10M per document chunk at t=10,000. That makes updates cheap to store, diff, back up or roll back.
- [I] The write is not cheap in compute. Each fact step ran a forward pass over 64×64 = 4,096 tokens through the 1.3B model plus backprop from the loss down to layer 12 of 22. That is on the order of 10¹³ FLOPs per fact (≈2·1.3e9·4,096 forward plus a comparable partial backward).
  - At an assumed, unsourced sustained 1–5 TFLOPS of on-device training throughput, that is seconds to tens of seconds per fact.
  - Document learning also requires generating Active-Reading paraphrases with an LLM first.
  - So SMF is feasible on a phone as an overnight or charging-time "consolidation" job, not an instant write.
- [I] Capacity is not characterised. The largest runs are 1,000 facts and 1,824 chunks. With about 100–500 core slots per fact and a 1M-slot pool, slot overlap and interference at 10⁵–10⁶ user facts is an open question.
- [I] "Never forget" is approximated but not guaranteed: NQ F1 still drops 11%, and the 2026 follow-up reports about 1 point of drift. Nothing prevents later writes from overwriting slots used by earlier user facts, because the IDF background protects pretraining usage, not previously written user facts.

### Gaps
- Exact numbers for 2604.05248 (memory size, healing cost, TriviaQA/GSM8K/NQ values) and 2605.03229 (absolute MedMCQA/WikiText/TriviaQA values, hardware) could not be read; arXiv, alphaxiv and other mirrors were blocked.
- The absolute TQA/NQ F1 values for the original SMF are only in figures. The appendix text rendered ambiguously ("TQA 1K F1 >> 0.7, NQ F1 << 0.15"), so those values are not reported here.
- No SMF result on multi-hop use of written facts, on retention across 10⁴+ sequential writes, or on-device.

---

## Q7. LM2, KBLaM, Memory Decoder, Lamini MoME, and 2025–2026 successors / on-device parametric memory

### Takeaway
None of these is a proven "cheap, permanent, growing, never-forgetting" parametric memory:
- **LM2** is a gated recurrent memory used within a sequence for long-context reasoning.
- **KBLaM** stores facts as encoder-produced key-value "knowledge tokens" outside the weights. Writes are forward passes, and it holds 10K+ triples on one A100.
- **Memory Decoder** is a separately trained 0.5B parametric domain memory. Writes need offline training.
- **Lamini MoME** is a vendor claim of millions of LoRA "memory experts" trained to zero loss on facts.
- **MeKi** (Samsung, 2026) is the most phone-relevant work on memory scaling, but its memory is static ROM lookup tables.
- A 2026 study formalises the **"Knowing–Using Gap"**: facts finetuned into weights are memorised but often not used in 2-hop reasoning.

### Cited Findings
**LM2 (Convergence Labs, 2025)**
- Memory bank M ∈ ℝ^{N×d×d} per decoder block. Each slot is initialised as an identity matrix, read by cross-attention and written with LSTM-style input/forget/output gates: M_{t+1} = g_in·tanh(E_mem) + g_forget·M_t. — [P] [LM2](https://huggingface.co/papers/2502.06049)
- Results: BABILong +37.1% over RMT and +86.3% over Llama-3.2 on average, at up to 128K context; MMLU +5.0% over a pretrained vanilla model. — [P] [LM2](https://huggingface.co/papers/2502.06049)

**KBLaM (Microsoft Research, ICLR 2025)**
- Each (name, property, value) triple is encoded by a frozen sentence encoder (OpenAI ada-002, 1,536-dim) plus learned linear adapters into one key-value "knowledge token" per layer.
  - Rectangular attention: prompt tokens attend to knowledge tokens, but knowledge tokens do not attend to each other. Cost is O((M+N)·N), linear in KB size M.
  - A triple is added, updated or removed "by only modifying its corresponding single knowledge token". — [P] [KBLaM](https://huggingface.co/papers/2410.10450)
- Training:
  - Only the adapters/heads are trained (W̃_K, W̃_V, per-layer W̃_Q), on Llama-3-8B-Instruct.
  - Synthetic KB of 45K names / about 135K triples, of which 120K were used for training.
  - 20K iterations at batch 400 on a single A100 80GB.
  - Training KBs had 10–100 triples; the attention scaling constant was C=100. — [P] [KBLaM](https://huggingface.co/papers/2410.10450)
- Results:
  - More than 10K triples in an 8B model with an 8K context window on one A100 80GB; the in-context baseline tops out at about 200 triples due to memory.
  - Comparable to in-context learning on the synthetic KB, including two-entity questions.
  - "Degraded performance" on out-of-distribution Enron data.
  - It refuses when the answer is absent; over-refusal grows with KB size but later than for ICL. — [P] [KBLaM](https://huggingface.co/papers/2410.10450)

**Memory Decoder (Shanghai Jiao Tong / Shanghai AI Lab, 2025)**
- A small transformer decoder is pretrained offline to imitate kNN-LM retrieval distributions (KL + LM loss, β=0.5). At inference its next-token distribution is interpolated with the base LLM's (weight α).
- Base parameters are unchanged. It works with any model sharing the tokenizer. — [P] [Memory Decoder](https://huggingface.co/papers/2508.09874)
- Results:
  - Average perplexity reduction of 6.17 across biomedicine, finance and law.
  - A single 0.5B Memory Decoder takes Qwen2.5-0.5B from average perplexity 14.30 to 4.06 (LoRA of equal parameter count: 7.55) and Qwen2.5-72B from 5.85 to 3.46.
  - Latency overhead is 1.28× versus 1.51× for in-context RAG and 2.17× for kNN-LM.
  - General-task average 69.79 versus 67.45 (base) and 60.84 (DAPT, which forgets). — [P] [Memory Decoder](https://huggingface.co/papers/2508.09874)
- Cost: training used a budget "equivalent to the computational cost of training a 7B parameter model for 1 epoch" on 8×A800. Cross-vocabulary transfer to Llama needed 10% of that budget. — [P] [Memory Decoder](https://huggingface.co/papers/2508.09874)

**Lamini Memory Tuning / MoME**
- The arXiv paper (Li et al., 2024) shows "LLMs augmented with a massive Mixture of Memory Experts (MoME) can easily memorize large datasets of random numbers". It argues hallucination arises when training loss stays above a threshold, and describes Lamini-1, which "stores facts in a massive mixture of millions of memory experts that are retrieved dynamically". — [P, abstract only] [Lamini arXiv 2406.17642](https://huggingface.co/papers/2406.17642)
- Vendor claims: "tuning millions of expert adapters (e.g. LoRAs) with precise facts on top of any open-source LLM", optimising "for zero error on specific facts". For "one Fortune 500 customer": 95% accuracy versus 50% with other approaches, with hallucinations reduced from 50% to 5%. — [S] [Lamini blog](https://www.lamini.ai/blog/lamini-memory-tuning)

**2025–2026 successors and on-device work**
- **MeKi** (Samsung Research, Feb 2026):
  - Per-layer token-level "memory experts" are re-parameterised into a static lookup table offloaded to ROM, with "zero inference latency overhead".
  - 1.7B-MeKi averages 59.7 zero-shot versus 60.5 for a 4B dense model, with a 2.26× decoding-speed advantage on a Qualcomm Snapdragon 8 Elite NPU (KV length 10K).
  - The knowledge is "pre-stored", i.e. fixed at training time. — [P] [MeKi](https://huggingface.co/papers/2602.03359)
- **Memory Bank Compression, MBC** (SAC '26): codebook compression of a continual-learning memory bank plus KV-LoRA. It reduces memory-bank size "to 0.3%" of the most competitive baseline while "maintaining high retention accuracy" on QA streams. — [P, abstract] [MBC](https://huggingface.co/papers/2601.00756)
- **Active Reading** (Meta FAIR, 2025; the augmentation used by SMF):
  - 8B expert models reach 66% on a Wikipedia-grounded SimpleQA subset (+313% relative over vanilla finetuning) and 26% on FinanceBench (+160%).
  - Meta WikiExpert-8B was trained on 1T generated tokens. — [P] [Active Reading](https://huggingface.co/papers/2508.09494)
- **Knowing–Using Gap** (HKUST, Jul 2026):
  - Finetuned LLMs "can quickly memorize new facts, yet fail to use them for downstream reasoning". There is an accuracy gap and a temporal lag on chaining and intersection 2-hop tasks.
  - The authors attribute this to "knowledge-circuit misalignment". A heuristic recovers 58–75% of the oracle headroom. — [P, abstract/intro] [Knowing–Using Gap](https://huggingface.co/papers/2607.08393)
- "Forget to Improve: On-Device LLM-Agent Continual Learning via Budget-Curated Memory" (arXiv 2606.25115) appeared in search, but only the title was seen; it is unknown whether its memory is parametric. — [S] [arXiv 2606.25115](https://arxiv.org/html/2606.25115)

### Inferences
- [I] KBLaM and Memory³ are the cheapest writers: one encoder or forward pass per fact or chunk. Both are external vector stores that the attention reads, so they fall outside the brief's "in parameters" requirement.
  - KBLaM's per-triple storage could be as small as the 1,536-dim base embedding (about 3 KB fp16), with per-layer keys/values produced by adapters on load.
  - Precomputing per-layer knowledge tokens for Llama-3-8B would instead cost about 128 KB per triple (32 layers × 2 × 8 KV heads × 128 × 2 B).
- [I] Memory Decoder and MeKi show that small parametric memories can hold domain knowledge efficiently, and that mobile SoCs tolerate ROM-offloaded lookup tables at zero latency cost. That is a strong hint that a large product-key memory stored in flash is hardware-feasible on phones. Neither supports cheap incremental writes.

### Gaps
- Not found in accessible sources: LM2 base-model and memory-module parameter counts, whether LM2 memory persists across sequences, and KBLaM's exact accuracy numbers at 10K triples.
- No independent evaluation of Lamini MoME claims was found.
- "SPRInG" (2026 continual LLM personalisation) was mentioned only in a search summary and was not verified.
- No 2026 paper was found demonstrating per-user writable parametric memory running on a phone.

---

## Q8. Cross-cutting: permanence, write cost on a phone, capacity before degradation, multi-hop use

### Takeaway
No published design meets all four requirements: gradient-free writes, true parameters, unbounded capacity without forgetting, and multi-hop use of written facts.
- **Cheap forward-pass writes** (MemoryLLM, Larimar, KBLaM, Memory³) either decay by design (MemoryLLM), cap out at about the memory size (Larimar, about K facts), or are really external vector stores (Memory³, KBLaM, M+ long-term memory).
- **True parametric writes with low forgetting** (sparse memory finetuning on product-key memory layers) need backprop, have been shown only for about 1k facts at 0.5–1.3B scale, and still lose about 11% on held-out QA.
- **Multi-hop use** of newly written knowledge is largely untested, and the 2026 Knowing–Using-Gap work shows finetuned facts often are not used in 2-hop reasoning.

### Cited Findings (summary table; numbers from the sections above)

| Design | What stores knowledge | Write op | Truly parametric & persistent? | Capacity / forgetting evidence | Multi-hop evidence |
|---|---|---|---|---|---|
| MemoryLLM (Llama2-7B) | 32×7,680×4,096 latent memory tokens (~1B) | Forward pass, K=256 tokens/layer, no backprop [P] | Persistent state, yes; permanent, no: random drop K/N per write [P] | Retains after ~20 updates, under the exponential bound; functional after 650k writes [P] | Not tested |
| M+ (Llama-3.1-8B) | 10,240 short-term + ≤150k long-term tokens/layer (CPU) + retriever | Forward pass [P] | Long-term memory is a latent-vector store [P] | Retention >160k tokens [P] | Trails Llama-3.1-8B on hotpotqa/musique [P] |
| Larimar (1.3B/6B) | 512×768 memory matrix | Closed-form least squares, one-shot; supports deletion [P] | Yes (matrix state) | ~100% up to 512 facts, 82% at 1,024; ERR 0.97 after 1,000 sequential edits [P] | Not tested |
| Memory³ (2.4B) | Sparse KV per 128-token chunk on disk (≈37 KB compressed [I]) | Forward pass [P] | No: external KV database + BGE-M3/FAISS retrieval [P] | Append-only; 1.1×10⁸ chunks = 4.02 TB [P] | Not tested |
| Memory layers / PKM / PEER | 1M–64M trainable key-value slots (up to 128B params) | Gradient only (pretraining) [P] | Yes | Factual QA scales with slots (NQ 7.76→20.78 at 1.3B) [P] | HotpotQA F1 13.92→20.47 (pretraining knowledge) [P] |
| Sparse memory finetuning | Top-t memory value rows (t=500 per fact) | SGD on selected rows only; needs backprop [P] | Yes | 1k facts / 1,824 chunks; NQ F1 −11% vs −89% FT / −71% LoRA [P] | Not tested |
| SMF 2026 follow-ups (Qwen-2.5-0.5B retrofit) | Retrofitted memory layers | SGD + KL/TF-IDF slot selection [S] | Yes | MedMCQA +2.5 pp with ≤~1 pt drift [S] | Not tested |
| KBLaM (Llama-3-8B) | 1 knowledge token per triple (external) | Encoder forward pass [P] | No: KB outside weights; only adapters trained [P] | 10K+ triples on 1 A100; OOD degradation [P] | Two-entity questions comparable to ICL [P] |
| Memory Decoder | Separate 0.5B decoder | Offline distillation training (7B-epoch-equivalent budget) [P] | Yes, but batch/offline | Domain perplexity average −6.17 [P] | Not tested |
| MeKi (Samsung) | Per-layer lookup tables in ROM | Training-time only [P] | Static | 1.7B ≈ 4B dense at 2.26× speed on Snapdragon 8 Elite [P] | Not tested |
| Lamini MoME | Millions of LoRA memory experts | Gradient to ~zero loss per fact [S] | Yes (vendor) | Vendor: 95% accuracy, hallucinations 50%→5% [S] | Not shown |

### Inferences
- [I] Hybrid implied by the evidence for the phone design:
  1. A **fast episodic buffer** with forward-pass or closed-form writes: Larimar-style or MemoryLLM-style, or Memory³/KBLaM-style KV entries on flash. Writes happen instantly when the user teaches something or reads a page.
  2. A **slow consolidation step** that runs sparse memory finetuning into a large product-key memory layer, with TF-IDF/KL slot selection, during idle or charging time. This mirrors Larimar's own complementary-learning-systems framing and Memory³'s proposed "memory consolidation".
  - Protecting earlier user facts would need an IDF background that includes the device's own prior writes, not only DCLM. No paper has tested this.
- [I] Storage budget: a 1M-slot × 1,024-dim memory layer is about 1.1 GB at int8. The Memory-Layers-at-Scale results suggest factual capacity keeps rising to 16M–64M slots (tens of GB), which a phone could only hold in flash with MeKi-style offload.
- [I] "Never forget" is not demonstrated by any design here. Every gradient method still shows measurable drift, the forward-pass methods either decay (MemoryLLM) or saturate (Larimar beyond K), and the only no-forgetting stores are append-only external stores.

### Gaps
- No study found measures multi-hop reasoning over facts written after deployment into memory layers, MemoryLLM or Larimar.
- No study found measures write latency or energy for any of these methods on a phone NPU.
- No study found characterises capacity for SMF-style writes beyond about 10³ facts.
