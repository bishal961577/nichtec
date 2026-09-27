# Test-time learning architectures and amortized "weight writers" (hypernetworks that turn a document into a weight update in one forward pass)

Evidence tags used throughout:
- **[P]** = verified from the primary paper's own text in this session (arXiv full text read through the Hugging Face paper mirror, `hf://papers/<id>/paper.md`; arxiv.org itself was blocked by the proxy).
- **[A]** = verified from the primary paper's abstract only.
- **[S]** = secondary source (blog, news, or search snippet), not checked against the paper.
- Links point to the Hugging Face paper page (huggingface.co/papers/<arXiv id>), which mirrors the arXiv paper.

Overall picture (details and citations below): **test-time-learning architectures** (Titans, ATLAS, TTT layers, DeltaNet/Gated DeltaNet, Longhorn, TTT-E2E, HOPE) update "fast weights" cheaply inside a sequence, but those weights are **reset for each sequence or document**. They are long-context mechanisms, not permanent memory. **Amortized weight writers** (Generative Adapter, Text-to-LoRA, Drag-and-Drop, Doc-to-LoRA, SHINE) do produce a stored artifact (a LoRA or adapter state) in about one forward pass, with sub-second latency and under 2–4 GB of update memory on a GPU. But they recover only about 65–90% of in-context accuracy, lose to strong pretrained priors when a document contradicts them, and have almost no published evidence for accumulating hundreds of writes without interference.

---

## 1. Titans (Behrouz, Zhong, Mirrokni; arXiv 2501.00663): mechanism, results, and what persists

### Takeaway
Titans adds a deep MLP "neural long-term memory". At test time it is trained online by gradient descent on an associative key→value loss, using momentum ("past surprise"), a data-dependent step size ("momentary surprise") and a weight-decay forget gate. It beats Transformers and linear RNNs at 340M–760M scale and scales past 2M-token contexts on BABILong. The memory is **within-context parametric memory**. Its authors' own follow-up (Nested Learning) says knowledge gained this way "does not persist once the current context is removed". Only the memory's initial state, which is meta-learned in pretraining, carries across contexts.

### Cited Findings
- **Update rule [P]:** the memory is updated as M_t = (1 − α_t)·M_{t−1} + S_t, with S_t = η_t·S_{t−1} − θ_t·∇ℓ(M_{t−1}; x_t). η_t is a data-dependent "surprise decay" (momentum), θ_t scales the "momentary surprise" (the gradient), and α_t ∈ [0,1] is a forget gate. α_t→1 clears the whole memory; α_t→0 leaves the past untouched. The paper describes this as "gradient descent with momentum and weight decay". — [Titans](https://huggingface.co/papers/2501.00663)
- **Loss [P]:** the loss is ℓ(M_{t−1}; x_t) = ‖M_{t−1}(k_t) − v_t‖², with k_t = x_t·W_K and v_t = x_t·W_V. W_K and W_V are outer-loop (pretrained) parameters, and only M's weights change in the inner loop. Retrieval is a plain forward pass with no weight update: y_t = M*(q_t). The memory is an MLP with L_M ≥ 1 layers. Deeper memory gave better perplexity at all sequence lengths but trained linearly slower. — [Titans](https://huggingface.co/papers/2501.00663)
- **"Surprise" [P]:** surprise is defined as the gradient of the memory's loss with respect to the input: "The larger the gradient is, the more different the input data is from the past data." Momentum was added because a pure-gradient surprise can "become extremely small after several surprising steps", so the memory misses what follows a surprising event. — [Titans](https://huggingface.co/papers/2501.00663)
- **Persistent memory [P]:** Titans' "persistent memory" is a set of learnable but **data-independent** parameters that store task meta-knowledge. It is fixed at test time and is not a per-user store. — [Titans](https://huggingface.co/papers/2501.00663)
- **Setup [P]:** Titans was trained at 340M, 400M and 760M parameters (Table 1) with the Llama-2 tokenizer (32K vocab), a 4K training length, AdamW at lr 4e-4 with cosine schedule, 0.5M-token batches and weight decay 0.1. — [Titans](https://huggingface.co/papers/2501.00663)
- **Table 1 rows [P, block size inferred]:** one block of Table 1 reads, in the order Wiki ppl / LMB ppl / LMB acc / PIQA / Hella / Wino / ARC-e / ARC-c / SIQA / BoolQ / Avg:
  - Titans (MAC) 19.93 / 20.12 / 39.62 / 70.46 / 49.01 / 53.18 / 67.86 / 36.01 / 41.87 / 62.05 / **52.51**
  - Titans (MAG) 18.61 / 19.86 / … / Avg **52.50**
  - Gated DeltaNet 21.18 / 22.09 / … / Avg 49.69
  - Mamba2 22.94 / 28.37 / … / Avg 48.34
  - TTT 24.17 / 23.51 / … / Avg 47.32
  - Gated DeltaNet-H2 19.88 / 20.83 / … / Avg 51.49

  The parsed text does not label which model size this block is. The perplexity levels are consistent with the largest (760M) block. — [Titans](https://huggingface.co/papers/2501.00663)
- **Needle-in-a-haystack [P]:** Titans was evaluated on RULER's single-needle test (S-NIAH) at 2K, 4K, 8K and 16K tokens. The neural memory beat TTT, Mamba2 and DeltaNet, and MAC was the best variant. The authors credit momentum plus weight decay for capacity management: "Mamba2 is not capable of removing a memory" and "DeltaNet… cannot erase the memory". The exact numbers are in a table that did not render in the text. — [Titans](https://huggingface.co/papers/2501.00663)
- **BABILong [P]:**
  - Few-shot setting: Titans (MAC) beat Mamba-2.8B, RWKV-6-7B, RecurrentGemma-9B, Gemma-9B, Llama3.1-8B, GPT-4 and GPT-4o-mini.
  - Fine-tuned setting: it beat GPT-4, GPT-4o-mini, Qwen2.5-72B, Llama3.1-70B and "Llama3.1-8B with RAG… with about ×70 less parameters", and also beat RMT, whose memory is a 16-vector state.
  - Titans "can scale to larger than 2M context window size with better accuracy than baselines".
  - The accuracies are only in figures (images), so exact numbers were not extractable.

  — [Titans](https://huggingface.co/papers/2501.00663)
- **Ablation and efficiency [P]:** every component helped. The largest contributions came from weight decay (the forget gate), then momentum, convolution and persistent memory. The memory module trains slightly slower than Mamba2 and Gated DeltaNet. MAL is the fastest variant thanks to FlashAttention. — [Titans](https://huggingface.co/papers/2501.00663)
- **Persistence across sessions [P, from the same group's later paper]:** "test time training… or test time memorization (Behrouz et al. 2025b)… when the context is removed the acquired in-context knowledge diminishes along with it". Titans, Atlas, Miras and TTT transfer knowledge to slower levels only "through meta-learning the initial state of memory" (that is, during pretraining). — [Nested Learning](https://huggingface.co/papers/2512.24695)
- **Also from Behrouz et al. [S]:** a Google Research post "Titans + MIRAS" (Dec 2025) exists, but research.google was blocked, so it was not read.

### Inferences
- A Titans-style memory is a candidate mechanism for fast writing during a session. For the phone use case, though, it would have to be deliberately not reset (carried across sessions). Nothing in the paper tests that setting: stability, drift, or forgetting over days or weeks of user data. The forget gate is designed to discard information, which works against "must not forget".
- The update is one gradient step on a small MLP for each token or chunk. That is cheap compared with fine-tuning the full model, so the write cost stays close to inference cost. Here that is a structural argument; Titans reports no phone numbers.

### Gaps
- No exact BABILong or 2M-token accuracies (figure-only). The Titans+MIRAS blog could not be read (blocked).
- Found no experiment in which the Titans memory is carried across independent sequences or sessions.

---

## 2. ATLAS, Nested Learning / HOPE, TTT layers, test-time regression, DeltaNet / Gated DeltaNet, Longhorn (plus TTT-E2E, In-Place TTT): what each learns at test time, and whether it persists

### Takeaway
All of these learn an associative key→value (or next-token) map in fast weights while reading a sequence. The test-time-regression view unifies them: they differ in the regression weights, the function class and the optimizer. None of the published versions keeps fast-weight knowledge after the context ends. Nested Learning says so explicitly, and TTT-E2E resets its updated MLPs at document boundaries. HOPE's "continuum memory system" is a design for multi-timescale persistence, but its published "nested" variant re-initializes each level after its context. The strongest 2025–26 result, TTT-E2E at 3B, matches full-attention loss scaling to 128K, but **full attention "dramatically outperforms" it on needle retrieval**. Weight-compressed memory is lossy for exact facts.

### Cited Findings
- **ATLAS (Behrouz et al., May 2025) [A]:** ATLAS is a high-capacity long-term memory that "learns to memorize the context by optimizing the memory based on the current and past tokens". This moves beyond the purely online, last-token update of Titans. It introduces "DeepTransformers", strict generalizations of the Transformer. It "further improves the long context performance of Titans, achieving +80% accuracy in 10M context length of BABILong". The abstract does not say whether that is +80 points over Titans or 80% absolute. — [ATLAS](https://huggingface.co/papers/2505.23735)
- **Nested Learning / HOPE (Behrouz et al., NeurIPS 2025; arXiv 2512.24695) [P]:**
  - **Continuum Memory System (CMS):** a chain of MLP blocks MLP^(f1)…MLP^(fk), where the block at level ℓ is updated only every C^(ℓ) tokens (a chunk size tied to its frequency). A standard Transformer MLP is the special case with k=1 and update frequency zero.
  - **Why it should resist forgetting:** knowledge forgotten from a fast block "is still stored in other components" (slower blocks), and backprop into initial states can "circle back the knowledge".
  - **Cost:** "on average, the update cost is for O((1/f̂) × (L_layer/5) × d_in²) of parameters", so only a few parameters update at each step.
  - **Nested CMS variant:** "each of the levels… re-initialized after the end of the context".
  - **Ad-hoc level stacking:** CMS blocks can be initialized from a pretrained model's MLPs, and setting the inner learning rate η→0 recovers the pretrained model. This is proposed as a way to convert pretrained Transformers into HOPE.
  - **Explicit statement:** "Test Time Training/Memorization are… instances of parametric in-context learning, where the acquired in-context knowledge does not persist once the current context is removed."

  — [Nested Learning](https://huggingface.co/papers/2512.24695)
- **HOPE results [S only]:** HOPE was evaluated at 340M, 760M and 1.3B parameters, with the 1.3B model trained on 100B tokens. It had lower perplexity and higher accuracy than Transformer++, RetNet, DeltaNet, Samba and Titans on language modeling and commonsense reasoning, and beat Titans, TTT and Mamba2 on long-context NIAH. The HF text mirror of 2512.24695 is truncated before the experiments section, so **no HOPE numbers were verified**. — [Google Research blog (via search snippet)](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/); [MarkTechPost (via search snippet)](https://www.marktechpost.com/2025/11/08/nested-learning-a-new-machine-learning-approach-for-continual-learning-that-views-models-as-nested-optimization-problems-to-enhance-long-context-processing/)
- **TTT layers (Sun et al. 2024, arXiv 2407.04620) [A]:** "the hidden state [is] a machine learning model itself, and the update rule a step of self-supervised learning". TTT-Linear's state is a linear model and TTT-MLP's is a 2-layer MLP. Both were evaluated at 125M–1.3B against a Transformer and Mamba. Like the Transformer, they "keep reducing perplexity by conditioning on more tokens, while Mamba cannot after 16k context". TTT-MLP "still faces challenges in memory I/O". — [TTT layers](https://huggingface.co/papers/2407.04620)
- **Test-time regression (Wang, Shi, Fox 2025; arXiv 2501.12352) [A]:** memorization is cast as a regression problem. "Linear attention, state-space models, fast-weight programmers, online learners, and softmax attention" arise from three choices: the regression weights, the regressor function class and the test-time optimization algorithm. The paper explains why linear attention misses inter-token correlations and gives a justification for query-key normalization. — [Test-time regression](https://huggingface.co/papers/2501.12352)
- **Unification, TTT-E2E view [P]:** "when g is a linear model, TTT-KVB recovers DeltaNet". Taking the gradient with respect to W_0 instead of W_{t−1} recovers linear attention, and a non-parametric kernel estimator recovers self-attention. KV-binding is "the core component" of MesaNet, Titans and Nested Learning. — [TTT-E2E](https://huggingface.co/papers/2512.23675)
- **DeltaNet / Gated DeltaNet (Yang, Kautz, Hatamizadeh, ICLR 2025; arXiv 2412.06464):**
  - [A] "gating enables rapid memory erasure while the delta rule facilitates targeted updates". Gated DeltaNet beats Mamba2 and DeltaNet on language modeling, commonsense, in-context retrieval, length extrapolation and long context. Hybrids with sliding-window attention or Mamba2 do better still. — [Gated DeltaNet](https://huggingface.co/papers/2412.06464)
  - [P, via Titans' description] the delta rule "first removes its past value" before writing a key-value pair. — [Titans](https://huggingface.co/papers/2501.00663)
  - The standard form, from memory and not re-verified this session, is S_t = S_{t−1}·α_t(I − β_t·k_t·k_tᵀ) + β_t·v_t·k_tᵀ.
- **Longhorn (Liu et al. 2024; arXiv 2407.14207) [A]:** SSMs are treated as "meta-modules for specific online learning problems". Longhorn's update "resembles the closed-form solution for solving the online associative recall problem". At 1.3B on SlimPajama it gives "1.8x improvement in sample efficiency compared to Mamba" and extrapolates to contexts up to 16× longer than its 2,048 training length. — [Longhorn](https://huggingface.co/papers/2407.14207)
- **TTT-E2E (Tandon, Sun et al., Dec 2025; arXiv 2512.23675) [P]:**
  - **Method:** a standard Transformer with sliding-window attention (8K window) that "continues learning at test time via next-token prediction on the given context, compressing the context it reads into its weights". The initialization is meta-learned (gradients of gradients).
  - **Scaling:** at 3B parameters trained on 164B tokens, it "scales with context length in the same way as Transformer with full attention, while others, such as Mamba 2 and Gated DeltaNet, do not". It has constant inference latency and is **2.7× faster than full attention at 128K on an H100**.
  - **Which weights update:** only MLPs in the **last 1/4 of blocks** are updated, in TTT mini-batches of 1K tokens. Updating 1 or 3 of 24 layers did not scale with context; 6 layers did; 12 layers was no better than 6.
  - **Guarding pretrained knowledge:** "one of the concerns of TTT is forgetting the knowledge learned during pre-training". Their fix is a second, static MLP in each updated block as "safe storage".
  - **Efficiency at 760M:** the state is 88M parameters (versus 18M for the multi-head variant), and prefill takes 0.0086 s per 1K tokens on an H100.
  - **Persistence:** the authors discard documents shorter than 8K tokens "to avoid resetting the updated MLPs across document boundaries", so the updates are per-document.
  - **Recall weakness:** on RULER NIAH at 128K, "Transformer with full attention dramatically outperforms the other methods, including ours… the strength of full attention lies in its nearly lossless recall".

  — [TTT-E2E](https://huggingface.co/papers/2512.23675)
- **In-Place TTT (ByteDance Seed / PKU, Apr 2026; arXiv 2604.06169) [A]:** treats "the final projection matrix of the ubiquitous MLP blocks as its adaptable fast weights". It is a "drop-in" addition to existing LLMs without retraining from scratch. It uses an objective aligned with next-token prediction and chunk-wise updates compatible with context parallelism. It lets a 4B model reach "superior performance on tasks with contexts up to 128k", and it beats other TTT approaches when pretrained from scratch. — [In-Place TTT](https://huggingface.co/papers/2604.06169)
- **LaCT, "Test-Time Training Done Right" (2505.23884) [S, HF summary]:** large-chunk TTT updates improve hardware utilization and state capacity. — [LaCT](https://huggingface.co/papers/2505.23884)

### Inferences
- For a "never forget" on-device assistant, these architectures supply the **fast learner** (cheap updates inside a context) but not the **consolidation step**. Something must copy fast-weight content into durable storage. Examples are HOPE-style slow blocks that are never reset, or a hypernetwork or gradient write into a persistent adapter.
- In-Place TTT and TTT-E2E update existing MLP down/output projections. That is the same site Doc-to-LoRA writes into (MLP down-projection), which suggests a common "writable" substrate in the MLP output matrices.
- TTT-E2E's NIAH result is a warning: compressing text into weights through next-token-prediction TTT keeps gist better than verbatim facts. An assistant that must recall exact user-taught facts (names, numbers) may need a separate exact-recall store.

### Gaps
- HOPE numbers (perplexity, NIAH, BABILong, continual-learning tasks) are unverified because the text mirror is truncated and research.google is blocked.
- No paper found that runs any of these architectures with fast weights kept across sessions or users.
- Found no on-device (phone) latency or energy figures for any TTT-layer architecture.

---

## 3. MEND (Mitchell et al. 2022) as the original amortized editor (brief)

### Takeaway
MEND is the prototype "learned weight writer". Small editor networks turn the fine-tuning gradient of one (input, desired output) pair into a weight edit, using a low-rank gradient decomposition. It needs one gradient computation but no optimization loop, and it scaled to models above 10B parameters.

### Cited Findings
- [A] MEND is "a collection of small auxiliary editing networks that use a single desired input-output pair to make fast, local edits". It "learns to transform the gradient obtained by standard fine-tuning, using a low-rank decomposition of the gradient". It "can be trained on a single GPU in less than a day even for 10 billion+ parameter models" and was "the only approach… that effectively edits the behavior of models with more than 10 billion parameters" (T5, GPT, BERT and BART were tested). — [MEND](https://huggingface.co/papers/2110.11309)
- Unlike Doc-to-LoRA or Generative Adapter, MEND still needs a backward pass to get the raw gradient (it transforms a gradient rather than generating from activations). — [MEND](https://huggingface.co/papers/2110.11309); [Doc-to-LoRA](https://huggingface.co/papers/2602.15902)

### Inferences
- MEND-style editors handle single facts. Later "document writers" (Generative Adapter, D2L, SHINE) generalize the idea to whole contexts, using forward passes only.

### Gaps
- Sequential-edit degradation numbers for MEND were not gathered here, since editing is covered by another researcher.

---

## 4. Generative Adapter (Chen et al., ICLR 2025; arXiv 2411.05877)

### Takeaway
A roughly 500M-parameter generator maps the base LM's hidden states for a context directly to additive adapter weights, using forward passes only. It has a **fixed-size, additive streaming state**: new context chunks add to a d_r×d_r accumulator. On StreamingQA it improves closed-book F1 from 19.5 (supervised fine-tuning) to 31.5, and it cuts compute and storage 4× versus full-history prompting on MSC. Accuracy is still well below in-context use (MSC F1 40.2 vs 66.0), and it works best for contexts under about 1K tokens.

### Cited Findings
- **Mechanism [P]:**
  - Adapters are generated for linear projections. The main experiments adapt only the attention output projection "for efficiency"; the FFN down-projection has 3× more parameters and was slightly better in ablation.
  - Generator: W_Δ = (A1·A2)·(Σ_m h_m⊗h_m)·(B1·B2), computed from the previous layer's hidden states, followed by rank-128 SVD normalization. The authors call this necessary because singular values are skewed; Frobenius normalization did worse.
  - Streaming update: S_t ← S_{t−1} + A2·H_tᵀ·H_t·B1, then W_Δt ← A1·norm(S_t)·B2. The memory S_t is d_r×d_r with d_r = 1,024, and "S_0 initialized as all zeros".
  - Size: ~500M generator parameters; each generated adapter is 32M parameters.

  — [Generative Adapter](https://huggingface.co/papers/2411.05877)
- **Training [P]:** self-supervised pretraining on 1B SlimPajama tokens (8,192-token segments) with reconstruction plus completion losses. Using only reconstruction "results in a significant drop in completion perplexity… overfitting to memorization". This is followed by instruction tuning. Contexts are chunked at 1,024 tokens. Base models are Mistral-7B-Instruct v0.2 and Llama2-7B-Chat. — [Generative Adapter](https://huggingface.co/papers/2411.05877)
- **Knowledge acquisition [P]:** on StreamingQA, closed-book F1 went from **19.5 (SFT) to 31.5** (+63.5%) for contexts up to 32K. GenerativeAdapter "is highly effective when the context is relatively short (< 1K tokens)" and "outperforms CPT [continual pretraining], especially when the context length is less than 8K tokens". Its preprocessing time is "orders of magnitude smaller than CPT". — [Generative Adapter](https://huggingface.co/papers/2411.05877)
- **In-context learning [P]:** average accuracy 44.9 across 26 MetaICL tasks, above the base model. It beats few-shot prompting "in most cases", especially on non-classification tasks. — [Generative Adapter](https://huggingface.co/papers/2411.05877)
- **Personalization, MSC (Mistral-7B) [P]:**

  | Method | F1 | Inference TFLOPs | Extra storage |
  |---|---|---|---|
  | Closed-book | 8.1 | 0.505 | 0 |
  | Full-conversation prompting | 66.0 | 2.059 | 128M+ floats |
  | UltraGist, 512 tokens | 40.8 | 0.772 | 32M floats |
  | UltraGist, 1K tokens | 44.4 | 1.067 | 64M floats |
  | **GenerativeAdapter** | **40.2** | **0.505** | **32M floats** |

  The average conversation is 2.5K tokens. The authors call full prompting's cost "4x those of GenerativeAdapter" and "highly undesirable… on edge devices". — [Generative Adapter](https://huggingface.co/papers/2411.05877)
- **Critique [P]:** Doc-to-LoRA reports that GA's higher ROUGE-L F1 over the base model comes from shorter outputs. GA "achieves much lower ROUGE-L recall… answering with less factual accuracy". — [Doc-to-LoRA](https://huggingface.co/papers/2602.15902)

### Inferences
- GA's accumulator is exactly an "accumulate many writes into a fixed-size state" design: linear-attention-like, 32M floats (about 64 MB in fp16) per user regardless of history length. The context-length results show the price. Recall falls as more is accumulated, the classic capacity and interference problem of additive associative memories.
- For a phone, the attractive part is that the write is a forward pass plus a small matrix update. The unattractive part is a 500M-parameter generator on top of a 7B base.

### Gaps
- Exact StreamingQA and SQuAD F1 by context length, and latency numbers, are figure-only.
- No test of accumulation beyond 32K tokens, and no test of paraphrased or multi-hop questions.

---

## 5. Text-to-LoRA, Drag-and-Drop LLMs, Doc-to-LoRA and other doc/knowledge→LoRA hypernetworks (2025–2026), and Cartridges

### Takeaway
- **Instruction/task → LoRA** (T2L, DnD) works for *skills and task styles*. T2L gets within about 2 points of per-task LoRAs zero-shot. It **does not internalize document knowledge**.
- **Document → LoRA via meta-learned context distillation** (Doc-to-LoRA, Feb 2026) is the most relevant for "read a page once, answer later without it". With gemma-2-2b it reaches **about 82–87% of in-context accuracy** in **0.09–0.55 s**, using under 2–4 GB of update memory. That is roughly 100× faster than gradient-based distillation, and it handles documents up to 32K tokens and needle recall to 40K tokens.
- 2026 follow-ups show that such adapters **fail to override strong priors** (46% accuracy on deep conflicts) and that hypernetwork knowledge injection follows **power-law scaling with good out-of-distribution generalization**.
- **Cartridges** use offline, gradient-trained context distillation into a KV prefix instead: ICL quality at 38.6× less memory, composable. Training is expensive ("sleep-time compute").

### Cited Findings
**Text-to-LoRA (Sakana; ICML 2025; arXiv 2506.06105) [P]**
- **Setup:** base model Mistral-7B-Instruct; the task description is embedded with gte-large-en-v1.5. The hypernetwork variants L, M and S have **55M, 34M and 5M trainable parameters**. They emit rank-8 LoRAs on q and v projections of every attention block (3.4M parameters per LoRA). Training used 479 Super-NaturalInstructions tasks with 128 descriptions each. — [T2L](https://huggingface.co/papers/2506.06105)
- **LoRA compression (reconstruction training, 9 benchmarks):** T2L average **73.4** versus 73.3 for the task-specific LoRAs (base model 55.8), so it "fully recovers the performance of the oracle adapters". As the number of distilled tasks grows (16→479), performance drops. The compression is lossy, but T2L keeps "around 65% of oracles' performance". — [T2L](https://huggingface.co/papers/2506.06105)
- **Zero-shot on unseen tasks (SFT-trained, 10 benchmarks):**
  - T2L-L **67.7** average versus multi-task LoRA 66.3, Hyperdecoders 67.3, 3-shot ICL 61.0, prepending the task description 60.6, and base Mistral 55.8.
  - On 8 tasks, T2L-L scores 73.9 versus 75.8 for the task-specific oracle LoRAs.
  - Misaligned or random descriptions drop reconstruction-trained T2L to 51.4 or 63.5 average, versus 73.3 with aligned descriptions.

  — [T2L](https://huggingface.co/papers/2506.06105)
- **Limit:** D2L reports that T2L "can instantly update the model, [but] cannot effectively internalize knowledge due to its highly focused training data". — [Doc-to-LoRA](https://huggingface.co/papers/2602.15902)

**Drag-and-Drop LLMs (NUS/UT Austin et al., Jun 2025; arXiv 2506.16406) [A/P]**
- A frozen lightweight text encoder turns a batch of **unlabeled task prompts** into condition embeddings. A "cascaded hyper-convolutional decoder" expands these into the full set of LoRA matrices "in seconds".
- Reported as "up to **12,000× lower overhead** than full fine-tuning" and "average gains up to **30%**… over the strongest training LoRAs on unseen common-sense reasoning, math, coding, and multimodal benchmarks". It transfers "from 0.5B to 7B parameter backbones".
- Motivating claim: LoRA-tuning even Qwen2.5-0.5B "occupies four A100 GPUs for half a day".

— [DnD](https://huggingface.co/papers/2506.16406)

**Doc-to-LoRA, D2L (Charakorn, Cetin, Uesaka, Lange, Sakana AI; Feb 2026; arXiv 2602.15902) [P]**
- **Objective:** the hypernetwork is meta-trained to approximate query-independent context distillation (CD). It minimizes KL(p_θ(y|x,c) ‖ p_{θ+H_φ(c)}(y|x)) over generated queries and self-responses, so "a trained mapping H amortizes *both* the query generation process and the backpropagation needed by traditional CD". — [D2L](https://huggingface.co/papers/2602.15902)
- **Architecture:**
  - The frozen target LLM encodes the context, giving per-layer activations.
  - A shared Perceiver-style cross-attention hypernetwork (8 cross-attention blocks, **309M trainable parameters**) outputs a rank-8 LoRA for each 8K-token chunk on each layer's **MLP down-projection**, with a learnable per-layer scale.
  - Long documents are chunked, and per-chunk LoRAs are concatenated along the rank dimension (total rank r·K).
  - The same architecture can instead output a compressed KV prefix.

  — [D2L](https://huggingface.co/papers/2602.15902)
- **Needle-in-a-haystack (gemma-2-2b-it, 8K native context):** trained only on 32–256-token contexts, D2L is "close to perfect up to 40 chunks (40K tokens)", more than 4× (they also say 5×) the native window, and then degrades gracefully. At a 128K haystack the base model needs **over 12 GB** of extra memory; D2L needs **under 50 MB**. — [D2L](https://huggingface.co/papers/2602.15902)
- **Short-passage QA:** D2L reaches **82.5%** of ICL performance on SQuAD, which is "roughly similar to compressing the context down to 40%" with LLMLingua-2 "while removing the context entirely". It internalizes "within less than a second". CD (oracle) takes about 40 s and CD with generated queries over 100 s. D2L and oracle CD use under 2 GB of VRAM for the update; CD with generated queries uses over 40 GB. — [D2L](https://huggingface.co/papers/2602.15902)
- **Long-document multi-hop QA (2WikiMultihopQA, documents up to 32K tokens; longest training sample 2,344 tokens):**

  | Method | Normalized performance | Extra update memory (GB) | Update latency (s) |
  |---|---|---|---|
  | CD (oracle query) | 0.901 | 7.82 | 40.2 |
  | **D2L batched** | **0.857** | 11.52 | **0.209** |
  | **D2L iterative** | **0.844** | **3.79** | 0.551 |
  | CD, 25 generated queries | 0.745 | 59.9 | 465 |
  | CD, 5 generated queries | 0.704 | 79.4 | 72.5 |

  During answering, ICL needs about 1 GB of VRAM versus under 100 MB for all in-parameter methods. — [D2L](https://huggingface.co/papers/2602.15902)
- **Amortization beats many-query CD (SQuAD, 100 samples):**

  | Method | Normalized performance | Update time |
  |---|---|---|
  | CD oracle | 0.988 | 8.76 s |
  | **D2L** | **0.866** | **0.086 s** |
  | CD, 100 generated queries | 0.650 | 631 s |
  | CD, 50 generated queries | 0.601 | 312 s |
  | CD, 20 generated queries | 0.506 | 129 s |

  — [D2L](https://huggingface.co/papers/2602.15902)
- **Other results:**
  - With a VLM as encoder (gemma-3-4b) writing into a text-only LLM, D2L gets 75.03% on Imagenette "purely through the internalized information".
  - Internalizing the *query* instead of the document gives recall 0.740 versus 0.886 with context.
  - Caveat from the authors: D2L "might assume that subsequent queries will always be related to the internalized knowledge".
  - Mistral-7B-Instruct-v0.2 and Qwen3-4B results are in the appendix (not read).

  — [D2L](https://huggingface.co/papers/2602.15902)

**SHINE (Feb 2026; arXiv 2602.06358) [A/P]**
- An "in-context hypernetwork" that reuses the frozen LLM's own parameters. A lightweight Transformer passes messages between layers to generate LoRAs for all layers with "no bottleneck". It was pretrained on **6B tokens** plus instruction tuning.
- It turns "in-context knowledge to in-parameter knowledge in one pass" and reports that it "greatly saves time, computation and memory costs compared to SFT-based LLM adaptation", with "no sign of hitting capacity bottleneck" in scaling.
- It criticizes earlier designs for generating LoRAs only for a subset of layers (GA), having tiny bottlenecks, or reusing a small MLP (T2L).
- Exact benchmark numbers were not extracted.

— [SHINE](https://huggingface.co/papers/2602.06358)

**Scaling Laws for Hypernetwork-Based Knowledge Injection (Nace AI / Purdue, Jul 2026; arXiv 2607.19604) [A]**
- This is *train-time* injection: a hypernetwork generates a **fixed** LoRA from a large fact corpus. It uses MegaWikiQA, which has "tens of millions of multi-hop question-answer examples across 39 knowledge domains" built from Wikidata5M.
- Findings: "broadly predictive power law scaling along all architecture axes" (hypernetwork depth, width and target size), and "reliable OOD generalization to unseen entities and relations at increasing scales". It shows "steeper scaling exponents [than LoRA finetuning and full fine-tuning] in all OOD evaluations".

— [Scaling laws](https://huggingface.co/papers/2607.19604)

**The Override Gap (Apr 2026; arXiv 2604.23750) [A]**
- Doc-to-LoRA-style hypernetworks "fail systematically on conflicts": accuracy "collapses to 46.4% on the deepest facts".
- Across 194 conflicts sorted by prior strength, baseline accuracy falls from 68% (weak prior) to 16% (strong prior). The cause is that the adapter margin is roughly constant across documents while the pretrained margin grows with how often a fact appeared in training.
- Training-free fixes (Selective Layer Boosting plus Conflict-Aware Internalization) raise deep-conflict accuracy **46.4%→71.0% (Gemma-2B)** and **53.6%→72.5% (Mistral-7B)**, preserve novel-knowledge recall, and "beat vanilla RAG on medium conflicts by 18 percentage points".
- The authors release KID-Bench (489 questions: novel recall, cross-knowledge combination, prior-graded conflicts).

— [Override Gap](https://huggingface.co/papers/2604.23750)

**Cartridges (Eyuboglu, Ehrlich, Arora et al., Stanford/Hazy Research, Jun 2025; arXiv 2506.06266) [P]**
- **Method:** a small KV cache (prefix-tuning parameterization) is trained offline for each corpus with SELF-STUDY. The model generates synthetic conversations about the corpus, chunked and steered by seed prompts, and the cartridge is trained on them with a context-distillation objective. — [Cartridges](https://huggingface.co/papers/2506.06266)
- **Results:**
  - Averaged across benchmarks (corpora of 100K–484K tokens), cartridges **match ICL quality using 38.6× less memory, enabling 26.4× higher peak throughput**. That is "an order of magnitude improvement over… cache compression baselines (e.g. DuoAttention)", which degrade beyond 2× compression.
  - On MTOB, a Llama-8B cartridge from a 484K-token textbook extends the effective context from 128K to 484K tokens and beats ICL on the first 130K tokens by 11.0 chrF.

  — [Cartridges](https://huggingface.co/papers/2506.06266)
- **Important negatives and positives:**
  - A cartridge trained with plain next-token prediction "memorize[s] the corpus perfectly using 107× less memory than the KV-cache" but is "not general" and degrades diverse question-answering.
  - Cartridges are "**composable without joint optimization**": several can be concatenated and queried together.
  - A KV-cache parameterization beat LoRA "on both in-domain and out-of-domain tasks".
  - Motivating numbers: LLaMA-70B needs 84 GB for a 128K context, and LLaMA-8B throughput drops 77× from 1K to 120K context on an H100.

  — [Cartridges](https://huggingface.co/papers/2506.06266)
- D2L describes Cartridges as using "abundant sleep-time compute budget for CD", i.e. an offline, gradient-based write. — [D2L](https://huggingface.co/papers/2602.15902)

**Memory of Amortized Contexts, MAC (Tack et al., NeurIPS 2024; arXiv 2403.04317) [A/P]**
- An amortization network compresses each new document into a compact PEFT modulation stored in a **memory bank**. At question time the relevant modulations are selected and aggregated into one (divide-and-conquer for large banks) to adapt a **frozen** LM "without requiring further gradient updates".
- Pitched as avoiding the catastrophic forgetting and gradient cost of online fine-tuning, and as suited to edge devices compared with RAG over many documents.
- Numbers were not extracted.

— [MAC](https://huggingface.co/papers/2403.04317)

**Other named work, cited only through D2L's reference list (not read)**
- HyperLoRA (Lv et al., ACL Findings 2024)
- Hypertuning (Phang et al., ICML 2023)
- Knowledge Modules with deep context distillation (Caccia et al., COLM 2025)
- Context parametrization with compositional adapters (Jukić et al. 2025)
- Zhyper (Abdalla et al. 2025)

— [D2L references](https://huggingface.co/papers/2602.15902); [SHINE](https://huggingface.co/papers/2602.06358)

### Inferences
- **Size of one D2L write (my arithmetic, assuming gemma-2-2b has 26 layers, d_model 2304 and d_ff 9216; config not re-verified):** a rank-8 LoRA on every down-projection is 26 × 8 × (9216 + 2304) ≈ 2.4M parameters, about 4.8 MB in fp16, per ≤8K-token chunk. Because chunks concatenate along rank, a stored library grows linearly: roughly 5 MB per page-sized write, or about 5 GB per 1,000 unmerged pages. Merging LoRAs to stay fixed-size is untested for knowledge adapters.
- **Two families:** T2L/DnD condition on task descriptions and produce *skills*; D2L, GA and SHINE condition on the document's own activations and produce *knowledge*. For "the user teaches it a fact" or "it reads a page once", the second family is the relevant one.
- The Override Gap result matters directly for a personal assistant, because user corrections often contradict priors. Written adapters need a conflict-aware amplitude or gating mechanism.

### Gaps
- T2L's FLOPs analysis (Appendix I), DnD's exact per-benchmark numbers, SHINE's benchmark numbers and MAC's StreamingQA numbers were not extracted.
- Cartridges' training cost per corpus (GPU-hours) was not verified.
- No doc→LoRA paper found that measures write latency on a phone or NPU.

---

## 6. Test-time training for knowledge and reasoning (ARC TTT, TTT on nearest neighbors, "on the fly", 2025–26 per-user continual TTT)

### Takeaway
Gradient-based TTT with a few LoRA steps per task or query gives large gains: ARC 53.0% with an 8B model, and 61.9% ensembled. It narrows 10× size gaps (Hardt & Sun). But it is per-instance and usually discarded. When updates are applied **sequentially and kept**, as in SEAL, earlier knowledge degrades. Sparse memory-layer fine-tuning is the most forgetting-resistant persistent write found: an 11% drop in held-out NaturalQuestions F1 versus 89% for full fine-tuning.

### Cited Findings
- **Akyürek et al. (MIT; ICML 2025; arXiv 2411.07279) [A]:** "temporarily updating model parameters during inference using a loss derived from in-context examples". On ARC, TTT gives "up to 6× higher accuracy compared to fine-tuned baselines". It reaches **53.0%** on the public validation set with an 8B model and **61.9%** when ensembled with program synthesis ("matching average human performance"). On BIG-Bench Hard (10-shot), TTT scores 57.8% versus 50.5% for few-shot prompting. On an 80-task ARC subset, TTT adds 27.5 points over the fine-tuned model. — [ARC TTT](https://huggingface.co/papers/2411.07279)
- **TTT on nearest neighbors (Hardt & Sun, ICLR 2024; arXiv 2305.18466) [A]:** retrieve neighbors from a Pile embedding index and fine-tune on them at test time. "Retrieving and training on as few as 20 neighbors, each for only one gradient iteration, drastically improves performance across more than 20 language modeling tasks in the Pile". It "significantly narrows the performance gap between a small GPT-2 and a GPT-Neo model more than 10 times larger". "Sufficient index quality and size… are necessary." — [TTT-NN](https://huggingface.co/papers/2305.18466)
- **SEAL (Zweiger, Pari et al., MIT; arXiv 2506.10943) [P]:** the model writes its own fine-tuning data (a "self-edit"), which is applied by LoRA SFT, and reinforcement learning (ReST-EM) trains self-edit generation.
  - **Knowledge incorporation, no-context SQuAD, Qwen2.5-7B, single passage:** base 32.7%; fine-tuning on the passage only 33.5%; self-generated implications 39.7%; GPT-4.1 implications 46.3%; **SEAL 47.0%**. In continued pretraining over 200 documents, SEAL reaches 58.2%.
  - **ARC subset, Llama-3.2-1B:** 72.5% success versus 20% without RL and 0% without adaptation.
  - **Limitations:** "performance on earlier tasks gradually declines as the number of edits increases… still susceptible to catastrophic forgetting" but "without complete collapse". Each self-edit evaluation "takes approximately 30–45 seconds".

  — [SEAL](https://huggingface.co/papers/2506.10943)
- **Sparse Memory Finetuning (Lin et al., FAIR/Berkeley, Oct 2025; arXiv 2510.15103) [A]:** updates only memory-layer slots that are "highly activated by a new piece of knowledge relative to usage on pretraining data". After learning new facts, held-out NaturalQuestions F1 "drops by **89%** after full finetuning… **71%** with LoRA, [and] only **11%**" with sparse memory fine-tuning, "with the same level of new knowledge acquisition". — [SMF](https://huggingface.co/papers/2510.15103)
- **2025–26 TTT-for-LLM variants [S, HF one-line summaries]:**
  - "Test-Time Learning for LLMs" (2505.20633): adapts on unlabeled test inputs by minimizing input perplexity.
  - "Self-Guided TTT for Long-Context LLMs" (2607.09415): selects evidence spans for adaptation on each instance.
  - TTT-E2E and In-Place TTT: see §2.

  — [TLM](https://huggingface.co/papers/2505.20633); [Self-Guided TTT](https://huggingface.co/papers/2607.09415)

### Inferences
- The gradient-TTT results show that small, cheap LoRA updates can carry real new capability. The durable-memory question is really about *where* updates go (sparse, slot-addressed parameters versus shared dense ones). Sparse memory fine-tuning, together with a hypernetwork writer, is a natural hybrid that nobody appears to have published yet (my inference).

### Gaps
- No paper found with the exact title "Test-time training on the fly for LLMs". The closest are TLM (2505.20633), In-Place TTT and SIFT/active fine-tuning (Hübotter et al. 2024, cited by TTT-E2E but not read).
- ARC TTT's compute per task was not extracted.
- Found no 2025–26 paper doing *per-user* continual TTT over weeks of real usage.

---

## 7. Cost: one hypernetwork forward pass versus a few gradient steps (FLOPs, latency, memory)

### Takeaway
Measured on datacenter GPUs, amortized writers internalize a document in **0.09–0.55 s with 2–4 GB of extra memory** (D2L on gemma-2-2b). Gradient context distillation needs **8.8–40 s even with an oracle query**, over 100 s to 10 min with generated queries, and **8–80 GB**. That is roughly 100× the latency, and about 10× the memory when queries are generated. At read time the gain is from dropping the KV cache (for example, over 12 GB → under 50 MB at 128K). No phone measurements exist in these papers.

### Cited Findings
- **D2L versus CD [P]:** latency 0.086 s versus 8.76 s (oracle CD) and 631 s (CD with 100 queries) on SQuAD; 0.209–0.551 s versus 40.2 s (oracle) and 72.5–465 s (generated queries) on 2WikiMultihopQA. Update memory 3.79–11.5 GB versus 7.82 GB (oracle) and 59.9–79.4 GB. On SQuAD, D2L and oracle CD both use under 2 GB. The hypernetwork itself is 309M parameters. — [D2L](https://huggingface.co/papers/2602.15902)
- **Read-time savings [P]:**
  - D2L: over 12 GB (base model at 128K) versus under 50 MB (D2L); ICL about 1 GB versus under 100 MB for 2WikiMultihopQA-length documents. — [D2L](https://huggingface.co/papers/2602.15902)
  - GA: 0.505 versus 2.059 TFLOPs per query, and 32M versus 128M+ floats (MSC). — [GA](https://huggingface.co/papers/2411.05877)
  - Cartridges: 38.6× less memory and 26.4× throughput. — [Cartridges](https://huggingface.co/papers/2506.06266)
- **DnD [A]:** "up to 12,000× lower overhead than full fine-tuning", adapters generated "in seconds". — [DnD](https://huggingface.co/papers/2506.16406)
- **Gradient TTT costs [P]:**
  - SEAL: about 30–45 s per self-edit evaluation, i.e. a fine-tune plus evaluate. — [SEAL](https://huggingface.co/papers/2506.10943)
  - TTT-E2E: a 760M model prefills at 0.0086 s per 1K tokens on an H100 *with* in-context gradient steps (last 1/4 of MLPs, 1K-token mini-batches), and the 3B model is 2.7× faster than full attention at 128K. — [TTT-E2E](https://huggingface.co/papers/2512.23675)
- **CMS update cost [P]:** O((1/f̂)·(L/5)·d²) parameters updated per step on average. — [Nested Learning](https://huggingface.co/papers/2512.24695)

### Inferences (not measured in any source)
- **FLOPs rule of thumb:** a forward pass costs about 2·N FLOPs per token, and a training step about 6·N (forward plus backward), plus optimizer state and stored activations.
- **D2L-style write:** about one prefill of the document through the base model (to get the activations) plus the hypernetwork pass, with no backward pass and no optimizer state. The write therefore costs about the same as *reading* the page once, which suits phone NPUs built for inference.
- **Gradient write:** k steps cost about 3k prefills of the training data, plus query/answer generation if distilling, plus fp32 or optimizer memory. NPUs usually do not support it.
- **Storage:** about 5 MB (D2L rank-8 on gemma-2-2b, per 8K chunk) to 64 MB (GA's 32M-float state, fp16) per write or user state.

### Gaps
- No on-device (phone or NPU) latency, energy or memory numbers exist for any hypernetwork writer or TTT layer in the sources found.
- T2L's own FLOPs analysis was not read.

---

## 8. Failure modes: do written updates generalize (paraphrase, multi-hop), and do they accumulate without interference?

### Takeaway
**Generalization is partial.**
- D2L reaches about 85% of ICL on multi-hop 2WikiMultihopQA and generalizes to 8× its training length.
- Train-time hypernetwork injection shows out-of-distribution generalization that improves with scale.
- But hypernetwork adapters fail when a document conflicts with a strong prior (16–46% accuracy).
- They may assume every later query relates to the written content.

**Accumulation is the weakest-evidenced area.**
- Chunk concatenation in D2L works up to about 40 chunks.
- GA's additive state degrades beyond about 1K–8K tokens.
- Cartridges compose by concatenation.
- Sequential gradient writes (SEAL) degrade prior knowledge gradually.
- Nobody reports hundreds or thousands of hypernetwork writes into one persistent model.

### Cited Findings
- **Multi-hop and length generalization [P]:** D2L scores 0.857 normalized on 2WikiMultihopQA with documents up to 32K tokens, trained on at most 2,344 tokens. NIAH is near-perfect to 40K tokens after training on at most 256-token contexts. — [D2L](https://huggingface.co/papers/2602.15902)
- **Query bias [P]:** D2L "might assume that subsequent queries will always be related to the internalized knowledge". — [D2L](https://huggingface.co/papers/2602.15902)
- **Conflicts with priors [A]:** deep-conflict accuracy is 46.4% for D2L-style adapters, and 16% on the strongest-prior quartile versus 68% on weak priors. Amplitude boosting partially fixes this (71.0–72.5%). — [Override Gap](https://huggingface.co/papers/2604.23750)
- **Out-of-distribution scaling [A]:** hypernetwork knowledge injection generalizes "to unseen entities and relations at increasing scales", with steeper OOD scaling exponents than LoRA or full fine-tuning. This is on multi-hop MegaWikiQA. — [Scaling laws](https://huggingface.co/papers/2607.19604)
- **Behavior versus facts [P]:** T2L recovers task behavior but "cannot effectively internalize knowledge". The next-token-prediction-trained Cartridge memorizes perfectly but is "not general". GA's training needed a completion loss to avoid "overfitting to memorization". — [D2L](https://huggingface.co/papers/2602.15902); [Cartridges](https://huggingface.co/papers/2506.06266); [GA](https://huggingface.co/papers/2411.05877)
- **Accumulation:**
  - GA's fixed-size additive memory is most effective under 1K tokens and beats continual pretraining only under about 8K. — [GA](https://huggingface.co/papers/2411.05877)
  - D2L's rank-concatenation degrades beyond about 40 chunks on NIAH. — [D2L](https://huggingface.co/papers/2602.15902)
  - Cartridges "can be concatenated and queried together". — [Cartridges](https://huggingface.co/papers/2506.06266)
  - SEAL's sequential self-edits: "performance on earlier tasks gradually declines as the number of edits increases". — [SEAL](https://huggingface.co/papers/2506.10943)
  - Dense fine-tuning forgets heavily (NaturalQuestions F1 −89% full fine-tuning, −71% LoRA), versus −11% for sparse memory slots. — [SMF](https://huggingface.co/papers/2510.15103)
- **Recall fidelity of weight-compressed context [P]:** full attention "dramatically outperforms" TTT-E2E on NIAH at 128K. — [TTT-E2E](https://huggingface.co/papers/2512.23675)
- **Fast-weight non-persistence [P]:** "acquired in-context knowledge does not persist once the current context is removed". — [Nested Learning](https://huggingface.co/papers/2512.24695)

### Inferences
- A plausible on-device design suggested by this evidence (not validated anywhere):
  1. A D2L/SHINE-style writer produces a per-document adapter in about one prefill.
  2. Adapters are stored as a growing library with retrieval or routing over them (as MAC does over its memory bank), rather than being merged.
  3. Adapters are consolidated periodically, during "sleep" such as overnight charging, into slot-addressed sparse memory parameters (as in sparse memory fine-tuning) or slow HOPE-style CMS blocks.
  4. Conflict-aware amplitude is used when a user correction contradicts the model's prior.
- Each piece has separate evidence. The combination, and its behavior after thousands of writes, is untested.

### Gaps
- There is no published experiment with more than about 100 sequential hypernetwork writes into one model, and none measuring interference among many stored doc-LoRAs, whether merged or routed.
- There is no paraphrase-robustness study specifically for doc→LoRA adapters. D2L's QA uses the benchmark questions, which may paraphrase the passage to varying degrees.
- D2L's Appendix C on failure modes, and results on Mistral-7B and Qwen3-4B, were not read.
