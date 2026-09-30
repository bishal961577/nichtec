# Using Newly Learned Knowledge in Reasoning: Where It Stands in September 2026

After test 6 closed the "model addresses its own memory" route, one open problem was left that every system still
fails: using facts learned after deployment inside multi-step reasoning. This report checks the 2024–2026
literature on that problem before any experiment is proposed. It asks whether a gap is left that this project
could close.

**Bottom line:**
- **With explicit step-by-step reasoning, one new fact at a time is largely solved.**
  - CODE (May 2026) reaches up to 83.5% two-hop accuracy on MQuAKE-CF-3k-v2 for 7–8B instruct models, editing one
    case at a time.
  - HPSE (Aug 2026) makes injected facts composable and keeps the gain as edits accumulate.
  - Both train the model to reproduce what it would say with the new fact in its context (self-distillation).
- **Without step-by-step reasoning, the barrier is structural, and now explained.**
  - It works when the new fact is the first step and points to an entity the model already knows (Balesni et al.).
  - It fails when both facts are new, or when the new fact is about an entity the model only reaches internally
    (the two-hop curse).
  - Individuals never seen in compositional contexts during pretraining never compose, even at 97% single-fact
    accuracy (Karmim et al., June 2026).
  - The known fixes are pretraining-scale: compositional exposure during training, or architectures that feed late
    states back to early layers (Back Attention, DiscoLoop).
- **Every neighbouring problem has several 2025–2026 papers:** accumulation, deletion, repeated updates of the same
  fact, and word-level lookup memories.
- **This project's unique asset** (exact, order-independent, certified, exactly deletable memory) does not touch
  any of the mechanisms above. Retrieval into the context has the same accountability properties.

I found no opening of breakthrough size for this project in "facts learned after deployment". The rest of this
report gives the evidence.

## 1. What this project's own tests established

| Test | Finding | Consequence |
|---|---|---|
| 5 (GPT-J-6B, MQuAKE-CF-3k-v2, all 2,764 edits) | Multi-hop with a step-by-step chain: 9.4% against 23.4% for the unedited model on its original answers (use ratio 0.40). Pasting the matched fact into each step: 8.4%. Asked directly, without the chain: 2.5% | Delivering the value inside the model is not what limits use: pasting the same fact does no better. The chain and the lookup under the model's own wording are the limit. |
| 6 (Qwen2.5-0.5B and 1.5B) | A name's identity is readable only in the early, word-level layers. Finding stored names unaided: 63–78%, against a 90% line. The hidden middle entity is not readable from the question's last token (2.8–4.0%; the probe's control is weak in the middle layers) | The only reliable native key is word-level, and word-level keys fire only on entities written as tokens. |
| Label audit | The v2 file tests 4–5 used has no contaminated labels, against 33.3% in the original CF-3k | Test 5's numbers stand. |

## 2. The mechanism, as the literature now explains it

A new fact is used only where the model can address it at the moment it needs it. There are three regimes:

| Regime | What happens | Evidence |
|---|---|---|
| Step-by-step reasoning (CoT) | Every intermediate entity is written out as tokens, so each step is a first-step lookup on a named entity. Composition reduces to recalling each fact in free generation, times overriding the old belief. | Separately learned facts compose with CoT but not without it (Balesni et al.). ENGRAFT's two-fact composition (4 of 83) follows its single-fact free recall ("the weak point is free generation, not putting two facts together"). CODE cuts self-refutation (the model negating its own new fact) from ~95% to 1.8%. |
| Latent, new fact is the first step, pointing to a known entity | The known entity has representations learned in pretraining, so the model can continue from it internally. | "Kevin's favorite programming language is Python" (fine-tuned) + "who created it?" works often without CoT (Balesni et al., Finding 4). |
| Latent, new fact is a later step, or both facts are new | The fact has to be found from an entity that exists only as a mid-layer state at another position. New facts are not stored under that state. Forcing the storage order across layers does not help. | Two-hop curse: chance level (Balesni et al., Findings 1–2). No transfer to individuals without compositional exposure (Karmim et al.). Relocating stored representations to the right layers recovers 58–75% of the lost headroom (Knowing–Using Gap), which confirms that this is where the failure lies. |

The third regime is exactly why test 6's native addressing could not have worked from frozen states. It is also why
word-level memories (Engram rows, this project's slots) compose only when the reasoning is written out.

## 3. Methods that make new facts usable, with their setups

| Method | Date | Setting | Result |
|---|---|---|---|
| CaKE | Mar 2025 | Circuit-aware curated training data per edit | +20% average multi-hop on MQuAKE |
| PropMEND | Jun 2025 | Meta-learned hypernetwork that turns a fact's gradient into a propagating update | 22.4% vs 12.7% on RippleEdit questions whose answer is not stated in the fact; the gain shrinks on unseen relations |
| Knowing–Using Gap | Jul 2026 | Fine-tuning; self-patching diagnosis | A fixed relocation heuristic recovers 58–75% of oracle headroom |
| CODE | May 2026 | Qwen2.5-7B / Llama-3.1-8B; MQuAKE-CF-3k-v2 and -T; 400 two-hop cases, one case edited at a time (a 90-edit batch mode exists); LoRA; explanations of each change written by DeepSeek with Wikidata grounding; LLM judge | Multi-hop up to 83.5% (CF), 85.3% (T); self-refutation 1.8% |
| HPSE | Aug 2026 | Self-distillation from the same model with the passage in context; plugs into gradient-based editors | Facts become atomically queryable and chainable; under continual editing the gain persists in 29 of 32 settings |
| Retrieval in structure (MeLLo, CHECK, GMeLLo, G-Walk) | 2023–2025 | External store, explicit sub-questions | 14–57% on MQuAKE variants |
| This project (test 5) | Sep 2026 | Keyed memory, all 2,764 edits, GPT-J, chain | 9.4% |

The numbers are not directly comparable: models, edit counts and scoring differ. But the direction is clear.
The frontier of usable new knowledge is self-distillation on compositional data generated around each new fact. In
Karmim et al.'s terms, that means supplying the compositional exposure after the fact.

## 4. The neighbouring problems are just as busy

| Problem | 2025–2026 work |
|---|---|
| Word-level lookup memories inside the model | Engram (Jan), NGM (May), Lngram (May), TF-Engram (Jul), User as Engram (Jun; facts as table rows plus one shared reasoning adapter), ENGRAFT (Sep) |
| Deletion that really removes | Illusion of Erasure (Jun): edits suppress, the old logic survives in hidden reasoning. Suppressed, Not Erased (Sep). Reversible routed LoRA (Mar), 99.9% rollback. Multi-hop unlearning benchmarks (GONE, Unlearning Mirage, deep unlearning) |
| Repeated updates of the same fact | Conflict-resolving multi-update editing (Feb). Retrieval bias under multiple in-context updates (Mar). RAG vs learning under continuous drift (Apr). PRISM Edit (Jul) |
| Many updated facts at once | TRACK (EACL 2026): supplying more updated facts can make multi-step reasoning worse than supplying none |

## 5. What this means for the project

From first principles, a parametric fact memory is justified over retrieval only if it gives something retrieval
cannot. The candidates are:
- **Native, latent use:** blocked for later-step facts by the mechanism in section 2, unless the model is trained
  on compositions or looped, and both are pretraining-scale.
- **Accountability (exact deletion, audit, order independence):** retrieval into the context already has these;
  deleting the text deletes the fact.
- **Context cost:** real, but it shrinks as models read context better (test 5: pasting the matched fact went from
  10.9% at 0.5B to 84.0% at 6B on the question form).

The exact joint solve, the certificate and exact deletion remain correct and apparently new as a package (see the
earlier report). They are a memory result, not a route to AGI.

## 6. Options

1. **Bank the memory result.** Phase 1 of the earlier plan: a lifelong-editing paper on modern 7–8B models under
   free-generation scoring, with a same-lookup retrieval baseline. Modest, honest and publishable. It needs rented
   GPU time: roughly 3–5 H100-hours per model and stream.
2. **Close the facts line and examine learning from experience, meaning skills.** This is the bottleneck the
   earlier report ranked first (CL-Bench: frontier systems capture 25.4% of the learnable gain). It deserves the
   same depth of prior-art check before anything is proposed. It is also compute-hungry and lab-dominated, which
   has to be weighed first.
3. **Stop.**

These experiments were considered and rejected, and why:
- **Learned canonicaliser:** it reproduces Entities as Experts, KBLaM and Lngram.
- **Latent use of first-step edits delivered by the memory:** it would replicate Balesni et al.'s Finding 4 in a
  different medium.
- **A shared "use adapter" over the keyed memory:** this is User as Engram's design. Its only gain over retrieval
  would be context cost.
- **Repeated updates with exact rewrites:** crowded (section 4), and retrieval handles the latest version exactly
  too.

## Sources

- CODE: [arXiv 2605.28303](https://arxiv.org/abs/2605.28303), [code](https://github.com/CrashBugger/CODE)
- HPSE: [arXiv 2608.11660](https://arxiv.org/abs/2608.11660)
- CaKE: [arXiv 2503.16356](https://huggingface.co/papers/2503.16356)
- PropMEND: [arXiv 2506.08920](https://huggingface.co/papers/2506.08920)
- Knowing–Using Gap: [arXiv 2607.08393](https://huggingface.co/papers/2607.08393)
- Lessons from Studying Two-Hop Latent Reasoning (two-hop curse): [arXiv 2411.16353](https://arxiv.org/abs/2411.16353)
- Multi-Hop Knowledge Composition is Bound by Pretraining Exposure: [arXiv 2606.09338](https://huggingface.co/papers/2606.09338)
- Identity Bridge: [arXiv 2509.24653](https://arxiv.org/abs/2509.24653)
- Back Attention: [arXiv 2502.10835](https://huggingface.co/papers/2502.10835)
- DiscoLoop: [arXiv 2607.00341](https://huggingface.co/papers/2607.00341)
- Hopping Too Late: [arXiv 2406.12775](https://arxiv.org/abs/2406.12775)
- ENGRAFT: [github.com/fulvian/engraft-ngram](https://github.com/fulvian/engraft-ngram)
- User as Engram: [arXiv 2606.19172](https://arxiv.org/abs/2606.19172), [code](https://github.com/19PINE-AI/user-as-engram)
- NGM [2605.16893](https://huggingface.co/papers/2605.16893); Lngram [2605.24869](https://huggingface.co/papers/2605.24869); TF-Engram [2607.07388](https://huggingface.co/papers/2607.07388); Engram [2601.07372](https://huggingface.co/papers/2601.07372)
- TRACK: [arXiv 2601.15495](https://arxiv.org/abs/2601.15495)
- Exposing the Illusion of Erasure: [arXiv 2606.23276](https://arxiv.org/html/2606.23276); Suppressed, Not Erased: [arXiv 2609.18985](https://arxiv.org/html/2609.18985); Reversible routed LoRA: [arXiv 2603.11239](https://arxiv.org/pdf/2603.11239); Breaking Chains: [arXiv 2410.13274](https://huggingface.co/papers/2410.13274)
- Multiple updates: [arXiv 2602.03696](https://arxiv.org/pdf/2602.03696), [arXiv 2603.12271](https://arxiv.org/html/2603.12271v1), [arXiv 2604.05096](https://arxiv.org/pdf/2604.05096), PRISM Edit [2607.11327](https://huggingface.co/papers/2607.11327)
- CL-Bench and the earlier plan: [AGI bottlenecks and research gaps 2026](AGI%20bottlenecks%20and%20research%20gaps%202026.md)
