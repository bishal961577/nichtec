# Does the Order of Experience Hold Back Learning on the Job? (October 2026)

This is the step after the CIFAR-10 stop. It tests one link: this repo's starting thesis (a learner whose result
depends on the order experience arrives in learns badly) against the failures CL-Bench names in today's AI agents.
They are "rigid in early, incorrect beliefs" and "misled by spurious correlations". Sections 1–6 were written and
committed before any run. Section 7, also written before any run, moves the main test to two larger Qwen3.5 models
on a GPU endpoint and adds two lines.

**Bottom line, before the run:**
- **Setup:** the same 40 experiences, five orders, three agent memories, one small model (Qwen2.5-1.5B-Instruct).
  It runs on the owner's laptop: about an hour, $0.
- **Two questions:**
  1. Does the order of experience change what an agent ends up knowing?
  2. Does an order-free tally of evidence (a ledger) remove that effect and learn more?
- **Prior art:** the ledger is published, and order effects in in-context learning have been measured before
  (section 1). What I did not find is a comparison of agent memories under adversarial orders of the same
  experience. A positive result would be a modest measurement, not a new method.
- **No forecast is made.** This project's forecasts so far are 0 for 5.

## 1. What is already known

| Finding | Source |
|---|---|
| In-context learning depends on the order of the examples: some orders come near the state of the art, others near chance | Lu et al. 2022 |
| On rule-like tasks, LLM in-context learning does better when examples arrive blocked than interleaved; learning in the weights shows the opposite | Russin, Pavlick and Frank 2024 |
| LLMs show primacy and recency effects; primacy dominates | Serial position effects, ACL Findings 2025 |
| Whether an LLM updates step by step or at the end changes its order effects | Query timing, arXiv 2608.12387 |
| Letting the model propose rules, then checking each against all the data, works better than trusting one guess | Hypothesis Search (ICLR 2024); Piriyakulkij et al. 2024; IDEA 2024 |
| Agents get stuck on early beliefs and spurious correlations; memory systems do worse than keeping the full history | CL-Bench, June 2026 |

**Correction.** Before this check I told the owner that nobody had measured order effects for AI. That was too
strong. Order effects in in-context learning are measured. What I did not find is the comparison in section 2.

## 2. First principles: where order can enter

An agent's state after its experience is a function of that experience. The state ignores order only if the
function does. The three agents differ in exactly this:

| Agent | What it keeps | How order can matter |
|---|---|---|
| A, history | Every case with its decision, in arrival order, in the prompt | Nothing is lost, so order acts only through where cases sit in the prompt. Listing the history in a fixed order removes the effect at no cost (control: A-sorted). |
| C, revise | One current rule and the last 10 cases | Lossy and path-dependent: it keeps whichever rule best fits the recent cases. Memory and notes agents work like this, and it is the kind of learner this repo's thesis says order breaks. |
| B, ledger | Every rule ever proposed, and how many of all seen cases each gets right | The tallies come out the same in any order. Only which rules get proposed depends on order, and a good rule proposed at any point wins once the evidence is in. |

B and C see the same proposals. The model writes rules from the last 10 cases and is not shown either agent's
current rule. So:
- **B against C** isolates the ledger.
- **B against A** compares the ledger with how full-history agents work today.

## 3. Design

**Items.** Four attributes, giving 576 possible items:
- colour (4 values);
- shape (4);
- size (1–9);
- material (4).

**Hidden rules.** Six two-condition rules. Each has a decoy: a one-condition rule on an attribute the true rule
does not use.

| Hidden rule | Decoy |
|---|---|
| `size >= 5 and color != 'red'` | `shape in ('circle', 'square')` |
| `shape == 'circle' or material == 'metal'` | `color in ('blue', 'green')` |
| `color in ('blue', 'green') and size <= 6` | `material in ('wood', 'glass')` |
| `material != 'wood' and shape != 'star'` | `size >= 5` |
| `size > 3 and shape in ('square', 'triangle')` | `color in ('red', 'yellow')` |
| `color == 'yellow' or size >= 7` | `material in ('metal', 'plastic')` |

**Tasks.** Each rule is run with two independent draws of items, giving 12 tasks. A task has:
- a stream of 40 cases (20 accepted, 20 rejected);
- a separate test set of 40 items (balanced, with the decoy right on exactly half).

**The misleading block.** Ten of the 40 stream cases (5 accepted, 5 rejected) fit the decoy as well as the true
rule. On the other 30 the decoy is right about half the time (14 of 30). Over the whole stream it is right 60% of
the time.

**Orders.** The same 40 cases each time:
- three random shuffles;
- decoy first (the block, then the rest);
- decoy last (the rest, then the block).

**Agent A.**
- It sees all past cases with their decisions and answers yes or no for a new item, from the model's next-word
  probabilities.
- On the test set it says yes for the items scoring above the median, since its experience shows half of the
  items are accepted. Accuracy at a threshold of zero is also reported.

**Agents B and C.**
- **Proposals:** after every 5 cases (8 rounds), the model sees the last 10 cases and writes 4 rules as Python
  expressions. Sampling: temperature 0.8, top-p 0.95, top-k 50, at most 64 new tokens.
- **Safety:** each expression is checked against a whitelist before it runs: comparisons, `and`/`or`/`not`, `in`,
  `+`, `-` and `%` only, with no calls or attributes.
- **B** answers with the rule that gets the most of all seen cases right. Ties go to the shortest, then the first
  alphabetically.
- **C** keeps one rule and swaps it for a new proposal only if the new one fits the last 10 cases strictly better.

**Controls.** A with no history (zero-shot), and A with the history sorted in a fixed order.

**Model.** Qwen/Qwen2.5-1.5B-Instruct, fp16, on an RTX 3050 with 6 GB.

**Measures,** on the test set after all 40 cases:
- **Learning:** accuracy averaged over the three random orders and the 12 tasks, minus 50 (chance on the balanced
  test set).
- **Order gap:** the larger of two averages over tasks:
  - random-order accuracy minus decoy-first accuracy;
  - random-order accuracy minus decoy-last accuracy.
- **Proposer check:** the share of the 60 streams whose proposals include a good rule. A good rule agrees with the
  true rule on at least 95% of all 576 items.

## 4. Lines, fixed before the run

1. **Check that A learns:** A's learning is at least 5 points. If not, A's order gap means nothing at this scale,
   and the stop and pass lines are not applied.
2. **Check the proposer:** at least 25% of streams contain a good rule. If not, B and C are inconclusive because
   the laptop model is too weak. Going further would need a stronger model, a decision for the owner.
3. **Stop:** A's order gap is under 5 points. Order is not what holds back a full-history agent here, and the route
   ends.
4. **Pass:** all three hold:
   - A's order gap is at least 5;
   - B's gap is at most half of A's;
   - B's learning is at least 5 points above A's.
5. **Credit:** a pass is credited to the order-free ledger only if either holds:
   - B's learning is at least 5 points above C's;
   - C's gap is at least 5 and B's gap is at most half of C's.

   Otherwise the credit goes to writing rules as code and testing them, which is published.
6. **The trivial cure:** if A-sorted scores within 2 points of A's random-order accuracy, or better, then showing
   the history in a fixed order removes A's order damage for free. The order half of a pass then says nothing new
   about full-history agents.
7. **Secondary, reported whatever happens:** C's order gap, and B against C. If C's gap is at least 5 and B's is at
   most half of it, order hurts a bounded-memory agent and the ledger removes the damage. That would be a secondary
   finding, not a pass.

The script prints the verdict from these lines mechanically. A stop ends the route, with no "one more variant".

## 5. What each outcome would mean

| Outcome | Meaning |
|---|---|
| Stop | Order is not a cause of poor learning for a full-history agent at this scale. The repo's order thesis does not transfer to this setting. |
| Pass, credited to the ledger | Order is one real cause of poor learning from experience for this kind of agent, and an order-free tally fixes it at small scale. The next test would be a CL-Bench task with a stronger model, which costs money; the owner decides. |
| Pass, credited to code | The gain comes from proposing rules as code and testing them, which is published. The order link is not shown. |
| A check fails | Inconclusive at 1.5B. The owner decides whether a stronger model is worth paying for. |

**Scale.** One 1.5B model, synthetic rules and 40 cases. Nothing here transfers to frontier agents without a test
there.

## 6. How it runs

`brain/order_gap.py` runs everything:
- It saves `results/order_gap_<model>/summary.md` and `result.json` after every task, and prints the verdict at
  the end.
- Before the run starts, it checks that the cached history gives the same answer as one full prompt, and stops if
  not.
- `--stub` checks the bookkeeping without a model.

## 7. Amendment before any run (18 October 2026): two Qwen3.5 models on an endpoint

Nothing had run when this section was written. Why the change:
- The 1.5B model may be too weak to learn the rules or to propose them, so lines 1 and 2 could end the test
  without saying anything about order.
- The owner has GPU credit on Modal, where larger open models can be served.

**Models:**
- **Primary: Qwen/Qwen3.5-27B-FP8.** The route's stop or pass is decided by this model.
- **Secondary: Qwen/Qwen3.5-4B.** Phone-sized. Its verdict is reported, not used for the route.
- **Same family, two sizes,** so the comparison is about size, not about how the models were trained.
- **The Qwen2.5-1.5B laptop run stays optional.** If it is run, it is reported as a third size.

**How they run:**
- Each model sits behind an OpenAI-compatible endpoint (vLLM or SGLang) on Modal and is called from the owner's
  laptop, with thinking switched off on every request.
- Command: `python brain/order_gap.py --base-url <endpoint>/v1 --model Qwen/Qwen3.5-27B-FP8`, with `--check` first.
- Cost: Modal GPU time, estimated at a few dollars per model.

**What changes in the measurement:**
- **Agent A's yes-or-no score** comes from the top 20 log-probabilities of the first answer token instead of the full
  vocabulary. A side missing from the top 20 counts as 1 below the lowest listed value.
- **The cached-history check is replaced by a probe.** The model's first answer token must be yes or no, or the run
  stops.
- **Proposals** use the same sampling settings, with a fixed seed per request.
- **Calls run in parallel.** Each prompt is the same text the local model sees, wrapped by the server's chat
  template.
- Everything else is unchanged: items, rules, orders, agents, measures, and lines 1–7.

**New lines, for every model:**

8. **A must answer the question.** If more than 10% of A's answers have neither yes nor no among their top 20
   tokens, A is not measured, and lines 3 and 4 are not applied.
9. **Too easy.** If A's random-order accuracy is 95 or more, these rules cannot show order damage, and lines 3 and 4
   are not applied. The harder rule set below then runs automatically on the same model, and that model's verdict
   comes from it. If A also scores 95 or more on the harder set, the result is inconclusive for that model. There is
   no third set.

**The harder rule set,** fixed now. Each rule has three conditions, and the decoy is a condition on the fourth
attribute. Tasks, orders and the misleading block are built exactly as in section 3.

| Hidden rule | Decoy |
|---|---|
| `size >= 4 and color != 'red' and material != 'glass'` | `shape in ('circle', 'square')` |
| `(shape == 'circle' or material == 'metal') and size <= 7` | `color in ('blue', 'green')` |
| `(color in ('blue', 'green') and size % 2 == 1) or material == 'wood'` | `shape in ('triangle', 'star')` |
| `material != 'wood' and shape != 'star' and size > 2` | `color in ('red', 'yellow')` |
| `(size > 3 and shape in ('square', 'triangle')) or color == 'green'` | `material in ('metal', 'plastic')` |
| `color == 'yellow' or (size >= 6 and material in ('metal', 'glass'))` | `shape in ('circle', 'triangle')` |

**Route decision.** It comes from the primary model. If the primary is inconclusive (line 1, 8 or 9), what to do
next is the owner's decision. No new model or rule set is tried without a new pre-registration.

## Sources

- Lu et al., Fantastically Ordered Prompts: [arXiv 2104.08786](https://arxiv.org/abs/2104.08786)
- Russin, Pavlick and Frank, Human Curriculum Effects Emerge with In-Context Learning in Neural Networks: [arXiv 2402.08674](https://arxiv.org/abs/2402.08674)
- Serial Position Effects of Large Language Models: [ACL Findings 2025](https://aclanthology.org/2025.findings-acl.52.pdf)
- Query Timing Produces Opposite Positional Biases Between LLMs and Humans: [arXiv 2608.12387](https://arxiv.org/html/2608.12387)
- Hypothesis Search: [arXiv 2309.05660](https://arxiv.org/abs/2309.05660); Piriyakulkij et al., Doing Experiments and Revising Rules with Natural Language and Probabilistic Reasoning: [arXiv 2402.06025](https://arxiv.org/abs/2402.06025); IDEA: [arXiv 2408.10455](https://arxiv.org/pdf/2408.10455)
- CL-Bench: [arXiv 2606.05661](https://arxiv.org/abs/2606.05661); earlier report: [Learning from experience 2026](Learning%20from%20experience%202026.md)
