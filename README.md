# nichtec

Two write-ups, each with the code and raw results behind every number:

- **[AGI_BOTTLENECK.md](AGI_BOTTLENECK.md)**: why AI can't learn in the order experience arrives, and a learner (SPARC) that can.
- **[COUNTING_AI.md](COUNTING_AI.md)**: how far a language model gets when training is nothing but counting on a CPU (no GPU, no gradient descent), measured on Penn Treebank against a neural network on the same machine.

The thesis: the deepest gap between today's AI and a human-like learner is that neural networks
cannot learn from experience *in the order it arrives*. Their gradient updates do not commute, so
learning B overwrites A. That one mechanical fact forces offline training on shuffled
trillion-token datasets, frozen weights at deployment, and everything that follows from those.

This repo contains the argument, the mechanism, a learner built to remove the problem (SPARC:
sparse, local, commutative memory over a consolidating representation), and measurements against
standard methods, all in NumPy on a CPU.

```bash
pip install numpy matplotlib
python3 run_experiments.py all   # all experiments (~1-2 h on 4 cores)
python3 report.py                # -> results/summary.md and results/*.png
```

| Path | What |
|---|---|
| `AGI_BOTTLENECK.md` | The full write-up: diagnosis, principle, results, limitations, prior art, research program |
| `sparc/model.py` | SPARC feature layer (consolidating, recruiting, winner-take-all patch units) |
| `sparc/baselines.py` | Backprop MLP, EWC, replay, NCM, streaming LDA, fly model, Hebbian and delta-rule readouts |
| `sparc/sleep.py` | "Sleep": generative replay from commutative memory into a backprop learner |
| `sparc/protocol.py` | Streams, single-pass training loops, metrics |
| `results/` | Raw JSON per run, aggregated tables, figures, ablations |
| `alm/`, `run_lm.py` | Counting-only language model: Kneser-Ney, cache, skip-grams, count-derived vectors, closed-form predictor, soft n-gram, CPU LSTM baseline |
| `results/lm/` | Per-token probabilities, timings, mixtures, exactness checks, analysis figure |
| `run_scale.py`, `alm/kn_fast.py`, `alm/imdb.py` | Data-scaling test (0.3M to 10M words of IMDB reviews): counting vs LSTM |
| `results/scale/` | Scale-test probabilities, timings, table (`scale.json`) and figure |
| `FIXING_THE_BOTTLENECKS.md`, `memsim/` | Fixes for a phone AI that learns into its own weights; simulations of the memory |
| `reallm/memtest.py`, `results/reallm/` | Real-model test 1 (Qwen2.5-0.5B, laptop GPU): last-position keys, four write rules |
| `reallm/memtest2.py`, `results/reallm2/` | Real-model test 2: person x relation keys, novelty gate, targets without a backward pass |
| `reallm/memtest3.py`, `results/reallm3/` | Real-model test 3: canonical snapped keys; the same questions answered from the prompt vs from memory |
| `reallm/memtest4.py`, `results/reallm4/`, `results/reallm4b/`, `results/reallm4c/` | Real-world test 4: MQuAKE (real Wikidata edits), subject and relation found by the system, compared with batch editing, GRACE and retrieval (first run; fixed calibration; fixed locality measurement) |
| `reallm/modal_gptj.py`, `results/reallm4_gptj/`, `results/reallm4_gptj_t/` | Test 5: test 4 on GPT-J-6B on one rented H100 (Modal), budget-capped, pre-registered in `FIXING_THE_BOTTLENECKS.md`; MQuAKE-CF and MQuAKE-T results |
| `reports/AGI bottlenecks and research gaps 2026.md`, `research_notes/AGI bottlenecks and research gaps 2026/` | What blocks AGI as of September 2026, what frontier labs cannot do, the 2024–2026 state of the art in continual learning and knowledge editing, where this project stands and a phased plan with pass and stop lines |
| `reallm/keytest.py`, `reallm/remastered_check.py` | Test 6: can the model's own hidden states find a stored name under unseen wordings and stay silent otherwise (results in `results/keytest_*`: no, 62.8% / 78.4% against a 90% line); whether MQuAKE's labels are right when every edit is applied at once (`results/mquake_audit/`: original CF-3k 33.3% contaminated, the v2 file tests 4–5 used 0%) |
