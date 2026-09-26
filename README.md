# nichtec

**Start here: [AGI_BOTTLENECK.md](AGI_BOTTLENECK.md)**

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
