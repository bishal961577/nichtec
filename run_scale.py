"""Data-scaling test: does counting's gap to a neural network shrink or grow with more data?

Corpus: IMDB reviews (alm/imdb.py). Training sets are the first 0.3M, 1M, 3M and 10M words of
the shuffled training stream. Every model is scored on the same 10,000 randomly chosen positions
of the validation stream (used only to fit mixture weights and for early stopping) and the same
10,000 positions of the test stream (reported). Hyperparameters are the ones chosen on Penn
Treebank, unchanged.

    python3 run_scale.py count 1000000   # counting components for one training size
    python3 run_scale.py lstm 1000000 [H] # LSTM for one training size (hidden size H, default 200)
    python3 run_scale.py report          # tables + figure
"""
import json
import os
import sys
import time

import numpy as np

from alm import imdb
from alm.closedform import ClosedFormLM, ContextFeatures, count_embeddings, target_probs_at
from alm.kn import perplexity
from alm.kn_fast import FastKN
from alm.mix import cache_probs, em_weights, mix_ppl
from alm.softgram import SoftGramStream

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "scale")
SIZES = [300_000, 1_000_000, 3_000_000, 10_000_000]
EPOCHS = {300_000: 20, 1_000_000: 13, 3_000_000: 8, 10_000_000: 4}
N_EVAL = 10_000


def positions():
    tr, va, te = imdb.load()
    rng = np.random.default_rng(7)
    pv = np.sort(rng.choice(np.arange(1, len(va)), N_EVAL, replace=False))
    pt = np.sort(rng.choice(np.arange(1, len(te)), N_EVAL, replace=False))
    return tr, va, te, pv, pt


def save(name, arr):
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, name + ".npy"), arr)


def count(n):
    tr, va, te, pv, pt = positions()
    x = tr[:n]
    V = imdb.V
    T = {}
    t = time.time(); kn = FastKN(5, V).fit(x); T["kn5_train"] = time.time() - t
    save(f"{n}_kn5_valid", kn.probs(va, pv)); save(f"{n}_kn5_test", kn.probs(te, pt))
    save(f"{n}_cache_valid", cache_probs(va, V, 200)[pv]); save(f"{n}_cache_test", cache_probs(te, V, 200)[pt])
    t = time.time(); E = count_embeddings(x, V, d=128); T["embeddings_train"] = time.time() - t
    t = time.time(); lm = ClosedFormLM(ContextFeatures(E, n_prev=4, m=1024), V); lm.add(x); lm.solve(1e-2, "ridge"); T["ridge_train"] = time.time() - t
    save(f"{n}_ridge_valid", target_probs_at(lm, va, pv)); save(f"{n}_ridge_test", target_probs_at(lm, te, pt))
    del lm
    t = time.time(); sg = SoftGramStream(x, E, (1, 0.5, 0.25, 0.125), V); T["soft_train"] = time.time() - t
    t = time.time(); save(f"{n}_soft_valid", sg.target_probs(va, pv, 0.1)); save(f"{n}_soft_test", sg.target_probs(te, pt, 0.1)); T["soft_score_20k"] = time.time() - t
    T = {k: round(v, 1) for k, v in T.items()}
    json.dump(T, open(os.path.join(OUT, f"{n}_count_timings.json"), "w"), indent=1)
    print(n, T, flush=True)


def lstm(n, H=200):
    from alm.lstm import token_probs_h, train_plateau
    tr, va, te, pv, pt = positions()
    t = time.time()
    p, curve = train_plateau(tr[:n], va[:100_000], imdb.V, H=H, rate=0.2, max_epochs=EPOCHS[n],
                             log=lambda s: print(n, s, flush=True))
    secs = time.time() - t
    save(f"{n}_lstm{H}_valid", token_probs_h(p, va, H)[pv]); save(f"{n}_lstm{H}_test", token_probs_h(p, te, H)[pt])
    json.dump({"curve": curve, "train_seconds": secs}, open(os.path.join(OUT, f"{n}_lstm{H}.json"), "w"), indent=1)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comps = ["kn5", "cache", "ridge", "soft"]
    rows = []
    for n in SIZES:
        f = lambda c, s: os.path.join(OUT, f"{n}_{c}_{s}.npy")
        if not all(os.path.exists(f(c, "test")) for c in comps):
            continue
        Pv = np.stack([np.load(f(c, "valid")) for c in comps], 1)
        Pt = np.stack([np.load(f(c, "test")) for c in comps], 1)
        lam = em_weights(Pv)
        r = {"train_words": n, "kn5": perplexity(Pt[:, 0]), "kn5+cache": mix_ppl(Pt[:, :2], em_weights(Pv[:, :2])),
             "counting": mix_ppl(Pt, lam), "counting_weights": dict(zip(comps, lam.round(3).tolist()))}
        for H in (200, 400):
            if os.path.exists(f(f"lstm{H}", "test")):
                lv, lt = np.load(f(f"lstm{H}", "valid")), np.load(f(f"lstm{H}", "test"))
                r[f"lstm{H}"] = perplexity(lt)
                both = em_weights(np.concatenate([Pv, lv[:, None]], 1))
                r[f"counting+lstm{H}"] = mix_ppl(np.concatenate([Pt, lt[:, None]], 1), both)
                r[f"gap{H}"] = r["counting"] / r[f"lstm{H}"]
                r[f"lstm{H}_train_minutes"] = json.load(open(os.path.join(OUT, f"{n}_lstm{H}.json")))["train_seconds"] / 60
        tj = os.path.join(OUT, f"{n}_count_timings.json")
        if os.path.exists(tj):
            T = json.load(open(tj))
            r["counting_train_minutes"] = (T["kn5_train"] + T["embeddings_train"] + T["ridge_train"] + T["soft_train"]) / 60
        rows.append(r)
    for r in rows:
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()})
    json.dump(rows, open(os.path.join(OUT, "scale.json"), "w"), indent=1)
    if not rows:
        return
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ns = [r["train_words"] for r in rows]
    series = [("kn5", "Kneser-Ney 5-gram"), ("counting", "Counting only (all components)"), ("lstm200", "LSTM 2x200, backprop"),
              ("lstm400", "LSTM 2x400, backprop"), ("counting+lstm200", "Counting + LSTM 2x200 mixed")]
    ax = axes[0]
    for (key, lab), col in zip(series, C):
        pts = [(r["train_words"], r[key]) for r in rows if key in r]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ns, [f"{n / 1e6:g}M" for n in ns]); ax.minorticks_off()
    yt = [60, 80, 100, 150, 200, 300, 400]
    ax.set_yticks(yt, [str(v) for v in yt])
    ax.set_xlabel("training words (log scale)"); ax.set_ylabel("test perplexity (log scale, lower is better)")
    ax.set_title("More data: how each approach improves", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(color=GRID)
    ax = axes[1]
    for H, col in ((200, C[2]), (400, C[3])):
        pts = [(r["train_words"], r[f"gap{H}"]) for r in rows if f"gap{H}" in r]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=col, lw=2, marker="o", ms=6, label=f"counting / LSTM 2x{H}")
            for x_, y_ in pts:
                ax.text(x_, y_ + 0.02, f"{y_:.2f}", ha="center", fontsize=8, color=INK)
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.text(ns[0], 1.01, "equal quality", fontsize=8, color=INK2, va="bottom")
    ax.set_xscale("log"); ax.set_xticks(ns, [f"{n / 1e6:g}M" for n in ns]); ax.minorticks_off()
    ax.set_xlabel("training words (log scale)"); ax.set_ylabel("perplexity ratio (above 1: counting is worse)")
    ax.set_title("The gap: does counting fall behind as data grows?", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, frameon=False); ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_scale.png"), dpi=150)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "count":
        count(int(sys.argv[2]))
    elif cmd == "lstm":
        lstm(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 200)
    elif cmd == "report":
        report()
