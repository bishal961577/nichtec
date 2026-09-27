"""Additive (count-only, no gradient descent) language model on Penn Treebank.

    python3 run_lm.py components   # per-token probabilities of every component -> results/lm/
    python3 run_lm.py mix          # EM mixture weights on validation, test perplexities
    python3 run_lm.py lstm         # CPU LSTM baseline (Zaremba 'small'), time-stamped curve
    python3 run_lm.py exact        # merge / unlearn exactness checks
"""
import json
import os
import sys
import time

import numpy as np

from alm import data
from alm.closedform import ClosedFormLM, ContextFeatures, count_embeddings
from alm.kn import KneserNey, perplexity
from alm.mix import cache_probs, em_weights, mix_ppl
from alm.skip import SkipGram
from alm.softgram import SoftGram, keys

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "lm")
SOFT = {"n_prev": 4, "weights": (1, 0.5, 0.25, 0.125), "tau": 0.1}  # chosen on the first 8k validation tokens


def log_time(timings, name, t0):
    timings[name] = round(time.time() - t0, 1)
    print(f"{name}: {timings[name]} s", flush=True)


def components():
    os.makedirs(OUT, exist_ok=True)
    tr, va, te, vocab = data.load()
    V = len(vocab)
    T = {}
    t = time.time(); kn = KneserNey(5, V); kn.add(tr); kn.finalize(); log_time(T, "train_kn5", t)
    t = time.time(); np.save(f"{OUT}/kn5_valid.npy", kn.stream_probs(va)); np.save(f"{OUT}/kn5_test.npy", kn.stream_probs(te)); log_time(T, "eval_kn5", t)
    for w in (200,):
        np.save(f"{OUT}/cache_valid.npy", cache_probs(va, V, w)); np.save(f"{OUT}/cache_test.npy", cache_probs(te, V, w))
    for k in (2, 3):
        t = time.time(); sk = SkipGram(k, V).add(tr).finalize(); log_time(T, f"train_skip{k}", t)
        np.save(f"{OUT}/skip{k}_valid.npy", sk.stream_probs(va)); np.save(f"{OUT}/skip{k}_test.npy", sk.stream_probs(te))
    t = time.time(); E = count_embeddings(tr, V, d=128); log_time(T, "train_embeddings", t)
    t = time.time(); f = ContextFeatures(E, n_prev=4, m=1024); lm = ClosedFormLM(f, V); lm.add(tr); log_time(T, "train_ridge_stats", t)
    t = time.time(); lm.solve(1e-2, "ridge"); log_time(T, "train_ridge_solve", t)
    t = time.time(); np.save(f"{OUT}/ridge_valid.npy", lm.target_probs(va, floor=0.1)); np.save(f"{OUT}/ridge_test.npy", lm.target_probs(te, floor=0.1)); log_time(T, "eval_ridge", t)
    t = time.time(); Kt = keys(tr, E, SOFT["n_prev"], SOFT["weights"]); sg = SoftGram(Kt, tr, V); log_time(T, "train_softgram", t)
    for split, ids in (("valid", va), ("test", te)):
        t = time.time(); np.save(f"{OUT}/soft_{split}.npy", sg.target_probs(keys(ids, E, SOFT["n_prev"], SOFT["weights"]), ids, SOFT["tau"])); log_time(T, f"eval_soft_{split}", t)
    json.dump(T, open(f"{OUT}/timings.json", "w"), indent=1)


def mix():
    names = ["kn5", "cache", "skip2", "skip3", "ridge", "soft"]
    P = {s: np.stack([np.load(f"{OUT}/{n}_{s}.npy") for n in names], 1) for s in ("valid", "test")}
    rows = []
    combos = [["kn5"], ["kn5", "cache"], ["kn5", "cache", "skip2", "skip3"], ["kn5", "cache", "ridge"],
              ["kn5", "cache", "soft"], ["kn5", "cache", "skip2", "skip3", "ridge", "soft"]]
    for n in names:
        rows.append({"model": n + " alone", "valid": perplexity(P["valid"][:, names.index(n)]), "test": perplexity(P["test"][:, names.index(n)])})
    for c in combos:
        idx = [names.index(n) for n in c]
        lam = em_weights(P["valid"][:, idx])
        rows.append({"model": " + ".join(c), "valid": mix_ppl(P["valid"][:, idx], lam), "test": mix_ppl(P["test"][:, idx], lam),
                     "weights": dict(zip(c, lam.round(3).tolist()))})
    for r in rows:
        print(f"{r['model']:45s} valid {r['valid']:7.1f}  test {r['test']:7.1f}  {r.get('weights', '')}")
    json.dump(rows, open(f"{OUT}/mix.json", "w"), indent=1)


def lstm():
    from alm.lstm import train, evaluate
    tr, va, te, vocab = data.load()
    budget = float(os.environ.get("LSTM_BUDGET", 5400))
    lines = []
    p, curve = train(tr, va, len(vocab), epochs=13, time_budget=budget, log=lambda s: (print(s, flush=True), lines.append(s)))
    test = evaluate(p, te)
    from alm.lstm import token_probs
    np.save(f"{OUT}/lstm_valid.npy", np.concatenate([[1.0 / len(vocab)], token_probs(p, va)]))
    np.save(f"{OUT}/lstm_test.npy", np.concatenate([[1.0 / len(vocab)], token_probs(p, te)]))
    json.dump({"curve": curve, "test_ppl_at_end": test}, open(f"{OUT}/lstm.json", "w"), indent=1)
    print("test", test)


if __name__ == "__main__" and sys.argv[1] not in ("exact", "analyze", "bucketed"):
    {"components": components, "mix": mix, "lstm": lstm}[sys.argv[1]]()


def exact():
    """Merge and unlearn checks. Both must reproduce retraining bit-for-bit."""
    tr, va, te, vocab = data.load()
    V, N = len(vocab), 5
    res = {}
    t = time.time(); full = KneserNey(N, V).add(tr).finalize(); ref = full.stream_probs(te); t_full = time.time() - t
    # 1) four people each count a quarter of the data on their own machine; the counts are added
    cuts = np.linspace(0, len(tr), 5).astype(int)
    merged = KneserNey(N, V)
    for a, b in zip(cuts[:-1], cuts[1:]):
        lead = max(0, a - (N - 1))
        merged.merge(KneserNey(N, V).add(tr[lead:b], start=a - lead))
    merged.finalize()
    res["merge_4_shards_max_abs_diff"] = float(np.abs(merged.stream_probs(te) - ref).max())
    # 2) unlearn a 20,000-token document from the middle of the corpus
    a, b = 400_000, 420_000
    t = time.time()
    forget = KneserNey(N, V)
    lo, hi = a - (N - 1), b + (N - 1)
    forget.add(tr[lo:hi])                     # every k-gram touching the document ...
    forget.add(tr[lo:a], sign=-1)             # ... minus those entirely before it
    forget.add(tr[b:hi], sign=-1)             # ... minus those entirely after it
    full.merge(forget, sign=-1)
    full.finalize()
    t_unlearn = time.time() - t
    retrained = KneserNey(N, V).add(tr[:a]).add(tr[b:]).finalize()   # two separate streams, no bridging n-grams
    res["unlearn_max_abs_diff_vs_retrain"] = float(np.abs(full.stream_probs(te) - retrained.stream_probs(te)).max())
    res["unlearn_seconds"] = round(t_unlearn, 2)
    res["full_train_seconds"] = round(t_full, 2)
    res["test_ppl_before"] = perplexity(ref)
    res["test_ppl_after_unlearn"] = perplexity(full.stream_probs(te))
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{OUT}/exact.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "exact":
    exact()


def analyze():
    """Where does counting lose to the neural net? Bucket test tokens by how often their
    two-word context occurred in training, and compare perplexities per bucket."""
    from collections import Counter

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr, va, te, vocab = data.load()
    names = ["kn5", "cache", "skip2", "skip3", "ridge", "soft"]
    Pv = np.stack([np.load(f"{OUT}/{n}_valid.npy") for n in names], 1)
    Pt = np.stack([np.load(f"{OUT}/{n}_test.npy") for n in names], 1)
    lam = em_weights(Pv)
    add_t, add_v = Pt @ lam, Pv @ lam
    lstm_t, lstm_v = np.load(f"{OUT}/lstm_test.npy"), np.load(f"{OUT}/lstm_valid.npy")
    both_v = np.stack([add_v, lstm_v], 1)
    mlam = em_weights(both_v)
    both_t = np.stack([add_t, lstm_t], 1) @ mlam
    kn_t = Pt[:, 0]
    bigrams = Counter(zip(tr[:-1].tolist(), tr[1:].tolist()))
    ctx = np.array([bigrams.get((int(te[t - 2]), int(te[t - 1])), 0) if t >= 2 else 0 for t in range(len(te))])
    buckets = [("never seen", ctx == 0), ("seen 1-4 times", (ctx >= 1) & (ctx <= 4)),
               ("seen 5-49 times", (ctx >= 5) & (ctx <= 49)), ("seen 50+ times", ctx >= 50)]
    rows = []
    for name, m in buckets:
        rows.append({"context": name, "share_of_tokens": float(m.mean()), "kn5": perplexity(kn_t[m]),
                     "additive": perplexity(add_t[m]), "lstm": perplexity(lstm_t[m]), "additive+lstm": perplexity(both_t[m])})
    rows.append({"context": "all", "share_of_tokens": 1.0, "kn5": perplexity(kn_t), "additive": perplexity(add_t),
                 "lstm": perplexity(lstm_t), "additive+lstm": perplexity(both_t)})
    for r in rows:
        print(r)
    json.dump({"buckets": rows, "additive_weights": dict(zip(names, lam.round(3).tolist())),
               "additive_lstm_weights": mlam.round(3).tolist()}, open(f"{OUT}/analysis.json", "w"), indent=1)

    # figure: perplexity per context bucket
    C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    x = np.arange(len(buckets))
    series = (("kn5", "Kneser-Ney 5-gram"), ("additive", "Counting only (this work)"), ("lstm", "LSTM, backprop on CPU"),
              ("additive+lstm", "Counting + LSTM mixed"))
    w = 0.2
    for i, (key, lab) in enumerate(series):
        vals = [r[key] for r in rows[:-1]]
        ax.bar(x + (i - 1.5) * w, vals, w - 0.03, color=C[i], label=lab)
        for xi, v in zip(x, vals):
            ax.text(xi + (i - 1.5) * w, v + 2, f"{v:.0f}", ha="center", fontsize=7, color=INK)
    ax.set_ylim(0, 200)
    ax.set_xticks(x, [f"{r['context']}\n({r['share_of_tokens'] * 100:.0f}% of tokens)" for r in rows[:-1]], fontsize=8)
    ax.set_xlabel("how often the previous two words occurred together in training")
    ax.set_ylabel("test perplexity (lower is better)")
    ax.set_title("Counting wins on rare contexts, the LSTM on common ones", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper right")
    ax.grid(axis="y", color=GRID)
    # right: perplexity vs CPU seconds
    ax = axes[1]
    T = json.load(open(f"{OUT}/timings.json"))
    L = json.load(open(f"{OUT}/lstm.json"))
    cur = np.array(L["curve"])
    ax.plot(cur[:, 1], cur[:, 2], color=C[2], lw=2, marker="o", ms=5, label="LSTM, after each epoch")
    kn_time = T["train_kn5"]
    add_time = kn_time + T["train_skip2"] + T["train_skip3"] + T["train_embeddings"] + T["train_ridge_stats"] + T["train_ridge_solve"] + T["train_softgram"]
    mixv = json.load(open(f"{OUT}/mix.json"))
    kn_val = [r for r in mixv if r["model"] == "kn5"][0]["valid"]
    all_val = [r for r in mixv if r["model"].startswith("kn5 + cache + skip2 + skip3 + ridge + soft")][0]["valid"]
    ax.scatter([kn_time], [kn_val], color=C[0], s=60, zorder=3, label="Kneser-Ney 5-gram")
    ax.scatter([add_time], [all_val], color=C[1], s=60, zorder=3, label="Counting only (this work)")
    ax.scatter([add_time + cur[-1, 1]], [perplexity(both_v @ mlam)], color=C[3], s=60, zorder=3, label="Counting + LSTM mixed")
    ax.set_xscale("log")
    ax.set_ylim(80, 185)
    ax.set_xlabel("training time on this 4-core CPU (seconds, log scale)")
    ax.set_ylabel("validation perplexity")
    ax.set_title("Training cost vs quality (no GPU)", fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(color=GRID)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lm.png"), dpi=150)


if __name__ == "__main__" and sys.argv[1] == "analyze":
    analyze()


def bucketed():
    """Mixture weights that depend on how often the current two-word context occurred in
    training (known at prediction time). Fit per bucket by EM on validation."""
    from collections import Counter

    tr, va, te, vocab = data.load()
    names = ["kn5", "cache", "skip2", "skip3", "ridge", "soft"]
    bigrams = Counter(zip(tr[:-1].tolist(), tr[1:].tolist()))

    def bucket(ids):
        c = np.array([bigrams.get((int(ids[t - 2]), int(ids[t - 1])), 0) if t >= 2 else 0 for t in range(len(ids))])
        return np.digitize(c, [1, 5, 50])  # 0: unseen, 1: 1-4, 2: 5-49, 3: 50+

    bv, bt = bucket(va), bucket(te)
    out = {}
    for label, extra in (("additive", []), ("additive+lstm", ["lstm"])):
        cols = names + extra
        Pv = np.stack([np.load(f"{OUT}/{n}_valid.npy") for n in cols], 1)
        Pt = np.stack([np.load(f"{OUT}/{n}_test.npy") for n in cols], 1)
        glob = em_weights(Pv)
        pt = np.empty(len(te))
        W = {}
        for b in range(4):
            lam = em_weights(Pv[bv == b])
            W[int(b)] = dict(zip(cols, lam.round(3).tolist()))
            pt[bt == b] = Pt[bt == b] @ lam
        out[label] = {"global_weights_test": mix_ppl(Pt, glob), "bucketed_weights_test": perplexity(pt), "weights": W}
        print(label, "global", round(out[label]["global_weights_test"], 1), "bucketed", round(out[label]["bucketed_weights_test"], 1))
    json.dump(out, open(f"{OUT}/bucketed.json", "w"), indent=1)


if __name__ == "__main__" and sys.argv[1] == "bucketed":
    bucketed()
