"""Aggregate results/raw/*.json into results/summary.md and results/*.png."""
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results")

# Reference categorical palette (validated for CVD separation; low-contrast slots get direct labels + tables)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
})


def load(prefix):
    rows = defaultdict(list)  # method -> list of result dicts (one per seed)
    extra = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(RAW, prefix + "*.json"))):
        for k, v in json.load(open(path)).items():
            (extra if k.startswith("_") else rows)[k].append(v)
    return rows, extra


def ms(vals):
    vals = np.asarray(vals) * 100
    return f"{vals.mean():.1f} ± {vals.std():.1f}" if len(vals) > 1 else f"{vals.mean():.1f}"


def split_table(dataset):
    rows, extra = load(f"split_{dataset}_")
    # pick EWC lambda by best ordered accuracy (tuned on the test set: generous to the baseline)
    ewc = {k: v for k, v in rows.items() if "EWC" in k}
    best_ewc = max(ewc, key=lambda k: np.mean([r["final_acc"] for r in ewc[k]])) if ewc else None
    methods = defaultdict(dict)
    for k, v in rows.items():
        if "EWC" in k and k != best_ewc:
            continue
        name, order = k.split(" | ")
        methods[name][order] = v
    order_names = [
        ("MLP (backprop)", "no", "no"), ("MLP (backprop), 5 epochs", "no", "no"),
        (best_ewc.split(" | ")[0] if best_ewc else None, "no", "no"),
        ("MLP + replay 500", "500 images", "no"), ("MLP + replay 5000", "5000 images", "no"),
        ("NCM (pixels)", "no", "yes"), ("SLDA (pixels)", "no", "no (matrix inverse at read time)"), ("Fly (pixels)", "no", "yes"),
        ("SPARC-online / ncm", "no", "yes"), ("SPARC-online / assoc", "no", "yes"), ("SPARC-online / slda", "no", "features yes; readout no (matrix inverse)"),
        ("SPARC-dev / ncm", "no", "yes"), ("SPARC-dev / assoc", "no", "yes"), ("SPARC-dev / slda", "no", "no (matrix inverse at read time)"),
        ("SPARC-dev / delta", "no", "yes (not commutative)"),
    ]
    lines = [f"| Method | Ordered stream (class-incremental) | Shuffled stream | Order gap | Forgetting (ordered) | Stores raw data | Local rules only |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for name, stores, local in order_names:
        if name is None:
            continue
        if name == "MLP (backprop), 5 epochs":
            v = rows.get("MLP (backprop) | shuffled, 5 epochs")
            if v:
                lines.append(f"| MLP (backprop), shuffled, 5 epochs *(offline upper bound)* | – | {ms([r['final_acc'] for r in v])} | – | – | all data | no |")
                summary[name] = {"shuffled": np.mean([r["final_acc"] for r in v])}
            continue
        m = methods.get(name)
        if not m:
            continue
        o = [r["final_acc"] for r in m.get("ordered", [])]
        s = [r["final_acc"] for r in m.get("shuffled", [])]
        f = [r["forgetting"] for r in m.get("ordered", [])]
        gap = f"{(np.mean(s) - np.mean(o)) * 100:+.1f}" if o and s else "–"
        lines.append(f"| {name} | {ms(o) if o else '–'} | {ms(s) if s else '–'} | {gap} | {ms(f) if f else '–'} | {stores} | {local} |")
        summary[name] = {"ordered": float(np.mean(o)) if o else None, "shuffled": float(np.mean(s)) if s else None,
                         "n_seeds": len(o), "matrix": np.mean([r["matrix"] for r in m["ordered"]], 0).tolist() if o else None}
    ewc_lines = []
    for k in sorted(ewc, key=lambda k: float(k.split()[3])):
        ewc_lines.append(f"| {k.split(' | ')[0]} | {ms([r['final_acc'] for r in ewc[k]])} |")
    return lines, summary, ewc_lines, extra


def fig_order_gap(summaries):
    show = [("MLP (backprop)", "Backprop MLP"), ("MLP + EWC", "Backprop + EWC (best λ)"),
            ("MLP + replay 500", "Backprop + replay (500 stored images)"), ("NCM (pixels)", "Nearest class mean (pixels)"),
            ("SLDA (pixels)", "Streaming LDA (pixels)"), ("Fly (pixels)", "Fly model (pixels)"),
            ("SPARC-online / assoc", "SPARC-online, Hebbian readout"), ("SPARC-online / slda", "SPARC-online, LDA readout"),
            ("SPARC-dev / assoc", "SPARC-dev, Hebbian readout"), ("SPARC-dev / slda", "SPARC-dev, LDA readout")]
    fig, axes = plt.subplots(1, len(summaries), figsize=(12, 5.6), sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(show))[::-1]
    h = 0.38
    for ax, (ds, summ) in zip(axes, summaries.items()):
        def get(prefix, key):
            hit = [k for k in summ if k.startswith(prefix)]
            return (summ[hit[0]].get(key) or 0) if hit else 0
        o = [get(p_, "ordered") for p_, _ in show]
        sh = [get(p_, "shuffled") for p_, _ in show]
        ax.barh(y + h / 2 + 0.01, [v * 100 for v in sh], h - 0.02, color=C[1], label="shuffled (i.i.d.)")
        ax.barh(y - h / 2 - 0.01, [v * 100 for v in o], h - 0.02, color=C[0], label="ordered (class by class)")
        for yi, v in zip(y, o):
            if v:
                ax.text(v * 100 + 1, yi - h / 2, f"{v * 100:.1f}", va="center", fontsize=8, color=INK)
        ub = summ.get("MLP (backprop), 5 epochs", {}).get("shuffled")
        if ub:
            ax.axvline(ub * 100, color=INK2, lw=1, ls="--")
            ax.text(ub * 100 - 1, -0.9, "offline backprop, 5 epochs", ha="right", va="center", fontsize=7.5, color=INK2)
        ax.set_xlim(0, 110)
        ax.set_ylim(-1.3, len(show) - 0.4)
        ax.set_xlabel("test accuracy on all 10 classes (%)")
        ax.set_title({"mnist": "Split-MNIST", "fashion": "Split-Fashion-MNIST"}[ds])
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [lab for _, lab in show])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Same data, same single pass: only the order changes", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, "fig1_order_gap.png"), dpi=150)


def fig_forgetting(summ, ds):
    show = [("MLP (backprop)", C[0]), ("MLP + EWC", C[1]), ("MLP + replay 500", C[2]),
            ("SPARC-online / assoc", C[3]), ("SPARC-dev / slda", C[4])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, col in show:
        hit = [k for k in summ if k.startswith(name)]
        if not hit or summ[hit[0]].get("matrix") is None:
            continue
        M = np.array(summ[hit[0]]["matrix"]) * 100
        x = np.arange(1, len(M) + 1)
        first = M[:, 0]
        avg = [M[i, : i + 1].mean() for i in range(len(M))]
        for i, (ax, series) in enumerate(zip(axes, (first, avg))):
            ax.plot(x, series, color=col, lw=2, marker="o", ms=5, label=hit[0] if i == 0 else None)
    axes[0].set_title("Accuracy on the FIRST task (digits 0/1) as later tasks are learned")
    axes[1].set_title("Average accuracy over all classes seen so far")
    for ax in axes:
        ax.set_xlabel("tasks learned")
        ax.set_ylabel("accuracy (%)")
        ax.set_xticks(range(1, 6))
        ax.set_xlim(0.8, 5.2)
        ax.set_ylim(-3, 103)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, f"fig2_forgetting_{ds}.png"), dpi=150)


def sleep_table():
    lines = []
    for ds in ("mnist", "fashion"):
        rows, _ = load(f"sleep_{ds}_")
        if not rows:
            continue
        lines += [f"\n**{ds}** (mean ± std over {len(next(iter(rows.values())))} seeds)\n",
                  "| Slow learner (2304-256-10 MLP, Adam, one pass) | Accuracy, all 10 classes | Forgetting |", "|---|---|---|"]
        for k, v in rows.items():
            lines.append(f"| {k.replace(' | ', ', ')} | {ms([r['final_acc'] for r in v])} | {ms([r['forgetting'] for r in v])} |")
    return lines


def fig_commutativity(summaries):
    """Headline: ordered vs shuffled for learners that differ in whether their updates commute."""
    fig, axes = plt.subplots(1, len(summaries), figsize=(11, 4.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (ds, summ) in zip(axes, summaries.items()):
        sleep_rows, _ = load(f"sleep_{ds}_")
        items = [
            ("Backprop MLP\n(pixels)", summ.get("MLP (backprop)")),
            ("Delta rule on\nsparse code\n(local, NOT\ncommutative)", summ.get("SPARC-dev / delta")),
            ("Hebbian counts\non same code\n(local,\ncommutative)", summ.get("SPARC-dev / assoc")),
            ("Streaming LDA\n(commutative)", summ.get("SPARC-dev / slda")),
        ]
        if sleep_rows:
            o = np.mean([r["final_acc"] for r in sleep_rows["Backprop on dev features + sleep | ordered"]])
            s_ = np.mean([r["final_acc"] for r in sleep_rows["Backprop on dev features, no sleep | shuffled"]])
            items.append(("Backprop +\nsleep replay\nfrom commutative\nmemory", {"ordered": o, "shuffled": s_}))
        items = [(n, v) for n, v in items if v]
        x = np.arange(len(items))
        w = 0.38
        for off, key, col, lab in ((-w / 2, "ordered", C[0], "ordered (class by class)"), (w / 2, "shuffled", C[1], "shuffled (i.i.d.)")):
            vals = [100 * (v.get(key) or 0) for _, v in items]
            ax.bar(x + off, vals, w - 0.03, color=col, label=lab)
            for xi, v in zip(x, vals):
                ax.text(xi + off, v + 1, f"{v:.0f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(x, [n for n, _ in items], fontsize=7.5)
        ax.set_ylim(0, 108)
        ax.set_title({"mnist": "Split-MNIST", "fashion": "Split-Fashion-MNIST"}[ds])
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("final accuracy, all 10 classes (%)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Order-invariance tracks commutativity, not sparsity or locality", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(os.path.join(OUT, "fig0_commutativity.png"), dpi=150)


def fewshot():
    lines, curves = [], {}
    for ds in ("mnist", "fashion"):
        files = sorted(glob.glob(os.path.join(RAW, f"fewshot_{ds}_*.json")))
        if not files:
            continue
        runs = [json.load(open(f)) for f in files]
        ns = sorted(runs[0], key=int)
        methods = list(runs[0][ns[0]])
        curves[ds] = {m: [np.mean([r[n][m] for r in runs]) for n in ns] for m in methods}
        lines.append(f"\n**{ds}** (mean of {len(runs)} seeds; test accuracy %)\n")
        lines.append("| Method | " + " | ".join(f"n={n}" for n in ns) + " |")
        lines.append("|---|" + "---|" * len(ns))
        for m in methods:
            lines.append(f"| {m} | " + " | ".join(f"{v * 100:.1f}" for v in curves[ds][m]) + " |")
    if curves:
        fig, axes = plt.subplots(1, len(curves), figsize=(11, 4.2), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, (ds, cur) in zip(axes, curves.items()):
            ns = [int(n) for n in sorted(json.load(open(sorted(glob.glob(os.path.join(RAW, f"fewshot_{ds}_*.json")))[0])), key=int)]
            for (m, v), col in zip(cur.items(), C):
                ax.plot(ns, np.array(v) * 100, color=col, lw=2, marker="o", ms=5, label=m)
            ax.set_xscale("log")
            ax.set_xticks(ns, [str(n) for n in ns])
            ax.minorticks_off()
            ax.set_xlabel("labelled examples per class (log scale)")
            ax.set_title({"mnist": "MNIST", "fashion": "Fashion-MNIST"}[ds])
        axes[0].set_ylabel("test accuracy (%)")
        axes[-1].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig3_fewshot.png"), dpi=150)
    return lines


def lifelong():
    rows, extra = load("lifelong_")
    if not rows:
        return []
    lines = ["| Method | All 20 classes | Digits (learned first) | Clothing (learned last) | Stores raw data |", "|---|---|---|---|---|"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(1, 11)
    show = ["MLP (backprop) | ordered", "MLP + replay 500 | ordered", "SPARC-online / assoc | ordered", "SPARC-online / slda | ordered"]
    w = 0.2
    for i, (k, col) in enumerate(zip(show, C)):
        if k not in rows:
            continue
        M = np.array(rows[k][0]["matrix"]) * 100
        axes[0].bar(x + (i - 1.5) * w, M[-1], w - 0.02, color=col, label=k.replace(" | ordered", ""))
    for k, v in rows.items():
        M = np.array(v[0]["matrix"]) * 100
        lines.append(f"| {k.replace(' | ', ', ')} | {v[0]['final_acc'] * 100:.1f} | {M[-1, :5].mean():.1f} | {M[-1, 5:].mean():.1f} | "
                     f"{'500 images' if 'replay' in k else 'no'} |")
    axes[0].set_xticks(x, [f"{2 * i}/{2 * i + 1}" for i in range(10)], fontsize=8)
    axes[0].axvline(5.5, color=INK2, lw=1, ls="--")
    axes[0].text(3.0, 103, "digits (learned first)", ha="center", fontsize=8, color=INK2)
    axes[0].text(8.0, 103, "clothing (learned last)", ha="center", fontsize=8, color=INK2)
    axes[0].set_ylim(0, 110)
    axes[0].set_xlabel("class pair, in the order learned")
    axes[0].set_ylabel("accuracy after the whole stream (%)")
    axes[0].set_title("What is still known at the end of a 20-class life")
    axes[0].grid(axis="x", visible=False)
    for key, col, lab in (("_growth", C[0], "256-unit cap (full after 13k images)"), ("_growth (512 units)", C[1], "512-unit cap (full after 56k images)")):
        if extra.get(key):
            g = np.array(extra[key][0])
            axes[1].plot(g[:, 0], g[:, 1], color=col, lw=2, label=lab)
    axes[1].axvline(60000, color=INK2, lw=1, ls="--")
    axes[1].text(61500, 545, "clothing begins", fontsize=8, color=INK2, va="bottom")
    axes[1].set_xlabel("images experienced")
    axes[1].set_ylabel("feature units recruited")
    axes[1].set_ylim(0, 580)
    axes[1].set_title("Feature units recruited over the lifetime")
    axes[1].legend(fontsize=8, loc="lower right")
    from matplotlib.patches import Patch
    proxies = [Patch(color=col, label=k.replace(" | ordered", "")) for k, col in zip(show, C) if k in rows]
    fig.legend(handles=proxies, loc="lower left", ncol=4, fontsize=8.5, bbox_to_anchor=(0.04, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(OUT, "fig4_lifelong.png"), dpi=150)
    return lines


def main():
    md = ["# Results (auto-generated by report.py)\n"]
    summaries = {}
    for ds in ("mnist", "fashion"):
        lines, summ, ewc_lines, extra = split_table(ds)
        if len(lines) <= 2:
            continue
        summaries[ds] = summ
        md += [f"\n## Split-{ds.upper() if ds == 'mnist' else 'Fashion-MNIST'}: 5 tasks x 2 classes, one pass, no task labels (mean ± std over seeds)\n"] + lines
        md += ["\nEWC lambda sweep (seed 0, ordered):\n", "| Setting | Accuracy |", "|---|---|"] + ewc_lines
        secs = extra.get("_seconds_ordered")
        if secs:
            md.append(f"\nSPARC-online wall-clock for one pass (single CPU thread): {np.mean(secs):.0f} s")
        fig_forgetting(summ, ds)
    if summaries:
        fig_order_gap(summaries)
        fig_commutativity(summaries)
        json.dump(summaries, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
    md += ["\n## Sleep: generative replay from the commutative memory into a backprop learner (no stored images)\n"] + sleep_table()
    md += ["\n## Few-shot (shuffled order)\n"] + fewshot()
    md += ["\n## Lifelong: 10 digit classes then 10 clothing classes, one pass (seed 0)\n"] + lifelong()
    open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
