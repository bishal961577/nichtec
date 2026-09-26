"""Reproduce every number in AGI_BOTTLENECK.md.

    python3 run_experiments.py all          # everything (about 1 hour on a 4-core CPU)
    python3 run_experiments.py split mnist 0 # one experiment / dataset / seed

Each job writes one JSON file into results/raw/. `python3 report.py` aggregates them.
"""
import json
import os
import sys
import subprocess

import numpy as np

from sparc import baselines as B
from sparc import data, protocol
from sparc.model import PatchDictionary
from sparc.sleep import run_sleep

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "raw")
SPLIT = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]
SPARC_FEAT = {"wta": True, "mature": 200, "recruit": 0.6}  # fully online setting (chosen by the design search in results/ablations.md)
ASSOC = {"m": 40000, "k": 100, "vote": "norm"}
EWC_LAMBDAS = [1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
OTHER = {"mnist": "fashion", "fashion": "mnist"}


def save(name, obj):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, name + ".json"), "w") as f:
        json.dump(obj, f)
    print("wrote", name, flush=True)


def flat(x):
    return x.reshape(len(x), -1)


def developmental_dictionary(dataset, seed, n=20000):
    """'Critical period': learn patch features, without labels, from a DIFFERENT dataset, then freeze."""
    other = data.load(OTHER[dataset])[0]
    rng = np.random.default_rng(seed + 100)
    imgs = other[rng.permutation(len(other))[:n]]
    fd = PatchDictionary(seed=seed)
    for i in range(0, len(imgs), 100):
        fd.learn(imgs[i : i + 100])
    fd.frozen = True
    return fd


def run_sparc_dev(fd, imgs, y, imgs_t, yt, stream, tasks, n_classes, seed):
    """SPARC with a frozen developmental feature layer; only the commutative readouts learn."""
    reads = {
        "assoc": B.SparseAssociative(fd.dim, n_classes, seed=seed, dense=True, **ASSOC),
        "slda": B.SLDA(fd.dim, n_classes, shrinkage=1e-1),
        "ncm": B.NCM(fd.dim, n_classes),
        "delta": B.SparseDelta(fd.dim, n_classes, seed=seed, dense=True, m=40000, k=100),
    }
    ft = fd.encode(imgs_t)
    mats = {k: [] for k in reads}
    for idx in stream:
        for i in range(0, len(idx), 500):
            j = idx[i : i + 500]
            f = fd.encode(imgs[j])
            for r in reads.values():
                r.learn(f, y[j])
        if len(stream) > 1:
            for k, r in reads.items():
                mats[k].append(protocol.task_accuracy(r.predict(ft), yt, tasks))
    classes = np.concatenate([np.asarray(t) for t in tasks])
    out = {}
    for k, r in reads.items():
        pred = r.predict(ft)
        m = mats[k] if len(stream) > 1 else [protocol.task_accuracy(pred, yt, tasks)]
        out[k] = protocol.summarize(m, tasks, pred, yt, classes)
    return out


# ----------------------------------------------------------------------------------------
def job_split(dataset, seed, part):
    """Class-incremental split (5 tasks x 2 classes), single pass, no task labels at test time."""
    xtr, ytr, xte, yte = data.load(dataset)
    X, Xt = flat(xtr), flat(xte)
    rng = np.random.default_rng(seed)
    seq = data.class_incremental_stream(ytr, SPLIT, rng)
    iid = data.iid_stream(ytr, SPLIT, np.random.default_rng(seed + 1))
    tag = f"split_{dataset}_{seed}_{part}"
    res = {}
    if part == "mlp":
        res["MLP (backprop) | ordered"] = protocol.run_mlp(X, ytr, Xt, yte, seq, SPLIT, 10, seed=seed)
        res["MLP (backprop) | shuffled"] = protocol.run_mlp(X, ytr, Xt, yte, iid, SPLIT, 10, seed=seed)
        res["MLP (backprop) | shuffled, 5 epochs"] = protocol.run_mlp(X, ytr, Xt, yte, iid, SPLIT, 10, epochs=5, seed=seed)
    elif part == "replay":
        res["MLP + replay 500 | ordered"] = protocol.run_mlp(X, ytr, Xt, yte, seq, SPLIT, 10, replay=500, seed=seed)
        res["MLP + replay 5000 | ordered"] = protocol.run_mlp(X, ytr, Xt, yte, seq, SPLIT, 10, replay=5000, seed=seed)
    elif part.startswith("ewc"):
        lam = float(part[3:])
        res[f"MLP + EWC {lam:g} | ordered"] = protocol.run_mlp(X, ytr, Xt, yte, seq, SPLIT, 10, ewc_lam=lam, seed=seed)
    elif part == "pixels":
        for order, stream in (("ordered", seq), ("shuffled", iid)):
            res[f"NCM (pixels) | {order}"] = protocol.run_commutative(B.NCM(784, 10), X, ytr, Xt, yte, stream, SPLIT)
            res[f"SLDA (pixels) | {order}"] = protocol.run_commutative(B.SLDA(784, 10, 1e-1), X, ytr, Xt, yte, stream, SPLIT)
            fly = B.SparseAssociative(784, 10, m=20000, k=1000, fan_in=10, seed=seed)
            res[f"Fly (pixels) | {order}"] = protocol.run_commutative(fly, X, ytr, Xt, yte, stream, SPLIT)
    elif part in ("sparc_seq", "sparc_iid"):
        order = "ordered" if part == "sparc_seq" else "shuffled"
        out = protocol.run_sparc(xtr, ytr, xte, yte, seq if order == "ordered" else iid, SPLIT, 10,
                                 feat_kw=SPARC_FEAT, assoc_kw=ASSOC, seed=seed)
        for k in ("assoc", "slda", "ncm"):
            res[f"SPARC-online / {k} | {order}"] = out[k]
        res["_growth_" + order] = out["growth"]
        res["_seconds_" + order] = out["seconds"]
    elif part == "sparc_dev":
        fd = developmental_dictionary(dataset, seed)
        for order, stream in (("ordered", seq), ("shuffled", iid)):
            out = run_sparc_dev(fd, xtr, ytr, xte, yte, stream, SPLIT, 10, seed)
            for k in ("assoc", "slda", "ncm", "delta"):
                res[f"SPARC-dev / {k} | {order}"] = out[k]
    save(tag, res)


def job_lifelong(seed, part):
    """20 classes, 10 tasks: all ten digits, then all ten clothing classes. One pass."""
    xtr, ytr, xte, yte = data.load_combined()
    tasks = [(2 * i, 2 * i + 1) for i in range(10)]
    seq = data.class_incremental_stream(ytr, tasks, np.random.default_rng(seed))
    res = {}
    if part == "mlp":
        res["MLP (backprop) | ordered"] = protocol.run_mlp(flat(xtr), ytr, flat(xte), yte, seq, tasks, 20, seed=seed)
        iid = data.iid_stream(ytr, tasks, np.random.default_rng(seed + 1))
        res["MLP (backprop) | shuffled"] = protocol.run_mlp(flat(xtr), ytr, flat(xte), yte, iid, tasks, 20, seed=seed)
    elif part == "replay":
        res["MLP + replay 500 | ordered"] = protocol.run_mlp(flat(xtr), ytr, flat(xte), yte, seq, tasks, 20, replay=500, seed=seed)
    elif part in ("sparc", "sparc512"):
        kmax = 512 if part == "sparc512" else 256
        out = protocol.run_sparc(xtr, ytr, xte, yte, seq, tasks, 20, feat_kw={**SPARC_FEAT, "k_max": kmax},
                                 assoc_kw=ASSOC, seed=seed)
        tag = "" if kmax == 256 else " (512 units)"
        for k in ("assoc", "slda", "ncm"):
            res[f"SPARC-online{tag} / {k} | ordered"] = out[k]
        res["_growth" + tag] = out["growth"]
    save(f"lifelong_{seed}_{part}", res)


def job_sleep(dataset, seed):
    """Complementary learning systems: commutative fast memory -> generative replay -> backprop learner."""
    xtr, ytr, xte, yte = data.load(dataset)
    fd = developmental_dictionary(dataset, seed)
    F, Ft = fd.encode(xtr), fd.encode(xte)
    seq = data.class_incremental_stream(ytr, SPLIT, np.random.default_rng(seed))
    iid = data.iid_stream(ytr, SPLIT, np.random.default_rng(seed + 1))
    res = {
        "Backprop on dev features, no sleep | ordered": run_sleep(F, ytr, Ft, yte, seq, SPLIT, 10, seed=seed, replay=False),
        "Backprop on dev features + sleep | ordered": run_sleep(F, ytr, Ft, yte, seq, SPLIT, 10, seed=seed, replay=True),
        "Backprop on dev features, no sleep | shuffled": run_sleep(F, ytr, Ft, yte, iid, SPLIT, 10, seed=seed, replay=False),
    }
    save(f"sleep_{dataset}_{seed}", res)


def job_fewshot(dataset, seed):
    """Accuracy after n labelled examples per class (shuffled order)."""
    xtr, ytr, xte, yte = data.load(dataset)
    rng = np.random.default_rng(seed)
    fd = developmental_dictionary(dataset, seed)
    ft_te = fd.encode(xte)
    res = {}
    for n in (1, 2, 5, 10, 20, 50, 100, 500):
        idx = np.concatenate([rng.choice(np.flatnonzero(ytr == c), n, replace=False) for c in range(10)])
        idx = rng.permutation(idx)
        row = {}
        f = fd.encode(xtr[idx])
        for name, r in (("SPARC-dev / slda", B.SLDA(fd.dim, 10, 1e-1)), ("SPARC-dev / ncm", B.NCM(fd.dim, 10)),
                        ("SPARC-dev / assoc", B.SparseAssociative(fd.dim, 10, seed=seed, dense=True, **ASSOC))):
            r.learn(f, ytr[idx])
            row[name] = float((r.predict(ft_te) == yte).mean())
        ncm = B.NCM(784, 10)
        ncm.learn(flat(xtr[idx]), ytr[idx])
        row["NCM (pixels)"] = float((ncm.predict(flat(xte)) == yte).mean())
        for epochs, label in ((1, "MLP (backprop), 1 pass"), (50, "MLP (backprop), 50 epochs")):
            net = B.MLP(784, 10, lr=1e-3, seed=seed)
            for e in range(epochs):
                o = np.random.default_rng(seed * 1000 + e).permutation(len(idx))
                for i in range(0, len(o), 10):
                    j = idx[o[i : i + 10]]
                    net.step(flat(xtr[j]), ytr[j])
            row[label] = float((net.predict(flat(xte)) == yte).mean())
        res[str(n)] = row
        print(dataset, seed, n, row, flush=True)
    save(f"fewshot_{dataset}_{seed}", res)


def all_jobs():
    jobs = []
    for dataset in ("mnist", "fashion"):
        for seed in (0, 1, 2):
            for part in ("sparc_seq", "sparc_iid", "sparc_dev", "mlp", "replay", "pixels"):
                jobs.append(f"split {dataset} {seed} {part}")
            if seed == 0:
                jobs += [f"split {dataset} 0 ewc{lam:g}" for lam in EWC_LAMBDAS]
        for seed in (0, 1, 2):
            jobs.append(f"fewshot {dataset} {seed}")
            jobs.append(f"sleep {dataset} {seed}")
    for part in ("sparc", "sparc512", "mlp", "replay"):
        jobs.append(f"lifelong 0 {part}")
    slow = [j for j in jobs if 'sparc' in j or 'lifelong' in j]
    return slow + [j for j in jobs if j not in slow]


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "all":
        todo = [j for j in all_jobs() if not os.path.exists(os.path.join(RAW, "_".join(j.split()[0:1] + j.split()[1:]) + ".json"))]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        procs = int(os.environ.get("JOBS", "4"))
        subprocess.run(["xargs", "-P", str(procs), "-I{}", "sh", "-c", f"{sys.executable} {__file__} {{}}"],
                       input="\n".join(todo), text=True, env=env, check=False)
    elif args[0] == "split":
        job_split(args[1], int(args[2]), args[3])
    elif args[0] == "lifelong":
        job_lifelong(int(args[1]), args[2])
    elif args[0] == "sleep":
        job_sleep(args[1], int(args[2]))
    elif args[0] == "fewshot":
        job_fewshot(args[1], int(args[2]))
    else:
        raise SystemExit(__doc__)
