"""Shared training/evaluation protocol: single pass over a stream, evaluate after every task."""
import time

import numpy as np

from . import baselines as B
from .model import PatchDictionary


def task_accuracy(pred, y, tasks):
    return [float((pred[np.isin(y, t)] == y[np.isin(y, t)]).mean()) for t in tasks]


def summarize(matrix, tasks, final_pred, y_test, seen_classes):
    """matrix[i][j] = accuracy on task j's test classes after training through task i."""
    mask = np.isin(y_test, seen_classes)
    final = float((final_pred[mask] == y_test[mask]).mean())
    M = np.array(matrix)
    T = len(M)
    forgetting = float(np.mean([M[j:, j].max() - M[-1, j] for j in range(T - 1)])) if T > 1 else 0.0
    return {"final_acc": final, "forgetting": forgetting, "matrix": M.tolist()}


# ----------------------------------------------------------------------------------------
def run_mlp(X, y, Xt, yt, stream, tasks, n_classes, *, lr=3e-4, bs=10, epochs=1, ewc_lam=None,
            replay=None, seed=0):
    t0 = time.time()
    net = B.MLP(X.shape[1], n_classes, lr=lr, seed=seed)
    ewc = B.EWC(net, ewc_lam, seed=seed) if ewc_lam else None
    buf = B.Reservoir(replay, X.shape[1], seed=seed) if replay else None
    matrix = []
    for idx in stream:
        for _ in range(epochs):
            order = idx if epochs == 1 else np.random.default_rng(seed + _).permutation(idx)
            for i in range(0, len(order), bs):
                j = order[i : i + bs]
                xb, yb = X[j], y[j]
                if buf is not None and buf.n > 0:
                    rx, ry = buf.sample(bs)
                    net.step(np.concatenate([xb, rx]), np.concatenate([yb, ry]))
                else:
                    net.step(xb, yb)
                if buf is not None:
                    buf.add(xb, yb)
        if ewc is not None:
            ewc.end_task(X[idx])
        if len(stream) > 1:
            matrix.append(task_accuracy(net.predict(Xt), yt, tasks))
    pred = net.predict(Xt)
    if len(stream) == 1:
        matrix = [task_accuracy(pred, yt, tasks)]
    out = summarize(matrix, tasks, pred, yt, np.concatenate([np.asarray(t) for t in tasks]))
    out["seconds"] = time.time() - t0
    out["params_updated_per_example"] = int(sum(p.size for p in net.params()))
    return out


def run_commutative(model, X, y, Xt, yt, stream, tasks, bs=100):
    """NCM / SLDA / Fly on raw pixels."""
    t0 = time.time()
    matrix = []
    for idx in stream:
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            model.learn(X[j], y[j])
        if len(stream) > 1:
            matrix.append(task_accuracy(model.predict(Xt), yt, tasks))
    pred = model.predict(Xt)
    if len(stream) == 1:
        matrix = [task_accuracy(pred, yt, tasks)]
    out = summarize(matrix, tasks, pred, yt, np.concatenate([np.asarray(t) for t in tasks]))
    out["seconds"] = time.time() - t0
    return out


def run_sparc(imgs, y, imgs_t, yt, stream, tasks, n_classes, *, feat_kw=None, assoc_kw=None, seed=0,
              bs=100, readouts=("assoc", "slda", "ncm")):
    """One SPARC feature layer feeding several commutative readouts at once (they never
    influence the feature layer, so sharing it is exactly equivalent to separate runs)."""
    t0 = time.time()
    fd = PatchDictionary(seed=seed, **(feat_kw or {}))
    reads = {}
    if "assoc" in readouts:
        reads["assoc"] = B.SparseAssociative(fd.dim, n_classes, seed=seed, dense=True, **(assoc_kw or {}))
    if "slda" in readouts:
        reads["slda"] = B.SLDA(fd.dim, n_classes, shrinkage=1e-1)
    if "ncm" in readouts:
        reads["ncm"] = B.NCM(fd.dim, n_classes)
    matrices = {k: [] for k in reads}
    growth = []
    for idx in stream:
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            fd.learn(imgs[j])
            f = fd.encode(imgs[j])
            for r in reads.values():
                r.learn(f, y[j])
            growth.append((fd.seen, fd.k))
        if len(stream) > 1:
            ft = fd.encode(imgs_t)
            for k, r in reads.items():
                matrices[k].append(task_accuracy(r.predict(ft), yt, tasks))
    ft = fd.encode(imgs_t)
    classes = np.concatenate([np.asarray(t) for t in tasks])
    out = {}
    for k, r in reads.items():
        pred = r.predict(ft)
        m = matrices[k] if len(stream) > 1 else [task_accuracy(pred, yt, tasks)]
        out[k] = summarize(m, tasks, pred, yt, classes)
    out["units"] = fd.k
    out["growth"] = growth[:: max(1, len(growth) // 200)]
    out["seconds"] = time.time() - t0
    return out
