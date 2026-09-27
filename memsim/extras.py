"""Extra checks for the exact-joint-memory write rule (small, fast configurations).

1. deletion   - remove 500 facts' constraints and re-solve; compare with a memory that never saw them.
2. growth     - overfill a small memory, watch the nightly residual certificate flag it, add a second
                slot table, re-solve, and check that every stored fact recovers.
3. wordings   - with unstable keys (low slot overlap between wordings), does writing each fact under
                more wordings recover recall for unseen wordings?
"""
import json

import numpy as np
import scipy.sparse as sp

M, DV, K, KSUB, TAU = 64, 64, 32, 32, 0.05


class PKM:
    """One or more product-key tables; a lookup returns top-K slots from each table."""

    def __init__(self, nsub, seed):
        r = np.random.default_rng(seed)
        self.nsub = nsub
        self.tables = []
        self.add_table(r)
        self.r = r

    def add_table(self, r=None):
        r = r or self.r
        k1 = r.standard_normal((self.nsub, M // 2)).astype(np.float32)
        k2 = r.standard_normal((self.nsub, M // 2)).astype(np.float32)
        self.tables.append((k1 / np.linalg.norm(k1, axis=1, keepdims=True), k2 / np.linalg.norm(k2, axis=1, keepdims=True)))

    @property
    def N(self):
        return len(self.tables) * self.nsub ** 2

    def lookup(self, Q):
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        idx_all, sc_all = [], []
        for t, (K1, K2) in enumerate(self.tables):
            s1, s2 = Q[:, : M // 2] @ K1.T, Q[:, M // 2 :] @ K2.T
            ks = min(KSUB, self.nsub)
            i1 = np.argpartition(-s1, ks - 1, axis=1)[:, :ks]
            i2 = np.argpartition(-s2, ks - 1, axis=1)[:, :ks]
            c = (np.take_along_axis(s1, i1, 1)[:, :, None] + np.take_along_axis(s2, i2, 1)[:, None, :]).reshape(len(Q), -1)
            top = np.argpartition(-c, K - 1, axis=1)[:, :K]
            ar = np.arange(len(Q))[:, None]
            idx_all.append(t * self.nsub ** 2 + i1[ar, top // ks] * self.nsub + i2[ar, top % ks])
            sc_all.append(np.take_along_axis(c, top, 1))
        idx, sc = np.concatenate(idx_all, 1), np.concatenate(sc_all, 1)
        w = np.exp((sc - sc.max(1, keepdims=True)) / TAU)
        return idx, (w / w.sum(1, keepdims=True)).astype(np.float32)


def facts(n, p, sigma, seed):
    r = np.random.default_rng(seed)
    base = r.standard_normal((n, M)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    Q = base[:, None, :] + sigma * r.standard_normal((n, p, M)).astype(np.float32) / np.sqrt(M)
    R = r.standard_normal((n, DV)).astype(np.float32)
    return Q, R / np.linalg.norm(R, axis=1, keepdims=True)


def design(idx, w, N):
    n, k = idx.shape
    return sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, N))


def solve(W, B, lam=1e-3, X0=None, iters=500, tol=1e-6):
    A = lambda X: W.T @ (W @ X) + lam * X
    rhs = W.T @ B
    X = np.zeros((W.shape[1], B.shape[1])) if X0 is None else X0.astype(np.float64).copy()
    Rr = rhs - A(X); P = Rr.copy(); rs = (Rr * Rr).sum(0); b = np.sqrt((rhs * rhs).sum(0)) + 1e-12
    for it in range(iters):
        AP = A(P); al = rs / ((P * AP).sum(0) + 1e-30); X += P * al; Rr -= AP * al
        rn = (Rr * Rr).sum(0)
        if np.all(np.sqrt(rn) / b < tol):
            break
        P = Rr + P * (rn / (rs + 1e-30)); rs = rn
    return X, it + 1


def out_cos(W, X, R_rows):
    o = W @ X
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    return (o * R_rows).sum(1)


def decode_acc(W, X, R_rows, R_all, true_ids):
    o = W @ X
    o /= np.linalg.norm(o, axis=1, keepdims=True) + 1e-12
    return float(((o @ R_all.T).argmax(1) == true_ids).mean())


def deletion():
    pk = PKM(256, seed=1)
    n, p = 5000, 4
    Q, R = facts(n, p, 0.3, seed=2)
    idx, w = pk.lookup(Q.reshape(-1, M))
    W = design(idx, w, pk.N)
    B = np.repeat(R, p, 0)
    X_full, _ = solve(W, B)
    keep = np.arange(500 * p, n * p)  # delete the first 500 facts
    X_del, it_del = solve(W[keep], B[keep], X0=X_full)          # warm-started re-solve after deletion
    X_ref, it_ref = solve(W[keep], B[keep])                      # memory that never saw the 500 facts
    Wd = W[: 500 * p]
    ids = np.repeat(np.arange(500), p)
    return {
        "max_abs_value_difference_deleted_vs_never_learned": float(np.abs(X_del - X_ref).max()),
        "max_abs_value": float(np.abs(X_ref).max()),
        "deleted_facts_recall_before": decode_acc(Wd, X_full, B[: 500 * p], R, ids),
        "deleted_facts_recall_after": decode_acc(Wd, X_del, B[: 500 * p], R, ids),
        "never_learned_recall_same_queries": decode_acc(Wd, X_ref, B[: 500 * p], R, ids),
        "kept_facts_recall_after": decode_acc(W[keep], X_del, B[keep], R, np.repeat(np.arange(500, n), p)),
        "resolve_iterations_warm_vs_cold": [it_del, it_ref],
    }


def growth():
    pk = PKM(64, seed=3)  # 64^2 = 4,096 slots per table
    p = 4
    Q, R = facts(4000, p, 0.3, seed=4)
    log = []
    X = None
    for n in (500, 1000, 2000, 3000, 4000):
        idx, w = pk.lookup(Q[:n].reshape(-1, M))
        W = design(idx, w, pk.N)
        B = np.repeat(R[:n], p, 0)
        X, it = solve(W, B, X0=X if X is not None and X.shape[0] == pk.N else None)
        cert = 1 - out_cos(W, X, B)  # nightly certificate: residual per stored constraint
        flagged = int((cert.reshape(n, p).max(1) > 0.05).sum())
        row = {"facts": n, "tables": len(pk.tables), "slots": pk.N, "constraints_per_slot": round(n * p / pk.N, 2),
               "facts_flagged_by_certificate": flagged, "mean_cos_written": round(float(1 - cert.mean()), 4)}
        if flagged:
            pk.add_table()  # grow: add a second slot table, every lookup now also reads from it
            idx, w = pk.lookup(Q[:n].reshape(-1, M))
            W = design(idx, w, pk.N)
            X, it = solve(W, B)
            cert = 1 - out_cos(W, X, B)
            row.update({"after_growth_tables": len(pk.tables), "after_growth_flagged": int((cert.reshape(n, p).max(1) > 0.05).sum()),
                        "after_growth_mean_cos_written": round(float(1 - cert.mean()), 4)})
        log.append(row)
        print(json.dumps(row), flush=True)
    return log


def wordings():
    pk = PKM(256, seed=5)
    out = []
    for p in (4, 8, 16):
        Q, R = facts(4000, p + 2, 0.6, seed=6)
        idx, w = pk.lookup(Q[:, :p].reshape(-1, M))
        W = design(idx, w, pk.N)
        X, _ = solve(W, np.repeat(R, p, 0))
        idh, wh = pk.lookup(Q[:, p:].reshape(-1, M))
        Wh = design(idh, wh, pk.N)
        acc = decode_acc(Wh, X, np.repeat(R, 2, 0), R, np.repeat(np.arange(4000), 2))
        out.append({"wordings_written_per_fact": p, "unseen_wording_recall": round(acc, 3)})
        print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    res = {"deletion": deletion()}
    print(json.dumps(res["deletion"], indent=1), flush=True)
    res["growth"] = growth()
    res["wordings"] = wordings()
    json.dump(res, open("/home/user/nichtec/memsim/extras_result.json", "w"), indent=1)
