"""A language model trained in closed form from additive statistics. No gradient descent.

Pipeline (every learned quantity is a sum over training tokens, plus deterministic algebra):
  1. Word vectors from co-occurrence counts: positive PMI over neighbours at offsets -2,-1,+1,+2,
     then a truncated SVD.
  2. Context features for predicting token t: the vectors of the previous n tokens, a decayed
     "topic" average of earlier tokens, and a random nonlinear expansion relu(R h + b).
  3. Readout from second-order sufficient statistics G = sum(phi phi^T) and
     B = sum(phi onehot(y)^T):
     - ridge: W = (G + lam I)^-1 B, then p = softmax(W^T phi / tau)
     - lda:   class means + shared covariance (Gaussian discriminant), then p = softmax(scores / tau)
  Because G and B are sums, shards can be merged by adding them, and data removed by subtracting.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds


def count_embeddings(ids, V, d=256, offsets=(-2, -1, 1, 2), alpha=0.75, seed=0):
    rows, cols = [], []
    n = len(ids)
    for j, off in enumerate(offsets):
        lo, hi = max(0, -off), min(n, n - off)
        rows.append(ids[lo:hi])
        cols.append(ids[lo + off : hi + off] + j * V)
    r, c = np.concatenate(rows), np.concatenate(cols)
    C = sp.coo_matrix((np.ones(len(r), np.float64), (r, c)), shape=(V, V * len(offsets))).tocsr()
    C.sum_duplicates()
    total = C.sum()
    pw = np.asarray(C.sum(1)).ravel() / total
    pc = np.asarray(C.sum(0)).ravel() ** alpha
    pc /= pc.sum()
    C = C.tocoo()
    pmi = np.log(C.data / total / (pw[C.row] * pc[C.col] + 1e-30) + 1e-30)
    keep = pmi > 0
    P = sp.csr_matrix((pmi[keep], (C.row[keep], C.col[keep])), shape=C.shape)
    U, S, _ = svds(P, k=d, random_state=seed)
    E = U * np.sqrt(S)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-8
    return E.astype(np.float32)


class ContextFeatures:
    def __init__(self, E, n_prev=4, topic_decay=0.98, m=4096, seed=0, scale=1.0):
        self.E, self.n = E, n_prev
        self.decay = topic_decay
        d = E.shape[1]
        self.din = d * (n_prev + 1)
        rng = np.random.default_rng(seed)
        self.R = (rng.standard_normal((self.din, m)) * scale / np.sqrt(self.din)).astype(np.float32)
        self.b = (rng.standard_normal(m) * 0.5).astype(np.float32)
        self.dim = self.din + m + 1

    def raw(self, ids, start, stop, prev_topic=None):
        """Context vectors for predicting ids[start:stop] (uses ids before each position)."""
        E, n = self.E, self.n
        d = E.shape[1]
        T = stop - start
        h = np.zeros((T, self.din), np.float32)
        for j in range(1, n + 1):
            pos = np.arange(start, stop) - j
            ok = pos >= 0
            h[ok, (j - 1) * d : j * d] = E[ids[pos[ok]]]
        # exponentially decayed topic vector of everything before t (computed sequentially)
        topic = np.zeros(d, np.float32) if prev_topic is None else prev_topic
        tp = np.empty((T, d), np.float32)
        a = self.decay
        for i, t in enumerate(range(start, stop)):
            tp[i] = topic
            if t >= 0:
                topic = a * topic + (1 - a) * E[ids[t]]
        h[:, n * d :] = tp / (np.linalg.norm(tp, axis=1, keepdims=True) + 1e-8)
        return h, topic

    def expand(self, h):
        z = np.maximum(h @ self.R + self.b, 0)
        return np.concatenate([h, z, np.ones((len(h), 1), np.float32)], axis=1)

    def stream(self, ids, chunk=8192):
        topic = None
        for s in range(0, len(ids), chunk):
            e = min(len(ids), s + chunk)
            h, topic = self.raw(ids, s, e, topic)
            yield s, e, self.expand(h)


class ClosedFormLM:
    def __init__(self, feats, V):
        self.f, self.V = feats, V
        D = feats.dim
        self.G = np.zeros((D, D), np.float64)
        self.B = np.zeros((D, V), np.float64)
        self.n = np.zeros(V, np.float64)

    def add(self, ids, sign=1.0):
        """Accumulate sufficient statistics (sign=-1 removes data exactly)."""
        for s, e, Phi in self.f.stream(ids):
            y = ids[s:e]
            self.G += sign * (Phi.T.astype(np.float64) @ Phi)
            Y = sp.csr_matrix((np.ones(len(y)), (np.arange(len(y)), y)), shape=(len(y), self.V))
            self.B += sign * np.asarray((Y.T @ Phi).T, dtype=np.float64)
            self.n += sign * np.bincount(y, minlength=self.V)

    def solve(self, lam=1e-2, kind="ridge"):
        self.kind = kind
        D = self.G.shape[0]
        N = self.n.sum()
        if kind == "ridge":
            A = self.G / N + lam * np.eye(D)
            self.W = np.linalg.solve(A, self.B / N).astype(np.float32)
            self.c = np.zeros(self.V, np.float32)
        else:  # lda: Gaussian classes with shared covariance
            nz = np.maximum(self.n, 1)
            mu = (self.B / nz).T  # V x D
            S = (self.G - (mu.T * self.n) @ mu) / N
            S += lam * np.trace(S) / D * np.eye(D)
            P = np.linalg.solve(S, mu.T)  # D x V
            self.W = P.astype(np.float32)
            self.c = (-0.5 * np.einsum("vd,dv->v", mu, P) + np.log(nz / N)).astype(np.float32)
        return self

    def target_probs(self, ids, tau=1.0, floor=1e-3, kind=None):
        """Ridge outputs estimate E[onehot(y) | phi] = P(y | context) directly (a linear probability
        model): clip at zero, add a small unigram floor, renormalise. LDA outputs are log-scores
        and go through a tempered softmax."""
        kind = kind or self.kind
        prior = self.n / self.n.sum()
        out = np.empty(len(ids))
        for s, e, Phi in self.f.stream(ids):
            z = Phi @ self.W + self.c
            r = np.arange(e - s)
            if kind == "ridge":
                q = np.maximum(z, 0) + floor * prior
                out[s:e] = q[r, ids[s:e]] / q.sum(1)
            else:
                z = z / tau
                z -= z.max(1, keepdims=True)
                out[s:e] = np.exp(z[r, ids[s:e]] - np.log(np.exp(z).sum(1)))
        return out


def target_probs_at(lm, ids, positions, floor=0.1):
    """Ridge probabilities of ids[t] at selected positions (features still use the full stream)."""
    positions = np.asarray(positions)
    prior = lm.n / lm.n.sum()
    out = np.empty(len(positions))
    for s, e, Phi in lm.f.stream(ids):
        sel = (positions >= s) & (positions < e)
        if not sel.any():
            continue
        rows = positions[sel] - s
        z = Phi[rows] @ lm.W + lm.c
        q = np.maximum(z, 0) + floor * prior
        out[sel] = q[np.arange(len(rows)), ids[positions[sel]]] / q.sum(1)
    return out
