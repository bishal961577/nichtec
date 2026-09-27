"""Soft n-gram: a nonparametric next-word model whose 'training' is storing (context, next word).

The key for position t is the concatenation of count-derived vectors of the previous n words,
each scaled by a per-distance weight, plus a decayed topic vector. Prediction is a kernel vote:
    P(w | c) is proportional to sum_i exp(<key(c), key_i> / tau) * [next_i == w]
Exact n-gram matching is the limit where only identical words count as similar. Here
'monday' partly matches 'tuesday', because their count vectors are close.
Adding data appends rows; removing data deletes rows. Nothing is ever re-optimised.
"""
import numpy as np
import scipy.sparse as sp


def keys(ids, E, n_prev=4, weights=None, topic_w=0.0, decay=0.98, start=0, history=None):
    """Key vectors for predicting ids[t] for every t (uses only tokens before t)."""
    ids = np.asarray(ids)
    full = ids if history is None else np.concatenate([history, ids])
    off = 0 if history is None else len(history)
    d = E.shape[1]
    weights = np.ones(n_prev) if weights is None else np.asarray(weights)
    T = len(ids)
    K = np.zeros((T, d * n_prev + (d if topic_w else 0)), np.float32)
    pos = np.arange(T) + off
    for j in range(1, n_prev + 1):
        p = pos - j
        ok = p >= 0
        K[ok, (j - 1) * d : j * d] = E[full[p[ok]]] * np.sqrt(weights[j - 1])
    if topic_w:
        topic = np.zeros(d, np.float32)
        tp = np.empty((len(full), d), np.float32)
        for i in range(len(full)):
            tp[i] = topic
            topic = decay * topic + (1 - decay) * E[full[i]]
        tp = tp[off:]
        tp /= np.linalg.norm(tp, axis=1, keepdims=True) + 1e-8
        K[:, d * n_prev :] = tp * np.sqrt(topic_w)
    return K


class SoftGram:
    def __init__(self, train_keys, train_next, V):
        self.K = np.ascontiguousarray(train_keys, dtype=np.float32)
        self.Yt = sp.csr_matrix((np.ones(len(train_next), np.float32), (train_next, np.arange(len(train_next)))),
                                shape=(V, len(train_next)))
        self.V = V

    def target_probs(self, qkeys, targets, tau=0.1, chunk=256, exclude_self=False, floor=1e-6):
        out = np.empty(len(targets))
        for s in range(0, len(targets), chunk):
            e = min(len(targets), s + chunk)
            S = qkeys[s:e] @ self.K.T
            if exclude_self:
                S[np.arange(e - s), np.arange(s, e)] = -np.inf
            S -= S.max(1, keepdims=True)
            np.exp(S / tau, out=S)
            P = np.asarray((self.Yt @ S.T).T)  # (b, V) kernel-weighted next-word counts
            P = P / P.sum(1, keepdims=True)
            out[s:e] = P[np.arange(e - s), targets[s:e]] + floor
        return out


class SoftGramStream:
    """The same kernel vote as SoftGram, computed without materialising keys.

    Because a key is a concatenation of scaled word vectors, the similarity between a query and
    stored position i decomposes into per-offset word-to-word similarities:
        <key(q), key_i> = sum_j w_j * <E[q_{t-j}], E[x_{i-j}]>
    so it can be computed by gathering rows of (E_q @ E^T) along the training stream. Memory is
    O(stream length) instead of O(stream length x key dimension)."""

    def __init__(self, train_ids, E, weights, V):
        self.x = np.asarray(train_ids, np.int64)
        self.E = np.vstack([E, np.zeros((1, E.shape[1]), E.dtype)]).astype(np.float32)  # last row = padding
        self.pad = len(E)
        self.w = np.asarray(weights, np.float32)
        n = len(self.x)
        self.prev = []
        for j in range(1, len(self.w) + 1):
            p = np.full(n, self.pad, np.int64)
            p[j:] = self.x[:-j]
            self.prev.append(p.astype(np.int32))
        self.Yt = sp.csr_matrix((np.ones(n, np.float32), (self.x, np.arange(n))), shape=(V, n))

    def target_probs(self, ids, positions, tau=0.1, chunk=32, floor=1e-6):
        ids = np.asarray(ids, np.int64)
        positions = np.asarray(positions)
        out = np.empty(len(positions))
        for s in range(0, len(positions), chunk):
            pos = positions[s : s + chunk]
            S = None
            for j, prev in enumerate(self.prev, start=1):
                q = np.where(pos - j >= 0, ids[np.maximum(pos - j, 0)], self.pad)
                simtab = self.E[q] @ self.E.T  # (b, V+1) word-to-word similarities
                part = simtab[:, prev]
                S = self.w[j - 1] * part if S is None else S + self.w[j - 1] * part
            S -= S.max(1, keepdims=True)
            np.exp(S / tau, out=S)
            P = np.asarray((self.Yt @ S.T).T)
            out[s : s + len(pos)] = P[np.arange(len(pos)), ids[pos]] / P.sum(1) + floor
        return out
