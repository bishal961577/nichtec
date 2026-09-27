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
