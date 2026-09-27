"""Cache model and linear mixing (mixture weights fit by EM on the validation stream)."""
import numpy as np


def cache_probs(ids, V, window=500, history=None):
    """Unigram cache: P(w) = share of w among the previous `window` tokens (document memory)."""
    ids = np.asarray(ids)
    full = ids if history is None else np.concatenate([history, ids])
    off = 0 if history is None else len(history)
    counts = np.zeros(V)
    out = np.zeros(len(ids))
    for i in range(len(full)):
        if i >= off:
            n = min(i, window)
            out[i - off] = counts[full[i]] / n if n else 0.0
        counts[full[i]] += 1
        if i >= window:
            counts[full[i - window]] -= 1
    return out


def em_weights(P, iters=200):
    """P: (T, K) per-token probabilities of the target under K models. Returns mixture weights."""
    K = P.shape[1]
    lam = np.full(K, 1.0 / K)
    for _ in range(iters):
        r = P * lam
        r /= r.sum(1, keepdims=True)
        lam = r.mean(0)
    return lam


def mix_ppl(P, lam):
    return float(np.exp(-np.mean(np.log(P @ lam))))
