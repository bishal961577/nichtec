"""Interpolated modified Kneser-Ney n-gram model (Chen & Goodman 1998). Pure counting.

Everything the model knows is a table of counts, so training is addition: two count tables
built on different shards of data can be summed into exactly the model trained on all of it,
and a document can be removed exactly by subtracting its counts.
"""
from collections import Counter, defaultdict

import numpy as np


class KneserNey:
    def __init__(self, order=5, vocab_size=None):
        self.N = order
        self.V = vocab_size
        self.counts = [Counter() for _ in range(order + 1)]  # counts[k][ngram tuple of length k]

    # ---- training: pure addition ---------------------------------------------------------
    def add(self, ids, sign=1, start=0):
        """Count every k-gram of `ids` that ENDS at position >= start (earlier tokens are context).
        sign=-1 subtracts, which is how data is removed."""
        ids = [int(i) for i in ids]
        for k in range(1, self.N + 1):
            c = self.counts[k]
            for i in range(max(0, start - k + 1), len(ids) - k + 1):
                c[tuple(ids[i : i + k])] += sign
        self._stale = True
        return self

    def merge(self, other, sign=1):
        for k in range(1, self.N + 1):
            self.counts[k].update({g: sign * v for g, v in other.counts[k].items()})
        self._stale = True

    # ---- derived statistics (a deterministic function of the counts) ---------------------
    def finalize(self):
        N = self.N
        for k in range(1, N + 1):  # drop zeros left behind by subtraction
            self.counts[k] = Counter({g: v for g, v in self.counts[k].items() if v > 0})
        # modified counts: highest order uses raw counts, lower orders use continuation counts
        self.mod = [None] * (N + 1)
        self.mod[N] = self.counts[N]
        for k in range(1, N):
            cont = Counter()
            for g in self.counts[k + 1]:
                cont[g[1:]] += 1
            self.mod[k] = cont
        self.D = [None] * (N + 1)
        self.ctx = [None] * (N + 1)  # ctx[k][h] = (total, n1, n2, n3plus)
        for k in range(1, N + 1):
            cc = Counter(self.mod[k].values())
            n1, n2, n3, n4 = (max(cc.get(i, 0), 1) for i in (1, 2, 3, 4))
            Y = n1 / (n1 + 2 * n2)
            self.D[k] = (0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3)
            stats = defaultdict(lambda: [0, 0, 0, 0])
            for g, v in self.mod[k].items():
                s = stats[g[:-1]]
                s[0] += v
                s[min(v, 3)] += 1
            self.ctx[k] = dict(stats)
        if self.V is None:
            self.V = len(self.counts[1])
        self._stale = False
        return self

    def prob(self, h, w):
        """P(w | h) for a context tuple h (any length; the last N-1 tokens are used)."""
        h = tuple(h[-(self.N - 1):]) if self.N > 1 else ()
        p = 1.0 / self.V
        for k in range(1, len(h) + 2):  # from unigram up to the full order
            hk = h[len(h) - (k - 1):] if k > 1 else ()
            st = self.ctx[k].get(hk)
            if st is None:
                continue  # unseen context: keep lower-order estimate
            total, a1, a2, a3 = st
            c = self.mod[k].get(hk + (w,), 0)
            D = self.D[k]
            gamma = (D[1] * a1 + D[2] * a2 + D[3] * a3) / total
            p = max(c - D[min(c, 3)], 0.0) / total + gamma * p
        return p

    def stream_probs(self, ids):
        """P(ids[t] | ids[:t]) for every position of a continuous stream."""
        ids = [int(i) for i in ids]
        return np.array([self.prob(tuple(ids[max(0, t - self.N + 1) : t]), ids[t]) for t in range(len(ids))])


def perplexity(p):
    return float(np.exp(-np.mean(np.log(p))))
