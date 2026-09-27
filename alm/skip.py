"""Skip-gram count models: P(w_t | w_{t-k}) for a fixed distance k, ignoring the words between.
Absolute discounting, interpolated with the unigram distribution. Pure counting."""
from collections import Counter, defaultdict

import numpy as np


class SkipGram:
    def __init__(self, k, V, D=0.75):
        self.k, self.V, self.D = k, V, D
        self.pair = Counter()
        self.uni = np.zeros(V)

    def add(self, ids, sign=1):
        ids = np.asarray(ids)
        for a, b in zip(ids[: -self.k].tolist(), ids[self.k :].tolist()):
            self.pair[(a, b)] += sign
        self.uni += sign * np.bincount(ids, minlength=self.V)
        return self

    def finalize(self):
        tot = defaultdict(lambda: [0, 0])
        for (a, _), c in self.pair.items():
            if c > 0:
                tot[a][0] += c
                tot[a][1] += 1
        self.tot = dict(tot)
        self.p_uni = (self.uni + 1) / (self.uni.sum() + self.V)
        return self

    def stream_probs(self, ids):
        ids = np.asarray(ids).tolist()
        out = np.empty(len(ids))
        for t, w in enumerate(ids):
            pu = self.p_uni[w]
            if t < self.k or ids[t - self.k] not in self.tot:
                out[t] = pu
                continue
            a = ids[t - self.k]
            n, types = self.tot[a]
            c = self.pair.get((a, w), 0)
            out[t] = max(c - self.D, 0) / n + self.D * types / n * pu
        return out
