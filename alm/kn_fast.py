"""Vectorised interpolated modified Kneser-Ney (same model as kn.py, NumPy arrays instead of
dicts), for corpora of tens of millions of tokens. Pure counting.

k-grams up to order 4 are packed exactly into int64 (14 bits per token, vocabulary <= 16384);
5-grams are hashed to 64 bits (splitmix64), where collisions are negligible at this scale."""
import numpy as np

BITS = 14
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _mix(z):
    z = z.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return z.view(np.int64)


def gram_keys(x, N):
    """G[k][i] = key of the k-gram ending at position i (valid for i >= k-1; -1 elsewhere)."""
    x = np.asarray(x, np.int64)
    assert x.max() < (1 << BITS)
    n = len(x)
    G = [None, x.copy()]
    for k in range(2, N + 1):
        g = np.full(n, -1, np.int64)
        prev = G[k - 1]
        if k <= 4:
            g[k - 1 :] = (prev[k - 2 : -1] << BITS) | x[k - 1 :]
        else:
            with np.errstate(over="ignore"):
                g[k - 1 :] = _mix(prev[k - 2 : -1].view(np.uint64) * np.uint64(1000003) ^ x[k - 1 :].astype(np.uint64))
            g[k - 1 :] = np.where(g[k - 1 :] == -1, -2, g[k - 1 :])
        G.append(g)
    return G


def ctx_keys(G, k):
    """Key of the (k-1)-token context before each position (0 for the empty unigram context)."""
    n = len(G[1])
    if k == 1:
        return np.zeros(n, np.int64)
    c = np.full(n, -1, np.int64)
    c[1:] = G[k - 1][:-1]
    return c


class FastKN:
    def __init__(self, order=5, V=None):
        self.N, self.V = order, V

    def fit(self, x):
        N = self.N
        G = gram_keys(x, N)
        self.tabs = {}
        for k in range(N, 0, -1):
            pos = np.arange(k - 1, len(x))
            if k == N:
                keys, first, cnt = np.unique(G[k][pos], return_index=True, return_counts=True)
                rep = pos[first]
            else:
                # continuation counts: number of distinct (k+1)-grams whose last k tokens are this k-gram
                up_keys, up_first = np.unique(G[k + 1][np.arange(k, len(x))], return_index=True)
                up_pos = np.arange(k, len(x))[up_first]
                suf = G[k][up_pos]
                keys, first, cnt = np.unique(suf, return_index=True, return_counts=True)
                rep = up_pos[first]
            ctx = ctx_keys(G, k)[rep]
            cc = np.bincount(np.minimum(cnt, 5))
            n1, n2, n3, n4 = (max(int(cc[i]) if i < len(cc) else 0, 1) for i in (1, 2, 3, 4))
            Y = n1 / (n1 + 2 * n2)
            D = np.array([0.0, 1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3])
            ukeys, inv = np.unique(ctx, return_inverse=True)
            total = np.bincount(inv, weights=cnt)
            nb = [np.bincount(inv, weights=(np.minimum(cnt, 3) == j)) for j in (1, 2, 3)]
            gamma = (D[1] * nb[0] + D[2] * nb[1] + D[3] * nb[2]) / total
            self.tabs[k] = dict(keys=keys, cnt=cnt, ctx_keys=ukeys, total=total, gamma=gamma, D=D)
        if self.V is None:
            self.V = int(np.max(x)) + 1
        return self

    @staticmethod
    def _lookup(sorted_keys, q):
        i = np.searchsorted(sorted_keys, q)
        i = np.minimum(i, len(sorted_keys) - 1)
        hit = sorted_keys[i] == q
        return i, hit

    def probs(self, x, positions=None):
        """P(x[t] | x[:t]) for the given positions (default: all), treating x as one stream."""
        x = np.asarray(x, np.int64)
        pos = np.arange(len(x)) if positions is None else np.asarray(positions)
        G = gram_keys(x, self.N)
        p = np.full(len(pos), 1.0 / self.V)
        for k in range(1, self.N + 1):
            T = self.tabs[k]
            ok = pos >= k - 1
            h = ctx_keys(G, k)[pos]
            ci, chit = self._lookup(T["ctx_keys"], h)
            chit &= ok
            gi, ghit = self._lookup(T["keys"], G[k][pos])
            c = np.where(ghit & ok, T["cnt"][gi], 0).astype(np.float64)
            tot = T["total"][ci]
            disc = T["D"][np.minimum(c, 3).astype(int)]
            new = np.maximum(c - disc, 0) / tot + T["gamma"][ci] * p
            p = np.where(chit, new, p)
        return p
