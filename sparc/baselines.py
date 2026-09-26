"""Baselines, all in NumPy so every number in the report is reproducible on a laptop CPU.

Backprop family (non-commutative learners):
  MLP            - 2-hidden-layer ReLU net, softmax over all classes, Adam, trained online.
  MLP + EWC      - elastic weight consolidation (Kirkpatrick et al. 2017), online Fisher.
  MLP + Replay   - experience replay from a reservoir buffer of raw training images.

Sufficient-statistic family (commutative learners):
  NCM            - nearest class mean.
  SLDA           - streaming linear discriminant analysis (Hayes & Kanan 2020).
  Fly            - random sparse expansion + k-winners-take-all + associative readout
                   (Dasgupta et al. 2017; Shen, Dasgupta & Navlakha 2023).
"""
import numpy as np


# ----------------------------------------------------------------------------------------
# Backprop MLP
# ----------------------------------------------------------------------------------------
class MLP:
    def __init__(self, d_in, n_classes, hidden=(400, 400), lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        sizes = (d_in,) + tuple(hidden) + (n_classes,)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]
        self.lr = lr
        self.t = 0
        self.m = [np.zeros_like(p) for p in self.params()]
        self.v = [np.zeros_like(p) for p in self.params()]
        self.penalty = None  # optional callable(params) -> list of extra gradients

    def params(self):
        return self.W + self.b

    def forward(self, x):
        acts = [x]
        h = x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < len(self.W) - 1:
                h = np.maximum(h, 0)
            acts.append(h)
        return acts

    def grads(self, x, y):
        acts = self.forward(x)
        logits = acts[-1]
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        d = p
        d[np.arange(len(y)), y] -= 1
        d /= len(y)
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ d
            gb[i] = d.sum(0)
            if i > 0:
                d = (d @ self.W[i].T) * (acts[i] > 0)
        return gW + gb

    def step(self, x, y):
        g = self.grads(x, y)
        if self.penalty is not None:
            g = [a + b for a, b in zip(g, self.penalty(self.params()))]
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for p, gi, m, v in zip(self.params(), g, self.m, self.v):
            m *= b1
            m += (1 - b1) * gi
            v *= b2
            v += (1 - b2) * gi * gi
            mh = m / (1 - b1 ** self.t)
            vh = v / (1 - b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + eps)

    def predict(self, x, batch=2000):
        return np.concatenate([self.forward(x[i : i + batch])[-1].argmax(1) for i in range(0, len(x), batch)])


class EWC:
    """Online EWC: after each task, accumulate the diagonal Fisher and anchor the weights."""

    def __init__(self, net, lam, n_fisher=500, seed=0):
        self.net, self.lam, self.n_fisher = net, lam, n_fisher
        self.F = None
        self.anchor = None
        self.rng = np.random.default_rng(seed)
        net.penalty = self._penalty

    def _penalty(self, params):
        if self.F is None:
            return [np.zeros_like(p) for p in params]
        return [self.lam * f * (p - a) for f, p, a in zip(self.F, params, self.anchor)]

    def end_task(self, x_task):
        idx = self.rng.choice(len(x_task), min(self.n_fisher, len(x_task)), replace=False)
        F = [np.zeros_like(p) for p in self.net.params()]
        for i in idx:
            x = x_task[i : i + 1]
            logits = self.net.forward(x)[-1][0]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            y = self.rng.choice(len(p), p=p)  # true Fisher: sample from the model
            for f, g in zip(F, self.net.grads(x, np.array([y]))):
                f += g * g
        F = [f / len(idx) for f in F]
        self.F = F if self.F is None else [a + b for a, b in zip(self.F, F)]
        self.anchor = [p.copy() for p in self.net.params()]


class Reservoir:
    def __init__(self, size, d, seed=0):
        self.x = np.zeros((size, d), np.float32)
        self.y = np.zeros(size, np.int64)
        self.n = 0
        self.size = size
        self.rng = np.random.default_rng(seed)

    def add(self, x, y):
        for xi, yi in zip(x, y):
            if self.n < self.size:
                j = self.n
            else:
                j = self.rng.integers(0, self.n + 1)
            if j < self.size:
                self.x[j], self.y[j] = xi, yi
            self.n += 1

    def sample(self, k):
        m = min(self.n, self.size)
        idx = self.rng.integers(0, m, k)
        return self.x[idx], self.y[idx]


# ----------------------------------------------------------------------------------------
# Commutative (sufficient-statistic) learners
# ----------------------------------------------------------------------------------------
class NCM:
    def __init__(self, d, n_classes):
        self.S = np.zeros((n_classes, d), np.float64)
        self.N = np.zeros(n_classes)

    def learn(self, x, y):
        np.add.at(self.S, y, x)
        np.add.at(self.N, y, 1)

    def predict(self, x):
        seen = self.N > 0
        mu = self.S[seen] / self.N[seen, None]
        d = (x**2).sum(1)[:, None] - 2 * x @ mu.T + (mu**2).sum(1)[None]
        return np.flatnonzero(seen)[d.argmin(1)]


class SLDA:
    """Streaming LDA: class means + one shared covariance, both exact running sums."""

    def __init__(self, d, n_classes, shrinkage=1e-2):
        self.S = np.zeros((n_classes, d))
        self.N = np.zeros(n_classes)
        self.XX = np.zeros((d, d))
        self.shrink = shrinkage

    def learn(self, x, y):
        x = x.astype(np.float64)
        np.add.at(self.S, y, x)
        np.add.at(self.N, y, 1)
        self.XX += x.T @ x

    def predict(self, x):
        seen = self.N > 0
        mu = self.S[seen] / self.N[seen, None]
        n = self.N.sum()
        # within-class scatter = sum xx^T - sum_c N_c mu_c mu_c^T
        Sw = (self.XX - (mu.T * self.N[seen]) @ mu) / n
        Sw = (1 - self.shrink) * Sw + self.shrink * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
        P = np.linalg.inv(Sw)
        Wt = mu @ P
        b = -0.5 * (Wt * mu).sum(1)
        return np.flatnonzero(seen)[(x @ Wt.T + b).argmax(1)]


class SparseAssociative:
    """k-WTA sparse code -> class-association counts. Shared by the Fly baseline and SPARC.

    Learning is a pure sum over examples (A += h y^T), so it is exactly order-invariant."""

    def __init__(self, d, n_classes, m=20000, k=None, fan_in=None, seed=0, dense=False, vote="parzen"):
        rng = np.random.default_rng(seed)
        self.vote = vote  # "parzen": sum of counts / class size; "norm": each active unit votes p(class|unit)
        self.k = k or m // 20
        if dense:
            self.P = rng.standard_normal((d, m)).astype(np.float32)
        else:
            fan_in = fan_in or max(6, d // 10)
            P = np.zeros((d, m), np.float32)
            for j in range(m):
                P[rng.choice(d, fan_in, replace=False), j] = 1.0
            self.P = P
        self.A = np.zeros((m, n_classes), np.float32)
        self.N = np.zeros(n_classes, np.float32)

    def code(self, x):
        z = x @ self.P
        thr = np.partition(z, -self.k, axis=1)[:, -self.k][:, None]
        return (z >= thr).astype(np.float32)

    def learn(self, x, y):
        h = self.code(x)
        Y = np.zeros((len(y), self.A.shape[1]), np.float32)
        Y[np.arange(len(y)), y] = 1
        self.A += h.T @ Y
        self.N += Y.sum(0)

    def predict(self, x, batch=2000):
        seen = self.N > 0
        out = []
        for i in range(0, len(x), batch):
            A = self.A[:, seen]
            if self.vote == "norm":
                A = A / (A.sum(1, keepdims=True) + 1e-3)
            else:
                A = A / self.N[seen]
            s = self.code(x[i : i + batch]) @ A
            out.append(np.flatnonzero(seen)[s.argmax(1)])
        return np.concatenate(out)


class SparseDelta(SparseAssociative):
    """Control: identical k-WTA code and locality (only active units' synapses change), but an
    error-driven delta rule instead of counting. The update depends on the current prediction,
    so updates do NOT commute."""

    def __init__(self, *a, lr=2.0, **k):
        super().__init__(*a, **k)
        self.lr = lr
        self.W = np.zeros_like(self.A)

    def learn(self, x, y):
        h = self.code(x)
        self.N[np.unique(y)] += 1
        seen = self.N > 0
        for hi, yi in zip(h, y):
            act = np.flatnonzero(hi)
            s = np.where(seen, self.W[act].sum(0) / len(act), -np.inf)
            p = np.exp(s - s.max())
            p /= p.sum()
            p[yi] -= 1
            self.W[act] -= self.lr * p

    def predict(self, x, batch=2000):
        seen = self.N > 0
        out = []
        for i in range(0, len(x), batch):
            s = self.code(x[i : i + batch]) @ self.W[:, seen]
            out.append(np.flatnonzero(seen)[s.argmax(1)])
        return np.concatenate(out)
