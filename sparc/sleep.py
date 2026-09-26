"""'Sleep': consolidate a fast commutative memory into a slow error-driven learner.

The fast memory is streaming LDA over features: running class means and one running covariance.
It commutes, so it is unaffected by the order of experience. The slow learner is an ordinary
backprop MLP, which does not commute and forgets when trained on an ordered stream.

After each task (the "night"), a generative model is refreshed from the fast memory's
sufficient statistics: one Gaussian per class seen so far, sharing the covariance. While learning
the next task (the "day"), every minibatch of real new data is interleaved with the same number
of pseudo-examples of old classes sampled from that model. No raw example is ever stored or
replayed. This is complementary-learning-systems theory (McClelland et al. 1995) with the
hippocampal store replaced by sufficient statistics; generative feature replay of this kind is
itself known (e.g. prototype augmentation, Zhu et al. 2021).
"""
import numpy as np

from .baselines import MLP, SLDA
from .protocol import summarize, task_accuracy


def _sampler(mem, rng, shrink=0.1):
    seen = np.flatnonzero(mem.N > 0)
    mu = mem.S[seen] / mem.N[seen, None]
    n = mem.N.sum()
    Sw = (mem.XX - (mu.T * mem.N[seen]) @ mu) / n
    Sw = (1 - shrink) * Sw + shrink * np.trace(Sw) / len(Sw) * np.eye(len(Sw))
    L = np.linalg.cholesky(Sw + 1e-8 * np.eye(len(Sw)))

    def sample(k):
        c = rng.integers(0, len(seen), k)
        x = mu[c] + rng.standard_normal((k, len(Sw))) @ L.T
        return x.astype(np.float32), seen[c]

    return sample


def run_sleep(F, y, Ft, yt, stream, tasks, n_classes, *, seed=0, bs=10, lr=1e-3, hidden=(256,), replay=True):
    rng = np.random.default_rng(seed)
    net = MLP(F.shape[1], n_classes, hidden=hidden, lr=lr, seed=seed)
    mem = SLDA(F.shape[1], n_classes, shrinkage=0.1)
    sample = None
    matrix = []
    for idx in stream:
        for i in range(0, len(idx), bs):
            j = idx[i : i + bs]
            xb, yb = F[j], y[j]
            mem.learn(xb, yb)
            if replay and sample is not None:
                rx, ry = sample(bs)
                xb, yb = np.concatenate([xb, rx]), np.concatenate([yb, ry])
            net.step(xb, yb)
        if replay:
            sample = _sampler(mem, rng)  # night: refresh the generative model from the fast memory
        if len(stream) > 1:
            matrix.append(task_accuracy(net.predict(Ft), yt, tasks))
    pred = net.predict(Ft)
    if len(stream) == 1:
        matrix = [task_accuracy(pred, yt, tasks)]
    return summarize(matrix, tasks, pred, yt, np.concatenate([np.asarray(t) for t in tasks]))
