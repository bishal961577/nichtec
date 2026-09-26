"""SPARC: Sparse, Plastic, Accumulating, Recruiting, Consolidating learner.

Two stages, both trained online in a single pass with purely local rules (no backprop):

1. PatchDictionary  - an unsupervised feature layer. Each unit is a prototype of a small image
   patch. For an incoming patch the best-matching unit moves toward it with learning rate
   1/n (n = how often the unit has won), so a unit's weight is the exact running mean of the
   patches it has captured: heavily used units consolidate and stop drifting, while the rate
   of change of the whole layer falls as experience accumulates. A patch that matches no unit
   well enough ("novelty") recruits a fresh, unused unit instead of overwriting an old one.
   Each unit's response depends only on its own weights, so recruiting new units never
   changes what old units report.

2. SparseAssociative (in baselines.py) - features are expanded into a large k-winners-take-all
   sparse code and associated with the label by counting (A += h y^T). That update is a sum,
   so it is exactly order-invariant: learning task B cannot overwrite what was stored for
   task A. Streaming LDA over the same features is provided as a linear alternative.
"""
import numpy as np

from .baselines import NCM, SLDA, SparseAssociative


def _patches(img, p):
    """All p x p patches (stride 1) of a batch of images -> (B, L, p*p)."""
    B, H, W = img.shape
    s = img.strides
    view = np.lib.stride_tricks.as_strided(
        img, shape=(B, H - p + 1, W - p + 1, p, p), strides=(s[0], s[1], s[2], s[1], s[2]), writeable=False
    )
    return view.reshape(B, (H - p + 1) * (W - p + 1), p * p)


def _normalize(P, eps):
    P = P - P.mean(-1, keepdims=True)
    n = np.sqrt((P * P).sum(-1, keepdims=True))
    return P / (n + eps), n[..., 0]


class PatchDictionary:
    def __init__(self, k_max=256, p=5, recruit=0.5, tau=0.4, grid=3, eps=0.1, contrast=0.3,
                 samples=24, mature=None, wta=False, seed=0):
        self.k_max, self.p, self.recruit, self.tau = k_max, p, recruit, tau
        self.mature = mature  # a unit that has won this many times stops changing (None = never)
        self.wta = wta  # encode each patch location by its best-matching unit only
        self.grid, self.eps, self.contrast, self.samples = grid, eps, contrast, samples
        self.D = np.zeros((k_max, p * p), np.float32)
        self.n = np.zeros(k_max, np.int64)
        self.k = 0  # units recruited so far
        self.rng = np.random.default_rng(seed)
        self.frozen = False
        self.log = []  # (images seen, units in use)
        self.seen = 0

    @property
    def dim(self):
        return self.k_max * self.grid * self.grid

    # --- learning -------------------------------------------------------------------------
    def learn(self, imgs):
        self.seen += len(imgs)
        if self.frozen:
            return
        P, norm = _normalize(_patches(imgs, self.p), self.eps)
        for b in range(len(imgs)):
            cand = np.flatnonzero(norm[b] > self.contrast)
            if len(cand) == 0:
                continue
            pick = self.rng.choice(cand, min(self.samples, len(cand)), replace=False)
            for v in P[b, pick]:
                v = v / (np.linalg.norm(v) + 1e-8)
                if self.k:
                    s = self.D[: self.k] @ v
                    w = int(s.argmax())
                    best = s[w]
                else:
                    best = -1.0
                if best < self.recruit and self.k < self.k_max:
                    self.D[self.k] = v  # novelty -> recruit a fresh unit
                    self.n[self.k] = 1
                    self.k += 1
                elif self.mature is not None and self.n[w] >= self.mature:
                    continue  # consolidated unit: frozen
                else:
                    self.n[w] += 1  # consolidation: learning rate 1/n
                    d = self.D[w] + (v - self.D[w]) / self.n[w]
                    self.D[w] = d / (np.linalg.norm(d) + 1e-8)
        self.log.append((self.seen, self.k))

    # --- encoding -------------------------------------------------------------------------
    def encode(self, imgs, batch=500):
        out = np.zeros((len(imgs), self.dim), np.float32)
        side = imgs.shape[1] - self.p + 1
        g = self.grid
        edges = np.linspace(0, side, g + 1).astype(int)
        for i in range(0, len(imgs), batch):
            P, _ = _normalize(_patches(imgs[i : i + batch], self.p), self.eps)
            S = P @ self.D.T
            if self.wta:
                S = np.where(S >= S.max(-1, keepdims=True), S, 0)
            A = np.maximum(S - self.tau, 0)  # (B, L, K): each unit fires independently
            A = A.reshape(len(P), side, side, self.k_max)
            F = np.stack(
                [A[:, edges[r] : edges[r + 1], edges[c] : edges[c + 1]].sum((1, 2)) for r in range(g) for c in range(g)],
                axis=2,
            )  # (B, K, g*g)
            out[i : i + batch] = F.reshape(len(P), -1)
        out = np.sqrt(out)  # power normalisation
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
        return out


class SPARC:
    """Feature layer + commutative readout, trained online on a stream of (image, label) batches."""

    def __init__(self, n_classes, readout="assoc", seed=0, feat_kw=None, assoc_kw=None):
        self.feat = PatchDictionary(seed=seed, **(feat_kw or {}))
        d = self.feat.dim
        if readout == "assoc":
            self.read = SparseAssociative(d, n_classes, seed=seed, dense=True, **(assoc_kw or {}))
        elif readout == "slda":
            self.read = SLDA(d, n_classes, shrinkage=1e-1)
        elif readout == "ncm":
            self.read = NCM(d, n_classes)
        else:
            raise ValueError(readout)

    def learn(self, imgs, y):
        self.feat.learn(imgs)  # unsupervised, local
        self.read.learn(self.feat.encode(imgs), y)  # supervised, local, commutative

    def predict(self, imgs):
        return self.read.predict(self.feat.encode(imgs))
