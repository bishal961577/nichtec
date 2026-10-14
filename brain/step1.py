"""Step 1 toward a brain built from tally marks (pre-registered in reports/Additive feature learning 2026.md).

Question: does one round of feature learning computed only from additive statistics (Hebbian co-activation counts plus
a ridge readout; see brain/agop_counts_check.py) make random features better on CIFAR-10?

Pipeline (fixed before any run):
  - CIFAR-10; 45,000 training images, 5,000 held out to choose the ridge strength, 10,000 test images.
  - 5x5x3 patches at every position (reflection padding), each contrast-normalised, then ZCA-whitened (epsilon 0.1)
    with the mean and covariance summed over all training patches.
  - D random ReLU filters (unit-norm Gaussian rows, no bias), sum-pooled over the four image quadrants (4D features),
    standardised with summed means and variances, ridge readout on one-hot labels.
  - B0: fixed random filters. B1: one round of feature learning, M = A^T [sum_q (W_q^T W_q) * G_q] A, where G_q counts
    how often each pair of units fires together in quadrant q; patches are mapped by M^(1/2) (trace-normalised) and
    fresh random filters are applied. B2: two rounds (the AGOP is always taken with respect to the whitened patch).
  - Every statistic is a sum over images, so each model is the same for any order of its training data.

Decision lines (E1, D = 2,048): pass if B1 >= B0 + 3.0 points and B2 >= B0 + 5.0; stop if B1 < B0 + 1.0.
E2 (only if E1 passes): features learned from the first half of a class-ordered stream versus a shuffled one.

  python brain/step1.py                    # full run (downloads CIFAR-10, 163 MB)
  python brain/step1.py --fake             # small synthetic smoke test
"""
import argparse
import json
import os
import pickle
import sys
import tarfile
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)    # ridge strength relative to the mean eigenvalue
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def load_cifar():
    folder = os.path.join(ROOT, "data", "cifar10")
    base = os.path.join(folder, "cifar-10-batches-py")
    if not os.path.isdir(base):
        os.makedirs(folder, exist_ok=True)
        tgz = os.path.join(folder, "cifar-10-python.tar.gz")
        if not os.path.exists(tgz):
            log("downloading CIFAR-10 (163 MB) ...")
            urllib.request.urlretrieve(URL, tgz)
        with tarfile.open(tgz) as t:
            try:
                t.extractall(folder, filter="data")
            except TypeError:
                t.extractall(folder)

    def part(name):
        d = pickle.load(open(os.path.join(base, name), "rb"), encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32), np.array(d[b"labels"], dtype=np.int64)

    xs, ys = zip(*[part(f"data_batch_{i}") for i in range(1, 6)])
    xte, yte = part("test_batch")
    return np.concatenate(xs), np.concatenate(ys), xte, yte


def fake_data(n_train=2400, n_test=600, seed=0):
    """Ten classes of oriented, coloured gratings plus noise: enough structure for features to matter."""
    rng = np.random.default_rng(seed)

    def make(n):
        y = rng.integers(0, 10, n)
        r, c = np.mgrid[0:32, 0:32]
        x = np.empty((n, 3, 32, 32), dtype=np.float64)
        for i in range(n):
            ang = y[i] * np.pi / 10 + rng.normal(0, 0.15)
            freq = 0.3 + 0.05 * (y[i] % 3)
            g = np.sin(freq * (np.cos(ang) * r + np.sin(ang) * c) + rng.uniform(0, 6.28))
            for ch in range(3):
                x[i, ch] = 128 + 60 * g * (0.5 + 0.5 * ((y[i] + ch) % 2)) + rng.normal(0, 40, (32, 32))
        return np.clip(x, 0, 255).astype(np.uint8), y

    xtr, ytr = make(n_train)
    xte, yte = make(n_test)
    return xtr, ytr, xte, yte


# ----------------------------------------------------------------------------------------------------------------
# patches, features and additive statistics
# ----------------------------------------------------------------------------------------------------------------
class Net:
    def __init__(self, dev, bs):
        self.dev, self.bs = dev, bs
        self.mu = None          # mean of contrast-normalised training patches (75)
        self.Z = None           # ZCA whitening matrix (75 x 75)

    def patches(self, x):
        """uint8 images (B,3,32,32) -> contrast-normalised patches (B, 1024, 75), float64 on the device."""
        x = torch.as_tensor(x, device=self.dev).double()
        p = F.unfold(F.pad(x, (2, 2, 2, 2), mode="reflect"), 5).transpose(1, 2)
        p = p - p.mean(-1, keepdim=True)
        return p / torch.sqrt(p.var(-1, keepdim=True) + 10.0)

    def fit_whitening(self, X):
        s1 = torch.zeros(75, dtype=torch.float64, device=self.dev)
        s2 = torch.zeros(75, 75, dtype=torch.float64, device=self.dev)
        n = 0
        for i in range(0, len(X), self.bs):
            p = self.patches(X[i:i + self.bs]).reshape(-1, 75)
            s1 += p.sum(0)
            s2 += p.T @ p
            n += p.shape[0]
        self.mu = s1 / n
        C = s2 / n - torch.outer(self.mu, self.mu)
        e, U = torch.linalg.eigh(C)
        self.Z = U @ torch.diag(1.0 / torch.sqrt(e.clamp(min=0) + 0.1)) @ U.T

    def whitened(self, x):
        return (self.patches(x) - self.mu) @ self.Z.T          # (B, 1024, 75)

    @staticmethod
    def pool(h):
        """(B, 1024, D) -> (B, 4D): sums over the four 16x16 quadrants, quadrant-major."""
        B, _, D = h.shape
        return h.reshape(B, 2, 16, 2, 16, D).sum((2, 4)).reshape(B, 4 * D)

    @staticmethod
    def quadrant_rows(s):
        """(B, 1024, D) -> (4, B*256, D): the patches of each quadrant."""
        B, _, D = s.shape
        return s.reshape(B, 2, 16, 2, 16, D).permute(1, 3, 0, 2, 4, 5).reshape(4, B * 256, D)

    def features(self, x, P):
        """Pooled features for images x under the patch map P (D x 75, applied to whitened patches)."""
        z = self.whitened(x).float() @ P.float().T
        return self.pool(torch.relu(z))

    def stats(self, X, Y, P, counts):
        """One pass: summed pooled features, their products with each other and with the labels, and (optionally)
        the per-quadrant co-activation counts G_q. Returns float64 tensors."""
        D = P.shape[0]
        Fd = 4 * D
        s1 = torch.zeros(Fd, dtype=torch.float64, device=self.dev)
        s2 = torch.zeros(Fd, Fd, dtype=torch.float64, device=self.dev)
        sy = torch.zeros(Fd, 10, dtype=torch.float64, device=self.dev)
        G = torch.zeros(4, D, D, dtype=torch.float32, device=self.dev) if counts else None   # exact below 2^24
        Pf = P.float()
        for i in range(0, len(X), self.bs):
            w = self.whitened(X[i:i + self.bs]).float()
            z = w @ Pf.T
            phi = self.pool(torch.relu(z)).double()
            y = F.one_hot(torch.as_tensor(Y[i:i + self.bs], device=self.dev), 10).double()
            s1 += phi.sum(0)
            s2.addmm_(phi.T, phi)
            sy.addmm_(phi.T, y)
            if counts:
                S = self.quadrant_rows((z > 0).float())
                for q in range(4):
                    G[q].addmm_(S[q].T, S[q])
        return {"n": len(X), "s1": s1, "s2": s2, "sy": sy, "ybar": torch.as_tensor(np.bincount(Y, minlength=10) / len(Y),
                                                                                   device=self.dev), "G": G}


def ridge_prepare(st):
    """Standardised covariance of the pooled features and their covariance with the labels, from summed statistics
    (moved to the CPU in float64, where the solves run)."""
    n = st["n"]
    mu = st["s1"] / n
    var = (torch.diagonal(st["s2"]) / n - mu ** 2).clamp(min=0)
    sd = torch.sqrt(var) + 1e-6 * torch.sqrt(var).mean() + 1e-12
    C = st["s2"].cpu() / n
    C -= torch.outer(mu.cpu(), mu.cpu())
    C /= torch.outer(sd.cpu(), sd.cpu())
    Cy = ((st["sy"] / n - torch.outer(mu, st["ybar"])) / sd[:, None]).cpu()
    st["s2"] = None                                                        # free the device copy
    return {"C": C, "Cy": Cy, "mu": mu, "sd": sd, "ybar": st["ybar"], "scale": torch.trace(C) / C.shape[0]}


def ridge_fit(prep, lam):
    """Ridge readout on standardised features. Returns a predictor dict."""
    C = prep["C"].clone()
    C.diagonal().add_(lam * prep["scale"])
    L = torch.linalg.cholesky(C)
    del C
    W = torch.cholesky_solve(prep["Cy"], L)                                # (F, 10)
    return {"W": W.to(prep["mu"].device), "mu": prep["mu"], "sd": prep["sd"], "ybar": prep["ybar"], "lam": lam}


def predict(net, X, P, model):
    out = []
    for i in range(0, len(X), net.bs):
        phi = net.features(X[i:i + net.bs], P.float()).double()
        out.append((((phi - model["mu"]) / model["sd"]) @ model["W"] + model["ybar"]).argmax(1).cpu())
    return torch.cat(out).numpy()


def fit_with_validation(net, st, Xv, Yv, P):
    """Choose the ridge strength on the held-out images; return the chosen model and every validation accuracy."""
    feats = []
    for i in range(0, len(Xv), net.bs):
        feats.append(net.features(Xv[i:i + net.bs], P.float()).double())
    feats = torch.cat(feats)
    prep = ridge_prepare(st)
    best, accs = None, {}
    for lam in LAMBDAS:
        m = ridge_fit(prep, lam)
        acc = float(((((feats - m["mu"]) / m["sd"]) @ m["W"] + m["ybar"]).argmax(1).cpu().numpy() == Yv).mean())
        accs[str(lam)] = round(acc, 4)
        if best is None or acc > best[0]:
            best = (acc, m)
    return best[1], accs


def agop_from_counts(model, G, A, R):
    """M = R^T A^T [sum_q (W_q^T W_q) * G_q] A R  (the AGOP with respect to the whitened patch)."""
    D = A.shape[0]
    Weff = (model["W"] / model["sd"][:, None]).T                           # (10, 4D): df/d(pooled feature)
    inner = torch.zeros(D, D, dtype=torch.float64, device=A.device)
    for q in range(4):
        Wq = Weff[:, q * D:(q + 1) * D]
        inner += (Wq.T @ Wq) * G[q].double()
    AR = A @ R
    return AR.T @ inner @ AR


def agop_direct(net, X, model, A, R):
    """Per-patch AGOP computed from explicit gradients (for the identity check only)."""
    D = A.shape[0]
    Weff = (model["W"] / model["sd"][:, None]).T
    AR = A @ R
    M = torch.zeros(75, 75, dtype=torch.float64, device=A.device)
    G = torch.zeros(4, D, D, dtype=torch.float64, device=A.device)
    for i in range(len(X)):
        z = net.whitened(X[i:i + 1]) @ AR.T                                # (1, 1024, D)
        S = net.quadrant_rows((z > 0).double())                            # (4, 256, D)
        for q in range(4):
            G[q] += S[q].T @ S[q]
            J = (S[q][:, None, :] * Weff[None, :, q * D:(q + 1) * D]) @ AR    # (256, 10, 75)
            M += torch.einsum("pci,pcj->ij", J, J)
    return M, G


def sqrt_psd(M):
    M = M * (M.shape[0] / torch.trace(M))
    e, U = torch.linalg.eigh(M)
    return U @ torch.diag(torch.sqrt(e.clamp(min=0))) @ U.T


def random_filters(D, seed, dev):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(D, 75, generator=g, dtype=torch.float64)
    return (A / A.norm(dim=1, keepdim=True)).to(dev)


# ----------------------------------------------------------------------------------------------------------------
# a small backprop CNN, for context only
# ----------------------------------------------------------------------------------------------------------------
def cnn_accuracy(Xtr, Ytr, Xte, Yte, dev, epochs, seed=0, order=None):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                        nn.Flatten(), nn.Linear(128 * 16, 10)).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    mean = torch.tensor([125.3, 123.0, 113.9], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([63.0, 62.1, 66.7], device=dev).view(1, 3, 1, 1)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        idx = rng.permutation(len(Xtr)) if order is None else order
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            x = (torch.as_tensor(Xtr[b], device=dev).float() - mean) / std
            loss = F.cross_entropy(net(x), torch.as_tensor(Ytr[b], device=dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(Xte), 500):
            x = (torch.as_tensor(Xte[i:i + 500], device=dev).float() - mean) / std
            correct += int((net(x).argmax(1).cpu().numpy() == Yte[i:i + 500]).sum())
    return correct / len(Xte)


# ----------------------------------------------------------------------------------------------------------------
# experiments
# ----------------------------------------------------------------------------------------------------------------
def rounds(net, Xtr, Ytr, Xv, Yv, Xte, Yte, D, n_rounds, seed, R=None):
    """B0, B1, ..., B_{n_rounds}: returns test accuracies and details per round."""
    dev = net.dev
    R = torch.eye(75, dtype=torch.float64, device=dev) if R is None else R
    out = []
    base_bs = net.bs
    net.bs = max(8, base_bs * 2048 // D)
    for t in range(n_rounds + 1):
        A = random_filters(D, seed + 1000 * t, dev)
        P = A @ R
        t0 = time.time()
        st = net.stats(Xtr, Ytr, P, counts=t < n_rounds)
        model, vacc = fit_with_validation(net, st, Xv, Yv, P)
        acc = float((predict(net, Xte, P, model) == Yte).mean())
        out.append({"round": t, "test": round(acc, 4), "val_by_lambda": vacc, "lambda": model["lam"],
                    "seconds": round(time.time() - t0, 1)})
        log(f"D={D} B{t}: test {acc:.4f} (validation {max(vacc.values()):.4f}, lambda {model['lam']}, "
            f"{time.time() - t0:.0f}s)")
        if t < n_rounds:
            M = agop_from_counts(model, st["G"], A, R)
            R = sqrt_psd(M)
            out[-1]["agop_top_eigen_share"] = round(float(torch.linalg.eigvalsh(M)[-5:].sum() / torch.trace(M)), 4)
        del st
    net.bs = base_bs
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="small synthetic smoke test")
    ap.add_argument("--dims", default="2048,1024,4096", help="D values; the first is the pre-registered decision")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-cnn", action="store_true")
    ap.add_argument("--force-e2", action="store_true", help="smoke tests only: run E2 whatever E1 shows")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    out = args.out or os.path.join(ROOT, "results", "step1_fake" if args.fake else "step1")
    os.makedirs(out, exist_ok=True)
    R = {"config": vars(args), "device": str(dev), "E1": {}, "notes": []}

    def save():
        json.dump(R, open(os.path.join(out, "result.json"), "w"), indent=1)
        open(os.path.join(out, "summary.md"), "w").write(report(R))

    if args.fake:
        Xall, Yall, Xte, Yte = fake_data()
        dims = [int(d) for d in "64,32".split(",")]
        n_val, n_check = 400, 20
    else:
        Xall, Yall, Xte, Yte = load_cifar()
        dims = [int(d) for d in args.dims.split(",")]
        n_val, n_check = 5000, 500
    perm = np.random.default_rng(0).permutation(len(Xall))
    Xv, Yv = Xall[perm[:n_val]], Yall[perm[:n_val]]
    Xtr, Ytr = Xall[perm[n_val:]], Yall[perm[n_val:]]
    log(f"device {dev}; train {len(Xtr):,}, validation {len(Xv):,}, test {len(Xte):,}; D values {dims}")

    net = Net(dev, args.bs)
    net.fit_whitening(Xtr)
    log("whitening fitted")

    # identity check: AGOP from counts == AGOP from explicit gradients (pre-registered, relative 1e-6)
    Dc = dims[0]
    A0 = random_filters(Dc, 0, dev)
    I75 = torch.eye(75, dtype=torch.float64, device=dev)
    st = net.stats(Xtr[:n_check], Ytr[:n_check], A0, counts=True)
    G32 = st["G"]
    m0 = ridge_fit(ridge_prepare(st), 1e-2)
    # explicit gradients in float64 on the CPU (consumer GPUs are slow in float64)
    cpu = torch.device("cpu")
    net_cpu = Net(cpu, 1)
    net_cpu.mu, net_cpu.Z = net.mu.cpu(), net.Z.cpu()
    m0c = {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in m0.items()}
    A0c, I75c = A0.cpu(), I75.cpu()
    Md, G64 = agop_direct(net_cpu, Xtr[:n_check], m0c, A0c, I75c)
    rel = float((agop_from_counts(m0c, G64, A0c, I75c) - Md).abs().max() / Md.abs().max())
    rel32 = float((agop_from_counts(m0c, G32.cpu(), A0c, I75c) - Md).abs().max() / Md.abs().max())
    R["identity_check"] = {"images": n_check, "relative_max_difference": rel, "pass": rel < 1e-6,
                           "production_float32_counts_relative_difference": rel32}
    log(f"identity check on {n_check} images: relative difference {rel:.2e} (float32 counts as used: {rel32:.2e})")
    del st
    save()

    # E1
    for i, D in enumerate(dims):
        try:
            R["E1"][str(D)] = rounds(net, Xtr, Ytr, Xv, Yv, Xte, Yte, D, 2, seed=D)
        except torch.cuda.OutOfMemoryError:
            R["notes"].append(f"D={D}: out of GPU memory; skipped")
            log(f"D={D}: out of GPU memory, skipped")
            torch.cuda.empty_cache()
            continue
        if i == 0:
            b = [r["test"] for r in R["E1"][str(D)]]
            d1, d2 = (b[1] - b[0]) * 100, (b[2] - b[0]) * 100
            R["E1_decision"] = {"D": D, "B0": b[0], "B1": b[1], "B2": b[2], "B1_minus_B0": round(d1, 2),
                                "B2_minus_B0": round(d2, 2),
                                "verdict": "pass" if d1 >= 3.0 and d2 >= 5.0 else "stop" if d1 < 1.0 else "in between"}
            log("E1 decision:", json.dumps(R["E1_decision"]))
        save()

    # E2 (protocol fixed before running; see the report): only if E1 passes
    if R.get("E1_decision", {}).get("verdict") == "pass" or args.force_e2:
        D = dims[0]
        b0 = R["E1"][str(D)][0]
        lam0 = b0["lambda"]
        half = len(Xtr) // 2
        ordered = np.argsort(Ytr, kind="stable")
        shuffled = np.random.default_rng(1).permutation(len(Xtr))
        A0 = random_filters(D, D, dev)
        A1 = random_filters(D, D + 1000, dev)
        e2 = {}
        for name, idx in (("shuffled", shuffled[:half]), ("ordered", ordered[:half])):
            st = net.stats(Xtr[idx], Ytr[idx], A0, counts=True)
            m = ridge_fit(ridge_prepare(st), lam0)
            Rm = sqrt_psd(agop_from_counts(m, st["G"], A0, I75))
            del st
            P = A1 @ Rm
            st = net.stats(Xtr, Ytr, P, counts=False)
            model, _ = fit_with_validation(net, st, Xv, Yv, P)
            pred = predict(net, Xte, P, model)
            e2[name] = {"test": round(float((pred == Yte).mean()), 4),
                        "classes_0_4": round(float((pred == Yte)[Yte < 5].mean()), 4),
                        "classes_5_9": round(float((pred == Yte)[Yte >= 5].mean()), 4),
                        "stage1_classes": sorted(set(Ytr[idx].tolist()))}
            log(f"E2 {name}: {json.dumps(e2[name])}")
            del st
        gap = (e2["shuffled"]["test"] - e2["ordered"]["test"]) * 100
        e2["gap_points"] = round(gap, 2)
        e2["verdict"] = ("pass" if gap <= 2.0 and e2["ordered"]["test"] >= b0["test"] + 0.03
                         else "stop" if gap > 10.0 else "in between")
        R["E2"] = e2
        log("E2 decision:", e2["verdict"], f"gap {gap:.2f} points")
        save()

    # context: a small backprop CNN (not part of the decision)
    if not args.no_cnn:
        epochs = 2 if args.fake else 10
        R["cnn_context"] = {"shuffled_10_epochs": round(cnn_accuracy(Xtr, Ytr, Xte, Yte, dev, epochs), 4),
                            "one_pass_shuffled": round(cnn_accuracy(Xtr, Ytr, Xte, Yte, dev, 1), 4),
                            "one_pass_class_ordered": round(cnn_accuracy(Xtr, Ytr, Xte, Yte, dev, 1,
                                                                         order=np.argsort(Ytr, kind="stable")), 4)}
        log("CNN context:", json.dumps(R["cnn_context"]))
    R["total_minutes"] = round((time.time() - T0) / 60, 1)
    save()
    log("done")
    return R


def report(R):
    L = ["# Step 1: feature learning from tally marks on CIFAR-10" + (" (synthetic smoke test)" if R["config"]["fake"]
                                                                       else ""), ""]
    ic = R.get("identity_check")
    if ic:
        L += [f"Identity check (AGOP from co-activation counts vs explicit gradients, {ic['images']} images): relative "
              f"difference {ic['relative_max_difference']:.2e} ({'pass' if ic['pass'] else 'FAIL'}).", ""]
    if R["E1"]:
        L += ["## E1: test accuracy", "", "| D | B0 (random features) | B1 (one round) | B2 (two rounds) | B1 − B0 | "
              "B2 − B0 |", "|---|---|---|---|---|---|"]
        for D, rs in R["E1"].items():
            t = [r["test"] for r in rs]
            row = [f"{100 * x:.2f}%" for x in t] + ["–"] * (3 - len(t))
            d1 = f"{100 * (t[1] - t[0]):+.2f}" if len(t) > 1 else "–"
            d2 = f"{100 * (t[2] - t[0]):+.2f}" if len(t) > 2 else "–"
            L.append(f"| {D} | {' | '.join(row)} | {d1} | {d2} |")
        L.append("")
    dec = R.get("E1_decision")
    if dec:
        L += [f"**E1 verdict (pre-registered lines, D = {dec['D']}): {dec['verdict']}.** Pass needs B1 − B0 ≥ +3.0 and "
              f"B2 − B0 ≥ +5.0; stop if B1 − B0 < +1.0.", ""]
    if "E2" in R:
        e = R["E2"]
        L += ["## E2: features learned from a class-ordered vs a shuffled first half", "",
              f"Shuffled: {100 * e['shuffled']['test']:.2f}% (classes 0–4 {100 * e['shuffled']['classes_0_4']:.2f}%, "
              f"5–9 {100 * e['shuffled']['classes_5_9']:.2f}%). Ordered: {100 * e['ordered']['test']:.2f}% (classes 0–4 "
              f"{100 * e['ordered']['classes_0_4']:.2f}%, 5–9 {100 * e['ordered']['classes_5_9']:.2f}%). Gap "
              f"{e['gap_points']:+.2f} points. **Verdict: {e['verdict']}.**", ""]
    if "cnn_context" in R:
        c = R["cnn_context"]
        L += [f"Context, small backprop CNN: 10 shuffled epochs {100 * c['shuffled_10_epochs']:.2f}%; one shuffled pass "
              f"{100 * c['one_pass_shuffled']:.2f}%; one class-ordered pass {100 * c['one_pass_class_ordered']:.2f}%.", ""]
    for n in R.get("notes", []):
        L.append(f"Note: {n}")
    if "total_minutes" in R:
        L.append(f"Total time: {R['total_minutes']} min on {R['device']}.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main(sys.argv[1:])
