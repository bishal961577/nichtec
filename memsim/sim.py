"""Toy simulation of the slow-store bottleneck: many facts written one night at a time into a
product-key memory whose slots become shared between facts.

A 'fact' is a target vector r_f (what the memory must output) reached through several paraphrase
queries q_f + noise. A lookup selects top-k slots by product keys and outputs sum_i w_i v_i.
Recall succeeds when the output is closer (cosine) to the fact's own target than to any other
stored fact's target, measured on held-out paraphrases never used for writing.

Write rules compared, all touching only the slots each fact's lookups select:
  sgd        - sequential gradient steps on the touched slots (sparse-memory-finetuning style)
  batch_ls   - each night, least squares on that night's facts only, anchored to yesterday's
               values (sequential closed-form editing, MEMIT-style batches)
  joint_ls   - each night, ridge least squares over ALL facts ever written, warm-started
               (the proposed 'exact joint memory'; order-invariant by construction)
"""
import json
import sys
import time

import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(0)
NSUB, M, DV, K, KSUB = 256, 64, 64, 32, 32  # 256^2 = 65,536 slots; query dim 64; value dim 64
N = NSUB * NSUB
K1 = rng.standard_normal((NSUB, M // 2)).astype(np.float32)
K2 = rng.standard_normal((NSUB, M // 2)).astype(np.float32)
K1 /= np.linalg.norm(K1, axis=1, keepdims=True)
K2 /= np.linalg.norm(K2, axis=1, keepdims=True)
TAU = 0.05


def lookup(Q):
    """Product-key top-K: returns slot indices (n, K) and softmax weights (n, K)."""
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    s1, s2 = Q[:, : M // 2] @ K1.T, Q[:, M // 2 :] @ K2.T
    i1 = np.argpartition(-s1, KSUB, axis=1)[:, :KSUB]
    i2 = np.argpartition(-s2, KSUB, axis=1)[:, :KSUB]
    c = np.take_along_axis(s1, i1, 1)[:, :, None] + np.take_along_axis(s2, i2, 1)[:, None, :]
    c = c.reshape(len(Q), -1)
    top = np.argpartition(-c, K, axis=1)[:, :K]
    sc = np.take_along_axis(c, top, 1)
    idx = (i1[np.arange(len(Q))[:, None], top // KSUB] * NSUB + i2[np.arange(len(Q))[:, None], top % KSUB])
    w = np.exp((sc - sc.max(1, keepdims=True)) / TAU)
    return idx, (w / w.sum(1, keepdims=True)).astype(np.float32)


def make_facts(n, p_train, p_test, sigma):
    base = rng.standard_normal((n, M)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    para = lambda p: base[:, None, :] + sigma * rng.standard_normal((n, p, M)).astype(np.float32) / np.sqrt(M)
    R = rng.standard_normal((n, DV)).astype(np.float32)
    R /= np.linalg.norm(R, axis=1, keepdims=True)
    return para(p_train), para(p_test), R


def rows(idx, w):
    n = len(idx)
    return sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), K), idx.ravel())), shape=(n, N))


def recall(V, idx, w, fact_ids, R_all):
    out = np.einsum("nk,nkd->nd", w, V[idx])
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    sims = out @ R_all.T
    return float((sims.argmax(1) == fact_ids).mean())


def cosine(V, idx, w, fact_ids, R_all):
    out = np.einsum("nk,nkd->nd", w, V[idx])
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return float((out * R_all[fact_ids]).sum(1).mean())


def cg_solve(W, B, V0, lam, iters, tol=1e-4):
    """Block CG for (W^T W + lam I) V = W^T B, warm-started at V0 (all slots)."""
    A = lambda X: W.T @ (W @ X) + lam * X
    rhs = W.T @ B
    X = V0.copy()
    Rr = rhs - A(X)
    P = Rr.copy()
    rs = (Rr * Rr).sum(0)
    b = np.sqrt((rhs * rhs).sum(0)) + 1e-12
    for it in range(iters):
        AP = A(P)
        alpha = rs / ((P * AP).sum(0) + 1e-30)
        X += P * alpha
        Rr -= AP * alpha
        rs_new = (Rr * Rr).sum(0)
        if np.all(np.sqrt(rs_new) / b < tol):
            break
        P = Rr + P * (rs_new / (rs + 1e-30))
        rs = rs_new
    return X, it + 1


def run(total, night, p_train=4, p_test=2, sigma=0.6, sgd_lr=0.5, sgd_steps=3, lam=1e-3, anchor=1e-2):
    Qtr, Qte, R = make_facts(total, p_train, p_test, sigma)
    itr, wtr = lookup(Qtr.reshape(-1, M))
    ite, wte = lookup(Qte.reshape(-1, M))
    itr, wtr = itr.reshape(total, p_train, K), wtr.reshape(total, p_train, K)
    ite, wte = ite.reshape(total, p_test, K), wte.reshape(total, p_test, K)
    # overlap diagnostics: how many slots each fact's paraphrases share
    share = np.mean([len(set(ite[f, 0]) & set(itr[f].ravel())) / K for f in range(200)])
    V = {"sgd": np.zeros((N, DV), np.float32), "batch_ls": np.zeros((N, DV), np.float32),
         "joint_ls": np.zeros((N, DV), np.float32)}
    log = []
    early = np.arange(min(1000, night))
    for s in range(0, total, night):
        e = min(total, s + night)
        new = np.arange(s, e)
        # --- sgd: sequential per fact
        Vs = V["sgd"]
        for f in new:
            for _ in range(sgd_steps):
                ix, ww = itr[f], wtr[f]
                out = np.einsum("pk,pkd->pd", ww, Vs[ix])
                g = (out - R[f])[:, None, :] * ww[:, :, None]
                np.add.at(Vs, ix.ravel(), -sgd_lr * g.reshape(-1, DV))
        # --- batch least squares on tonight's facts, anchored to yesterday
        Wn = rows(itr[new].reshape(-1, K), wtr[new].reshape(-1, K))
        Bn = np.repeat(R[new], p_train, axis=0)
        Vb = V["batch_ls"]
        # min ||Wn V - Bn||^2 + anchor ||V - Vb||^2  ->  (Wn^T Wn + anchor I) V = Wn^T Bn + anchor Vb
        A = lambda X: Wn.T @ (Wn @ X) + anchor * X
        rhs = Wn.T @ Bn + anchor * Vb
        X = Vb.copy(); Rr = rhs - A(X); P = Rr.copy(); rs = (Rr * Rr).sum(0)
        for _ in range(200):
            AP = A(P); al = rs / ((P * AP).sum(0) + 1e-30); X += P * al; Rr -= AP * al
            rn = (Rr * Rr).sum(0)
            if np.all(np.sqrt(rn) / (np.sqrt((rhs * rhs).sum(0)) + 1e-12) < 1e-4): break
            P = Rr + P * (rn / (rs + 1e-30)); rs = rn
        V["batch_ls"] = X.astype(np.float32)
        # --- joint least squares over everything written so far
        allf = np.arange(0, e)
        Wa = rows(itr[allf].reshape(-1, K), wtr[allf].reshape(-1, K))
        Ba = np.repeat(R[allf], p_train, axis=0)
        t = time.time()
        V["joint_ls"], iters = cg_solve(Wa, Ba, V["joint_ls"].astype(np.float64), lam, 300)
        V["joint_ls"] = V["joint_ls"].astype(np.float32)
        solve_s = time.time() - t
        # --- evaluate on held-out paraphrases
        R_all = R[:e]
        row = {"facts": int(e), "constraints_per_slot": e * p_train / N, "joint_iters": iters, "joint_solve_s": round(solve_s, 1)}
        for name, Vm in V.items():
            q = lambda ids: recall(Vm, ite[ids].reshape(-1, K), wte[ids].reshape(-1, K), np.repeat(ids, p_test), R_all)
            qt = lambda ids: recall(Vm, itr[ids].reshape(-1, K), wtr[ids].reshape(-1, K), np.repeat(ids, p_train), R_all)
            ct = lambda ids: cosine(Vm, itr[ids].reshape(-1, K), wtr[ids].reshape(-1, K), np.repeat(ids, p_train), R_all)
            ch = lambda ids: cosine(Vm, ite[ids].reshape(-1, K), wte[ids].reshape(-1, K), np.repeat(ids, p_test), R_all)
            samp = rng.choice(e, min(2000, e), replace=False)
            row[name] = {"first_1000": round(q(early), 3), "latest_night": round(q(new[-1000:]), 3), "all": round(q(samp), 3),
                         "first_1000_written_wording": round(qt(early), 3),
                         "cos_written": round(ct(early), 3), "cos_heldout": round(ch(early), 3)}
        log.append(row)
        print(json.dumps(row), flush=True)
    return {"slot_share_heldout_vs_train": share, "log": log}


if __name__ == "__main__":
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    night = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    sigma = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    out = run(total, night, sigma=sigma)
    print("slot share held-out vs written wordings:", round(out["slot_share_heldout_vs_train"], 3))
    json.dump(out, open(f"/home/user/nichtec/memsim/result_{total}_{night}.json", "w"), indent=1)
