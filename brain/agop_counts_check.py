"""Check: the feature-learning signal of a Recursive Feature Machine (the average gradient outer product, AGOP) can be
computed exactly from additive statistics, so one round of feature learning needs only tally marks.

Model: h = relu(A x) (random features), f(x) = W h with W a ridge readout. The AGOP over the data is
    M = mean_x J(x)^T J(x),  J(x) = df/dx = W diag(s(x)) A,  s(x) = 1[A x > 0].
Then M = A^T [ (W^T W) * G ] A  with  G = mean_x s(x) s(x)^T  (co-activation counts; * is elementwise),
and W comes from the additive sums H^T H and H^T Y. Every statistic is a sum over examples, so the result is the same
for any order of the data and for any split into shards that are added afterwards.

  python3 brain/agop_counts_check.py
"""
import numpy as np


def main(n=500, d=20, D=300, c=5, lam=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    Y = np.eye(c)[rng.integers(0, c, n)]
    A = rng.standard_normal((D, d)) / np.sqrt(d)
    Z = X @ A.T
    S, H = (Z > 0).astype(float), np.maximum(Z, 0)

    W = np.linalg.solve(H.T @ H + lam * np.eye(D), H.T @ Y).T
    direct = sum((W @ (S[i][:, None] * A)).T @ (W @ (S[i][:, None] * A)) for i in range(n)) / n
    counts = A.T @ ((W.T @ W) * (S.T @ S / n)) @ A
    print(f"AGOP direct vs from co-activation counts: max difference {np.abs(direct - counts).max():.1e} "
          f"(values up to {np.abs(direct).max():.2f})")

    perm = rng.permutation(n)
    a, b = perm[: n // 3], perm[n // 3:]
    W2 = np.linalg.solve(H[a].T @ H[a] + H[b].T @ H[b] + lam * np.eye(D), H[a].T @ Y[a] + H[b].T @ Y[b]).T
    merged = A.T @ ((W2.T @ W2) * ((S[a].T @ S[a] + S[b].T @ S[b]) / n)) @ A
    print(f"two shards counted separately, then added: max difference {np.abs(merged - counts).max():.1e}")


if __name__ == "__main__":
    main()
