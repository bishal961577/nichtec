# Ablations and design search (Split-MNIST, seed 0, single run each)

These are the exploratory runs that led to the final SPARC settings. **They were selected on the
MNIST test set** (there is no separate validation split), so the MNIST numbers for SPARC carry
some selection bias. Fashion-MNIST was never used for tuning; its results in `summary.md` are
the out-of-sample check.

Readouts: `assoc` = Hebbian count readout over a k-WTA code (m=20000, k=1000, Parzen vote unless
noted), `slda` = streaming LDA, `ncm` = nearest class mean. Accuracy (%) on all 10 classes after
one pass over the class-ordered stream (0/1 -> 2/3 -> 4/5 -> 6/7 -> 8/9).

## A. Where does forgetting come from when the FEATURE layer also learns from the stream?

| Feature layer (all local, unsupervised, online) | assoc | slda | ncm | units |
|---|---|---|---|---|
| soft (ReLU) encoding, no consolidation | 50.2 | 22.4 | 53.6 | 238 |
| same, **shuffled stream** (reference) | 88.3 | 98.5 | 89.4 | 240 |
| soft encoding, units freeze after 200 wins | 53.4 | 19.8 | 58.9 | 224 |
| **winner-take-all** encoding, no consolidation | 92.5 | 81.2 | 92.0 | 238 |
| WTA + freeze after 200 wins | 93.1 | 90.1 | 93.1 | 224 |
| same, **shuffled stream** (reference) | 91.5 | 98.6 | 93.1 | 222 |
| WTA + freeze after 50 wins | 91.6 | 72.8 | 92.2 | 209 |
| WTA + freeze 200, encoding threshold 0.2 | 93.1 | 90.0 | 93.3 | 224 |
| WTA + freeze 200, recruit threshold 0.4 | 92.6 | 85.9 | 92.2 | 115 |
| **WTA + freeze 200, recruit threshold 0.6** (final; capacity fills during task 1) | 92.6 | **97.2** | 93.0 | 256 |
| WTA + freeze 200, recruit 0.6, **512 units** (keeps recruiting in later tasks) | 92.4 | 78.5 | 90.6 | 512 |
| WTA + freeze 200, 4x4 pooling grid | 90.6 | 92.5 | 93.0 | 224 |
| soft encoding, dictionary learned on task 1 only, then frozen | 84.1 | 84.9 | 87.9 | 126 |
| soft encoding, dictionary learned **without labels on Fashion-MNIST**, then frozen | 88.4 | **98.67** | 89.45 | 256 |
| same, shuffled stream | 88.4 | 98.67 | 89.45 | 256 |

What this says:
1. **Representation drift alone destroys old knowledge**, even with no labels and purely local
   updates: SLDA falls from 98.5 (shuffled) to 22.4 (ordered) when the features underneath it
   keep changing. The readout is commutative, but its inputs are not stable.
2. **Soft codes cause the drift; winner-take-all codes mostly stop it.** When every unit responds
   to every patch, a newly recruited unit changes the code of *old* images. With WTA, a new unit
   only changes codes for patches it matches better than any existing unit.
3. **Growth versus stability is a real tension.** The linear readout (SLDA) is hurt whenever new
   feature units appear after old classes were stored (78.5 with 512 units that keep recruiting,
   97.2 when capacity is filled early). The Hebbian readout barely notices (92.4 vs 92.6).
4. With a frozen "developmental" feature layer learned without labels from *different* data,
   ordered and shuffled streams give **identical** results to 4 decimals.

## B. Same code, same locality - only the update rule changes (frozen developmental features)

| Readout on identical k-WTA codes (only active units' synapses change) | ordered | shuffled |
|---|---|---|
| Hebbian counts, Parzen vote, m=20000 k=1000 | 88.66 | 88.66 |
| Hebbian counts, normalised vote, m=20000 k=200 | 89.94 | 89.94 |
| Hebbian counts, normalised vote, m=40000 k=400 | 90.80 | 90.80 |
| Hebbian counts, normalised vote, m=40000 k=100 (final) | 91.63 | 91.63 |
| Error-driven delta rule, m=20000 k=200, lr 0.5 | **21.56** | 98.11 |
| Error-driven delta rule, m=20000 k=200, lr 2.0 | **21.88** | 98.54 |

Sparsity and locality are identical in every row. The error-driven rule is far more accurate on
shuffled data and forgets catastrophically on ordered data; the counting rule is exactly
order-invariant. The variable that matters is whether updates commute.
