# Can a Brain Be Built From Tally Marks? Step 0: The Record Before Any Experiment (October 2026)

The project built a memory that learns in any order: additive statistics, exact merging, exact deletion. It could
not build the brain, meaning the features or concepts that make a learner sharp. The open target, stated in
`COUNTING_AI.md` §7, is **an additive learning rule that is as sharp as backprop**. This step checks the published
record first, before anything is built.

**Bottom line:**
- **Additive learning tops out around 90–91% on CIFAR-10, against about 95–96% for backprop.** The best additive
  result comes from a fixed deep kernel that learns no features at all. Every method that learns features without
  backprop and without error signals scores lower, about 49–80%.
- **A first-principles argument explains this.** Exact order-invariance with bounded memory forces the learner to
  add up a fixed function of each example and do all its learning at read time. Stacked, nonlinear, learned features
  cannot be updated exactly that way.
- **No paper proves exact order-invariance for features learned from scratch.** The exact results all sit on a
  frozen, backprop-pretrained backbone.
- **One concrete, apparently unpublished opening was found and checked.** The feature-learning signal of Recursive
  Feature Machines (a backprop-free method that learns features) can be computed exactly from Hebbian co-activation
  counts. One round of feature learning then becomes a pure tally-mark operation. The identity holds to 7×10⁻¹⁶
  (`brain/agop_counts_check.py`).
- **The odds are low and stated as such.** This could narrow the 5-point gap on small images. It says nothing yet
  about language, where pure counting is far behind at scale.

## 1. The scoreboard (CIFAR-10, accuracy)

| Method | Additive / order-free? | Learns features? | CIFAR-10 | Source |
|---|---|---|---|---|
| Backprop CNN (SGD) | No | Yes | 95.4–96% | Myrtle (Shankar et al. 2020); DKM paper (2024) |
| Deep kernel machine | No (gradient-optimised Gram matrices) | Yes | 92.7–94.5% | Milsom et al. 2024; arXiv 2410.06171 |
| Forward-only, local error losses (ASGE VGG11; forward-only ResNet-17; DTP) | No (error-driven per layer) | Yes | 89.4–90.6% | arXiv 2509.12394; 2606.09928 |
| **Fixed compositional kernel (Myrtle), exact ridge** | **Yes** (adds up a similarity table; cost grows with n²) | **No** | **90–91.2%** | Shankar et al. 2020 |
| Random features, 256k, ridge | Yes (fixed features, additive statistics) | No | 85.6% | Recht et al. 2019 |
| Self-Contrastive Forward-Forward (unsupervised) | No | Yes | 80.75% | Nature Comms 2025 |
| SoftHebb (Hebbian, no error signal) | Partly (local, not order-free) | Yes | 80.3% | Journé et al. 2022/23 |
| Hebbian winner-take-all CNN | Partly | Yes | 75–76% | arXiv 2501.17266 |
| Forward Projection (closed form, one pass, per-layer ridge) | Largely | Yes | 48.8% vs backprop 54.8% (same small network, no augmentation) | Nature Comms 2026 (arXiv 2501.16476) |
| Recursive Feature Machines (backprop-free feature learning) | Per iteration; needs the data again for each iteration | Yes | Not found (authors report "a sizable gain" over fixed convolutional kernels) | arXiv 2309.00570; NeurIPS 2024 |

Two adjacent lines are related but not the target:
- **Hebbian learning with a global sign signal** (GHL, January 2026) claims near-backprop results, but its global
  signal makes it error-driven.
- **A 2026 review of backprop-free continual learning** uses "local generative dreaming", which is the same loop as
  SPARC's sleep.

**Language.**
- The largest count-based model ever built, infini-gram on 5 trillion tokens, reaches 47% next-token accuracy and is
  useful mainly *alongside* neural models.
- This project's counting model kept pace with a small LSTM up to 10M words. Published results at 1 billion words put
  counting at about twice the perplexity of LSTMs.

**Exact order-invariance already proved:**
- analytic continual learning (ACIL and successors, 2022–2025);
- exact federated ridge heads with proved order and partition invariance (arXiv 2603.12977, March 2026).

Both work on frozen, backprop-pretrained backbones. I found no paper with exact order-invariance for a backbone
learned from scratch.

## 2. First principles: why learned features and order-invariance pull apart

- **The state must be a sum.** A learner whose final state is the same for every order of the same data, and whose
  memory does not grow with the data, must keep a summary that is a sum (more generally, a commutative merge) of a
  fixed function of each example: S = Σ φ(x). Learning is whatever is computed from S at read time. Streaming LDA,
  ridge, PCA, counts and kernel sums are all of this form. This is the classical sufficient-statistic view; nothing
  here is new.
- **One learned layer works.** Its features (PCA, CCA, LDA directions, a kernel predictor) can be computed from S at
  read time.
- **A second learned nonlinear layer breaks it.** Its statistics are sums of φ₂(f₁(x)), where f₁ is the first
  layer's learned map. When f₁ changes as data arrives, those sums are stale and cannot be recomputed from S without
  the data.
- **There are three ways out:**
  - **Approximate.** Under a Gaussian assumption, a ReLU layer's second moments follow in closed form from the
    previous layer's. This is exactly how fixed deep kernels are computed, and it is why they are the strongest
    additive method (90–91%) while learning nothing.
  - **Develop in stages.** Learn layer k from sums during its own period, then freeze it and start layer k+1. Order
    then matters only across stages, like a childhood. SPARC-dev and greedy layer-wise learning (2006–2015) are this,
    and reach about 80%.
  - **Keep the data and replay.** This gives up additivity.
- **So the gap has a name.** It is the value of *feature learning*: about 5 points on CIFAR-10 between the best
  additive method (91%) and backprop (96%). Hebbian rules have not captured that value; they score below fixed kernels.

## 3. The opening found

Recursive Feature Machines learn features without backprop:
- fit a kernel predictor;
- compute the average gradient outer product (AGOP) of the predictor over the data;
- reweight the input by it, and repeat.

The method matches deep networks on some tasks and reproduces their features (edge detectors on CIFAR-10). Its AGOP
is computed by revisiting the data.

**The identity.** For random ReLU features h = relu(Ax) and a ridge readout W:

> M = A^T [ (W^T W) ∘ G ] A,  where G = mean of s s^T, s = 1[Ax > 0]

G is the matrix of **co-activation counts**: how often each pair of feature units fires together, the oldest
Hebbian statistic there is. W itself comes from the additive sums HᵀH and HᵀY. So one round of RFM feature learning
can be computed from tally marks alone:
- in any order;
- merged exactly across shards (difference 2.5×10⁻¹⁴ in the check);
- with exact deletion by subtraction.

For convolutional features with pooling, the same holds with one G per pooling region. Searches found the two pieces
(AGOP feature learning; Hebbian co-activation statistics) but no paper joining them.

**What this does not solve:**
- It gives one round of feature learning per stage. The next round's statistics are about new features and need new
  data, so a multi-stage learner is order-free within each stage only.
- In a class-incremental stream, the first stage's features are shaped by the first classes. That may hurt later
  ones; this is measured below, not assumed.
- How much one RFM round adds on CIFAR-10 is not known to me; the paper's tables were not reachable from here. Step 1
  measures it.

## 4. Step 1, pre-registered (nothing has been run)

**Setup:**
- **Data:** CIFAR-10, 50,000 training and 10,000 test images; 5,000 training images held out to choose the ridge
  strength.
- **Patches:** 5×5×3, whitened with ZCA (itself computed from additive second moments). Every position is used, with
  padding.
- **Features:** D random ReLU filters, sum-pooled over the four image quadrants. The readout is ridge regression on
  one-hot labels. D = 2,048 is the main setting, with 1,024 and 4,096 reported too.
- **Arms:**
  - **B0:** fixed random features.
  - **B1:** one round of AGOP from co-activation counts. Patches are transformed by M^½, then new random filters
    are applied.
  - **B2:** two rounds.
  - **Identity check:** the per-patch AGOP computed directly on 500 images; it must agree with the counts version to
    a relative 10⁻⁶.
  - **Context:** a small backprop CNN trained on the same data, shuffled and in class order.
- **Where:** the laptop, or this machine if a CIFAR-10 mirror is reachable. CPU or GPU, minutes to hours.

**E1. Does feature learning from counts help?** Shuffled data; B1 and B2 statistics recomputed on the same 45,000
images.
- **Pass:** B1 ≥ B0 + 3.0 points at D = 2,048, and B2 ≥ B0 + 5.0.
- **Stop (the program ends):** B1 < B0 + 1.0. Feature learning from counts adds nothing.
- **In between:** report, and do not proceed to E2 without discussing it.

**E2 (only if E1 passes). Order.**
- **Setup:** one pass over a class-incremental stream (5 tasks of 2 classes). The B1 statistics accumulate during
  the first half of the stream, B1's features are used for the second half, and the readout statistics are always
  additive.
- **Pass:** order gap (shuffled minus ordered) ≤ 2 points, with ordered accuracy ≥ B0 + 3.
- **Stop:** gap above 10 points. Supervised stage features then lock in early classes, and stages need unlabeled
  or developmental data instead.

**What would count as a real result:** beating 91.2%, the best published additive method, with learned features and
order-invariance at least within stages. E1 and E2 cannot reach that at D = 2,048. They only decide whether the
route deserves a scaled attempt.

**My guess, low confidence (my previous four predictions in this project were wrong or overstated):**
- B0 roughly 70–78% at D = 2,048;
- B1 − B0 between +1 and +6 points;
- E2's order gap above 2 points, because supervised features learned from the first classes are biased.

## Sources

- Myrtle kernels: [Neural Kernels Without Tangents](https://arxiv.org/abs/2003.02237); deep kernel machines: [arXiv 2410.06171](https://arxiv.org/abs/2410.06171); CNTK: [arXiv 1904.11955](https://arxiv.org/abs/1904.11955)
- Random features, 85.6%: [Recht et al. 2019](http://proceedings.mlr.press/v97/recht19a/recht19a.pdf)
- SoftHebb: [arXiv 2209.11883](https://arxiv.org/abs/2209.11883); Hebbian WTA CNN: [arXiv 2501.17266](https://arxiv.org/abs/2501.17266); GHL: [arXiv 2601.21367](https://arxiv.org/abs/2601.21367)
- Forward-only: [ASGE, arXiv 2509.12394](https://arxiv.org/pdf/2509.12394); [arXiv 2606.09928](https://arxiv.org/pdf/2606.09928); [benchmark, arXiv 2511.01061](https://arxiv.org/pdf/2511.01061); [SCFF, Nature Comms 2025](https://www.nature.com/articles/s41467-025-61037-0)
- Forward Projection: [arXiv 2501.16476](https://arxiv.org/abs/2501.16476), [Nature Comms 2026](https://www.nature.com/articles/s41467-026-69161-1)
- Recursive Feature Machines: [Science 2024 / arXiv 2212.13881](https://arxiv.org/abs/2212.13881); ConvRFM: [arXiv 2309.00570](https://arxiv.org/abs/2309.00570); Deep RFM and neural collapse: [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ec045a5ca2d8cfc528591b4c34296370-Paper-Conference.pdf)
- Exact order- and partition-invariant ridge heads: [arXiv 2603.12977](https://arxiv.org/html/2603.12977); backprop-free continual learning review: [Preprints 2026](https://www.preprints.org/frontend/manuscript/a045d1304523d9af6f2a5de72752f0f4/download_pub)
- Infini-gram: [arXiv 2401.17377](https://arxiv.org/abs/2401.17377)
- This project: [AGI_BOTTLENECK.md](../AGI_BOTTLENECK.md), [COUNTING_AI.md](../COUNTING_AI.md)
