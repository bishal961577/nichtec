# Native-key test — Qwen/Qwen2.5-1.5B (v2)

2,580 stored test subjects, 8,082 unseen sentences; thresholds from 382 calibration subjects (false-fire allowance 2%). 263 names contain a stored name as part of a longer name.

**Native addressing: not viable as is (hit 0.784, false fire 0.036); needs a learned canonicaliser** (best variant: write sentence->sentence). **Two-hop middle entity: none** (top-1 0.0258 at layer 28, chance 0.00061).

## Native scan (no position given; layer chosen on calibration)

| Stored key | Layer | Hit on unseen sentences | False fire, never-stored names | Fire on longer names containing a stored one |
|---|---|---|---|---|
| alone->sentence | 5 | 0.586 | 0.031 | 0.2969 |
| write sentence->sentence | 5 | 0.784 | 0.036 | 0.5836 |

## Name position given (layer chosen on calibration; best test layer for reference)

| Variant | Calib. layer | Hit | Top-1 | False fire | Best test hit (layer) |
|---|---|---|---|---|---|
| alone->sentence / last | 1 | 0.681 | 0.953 | 0.039 | 0.681 (1) |
| alone->sentence / mean | 1 | 0.982 | 0.999 | 0.023 | 0.982 (1) |
| write sentence->sentence / last | 5 | 0.858 | 0.967 | 0.052 | 0.861 (4) |
| write sentence->sentence / mean | 0 | 0.715 | 0.936 | 0.026 | 0.742 (1) |

## A name inside a sentence vs the same name alone (median cosine, whitened, last token)

| Layer | Same name | Nearest other name |
|---|---|---|
| 0 | 1.000 | 0.869 |
| 1 | 0.994 | 0.726 |
| 2 | 0.976 | 0.648 |
| 3 | 0.982 | 0.625 |
| 4 | 0.977 | 0.592 |
| 5 | 0.963 | 0.571 |
| 6 | 0.954 | 0.567 |
| 7 | 0.939 | 0.582 |
| 8 | 0.898 | 0.570 |
| 9 | 0.888 | 0.568 |
| 10 | 0.873 | 0.562 |
| 11 | 0.860 | 0.552 |
| 12 | 0.848 | 0.533 |
| 13 | 0.835 | 0.523 |
| 14 | 0.830 | 0.518 |
| 15 | 0.814 | 0.516 |
| 16 | 0.805 | 0.514 |
| 17 | 0.789 | 0.504 |
| 18 | 0.758 | 0.494 |
| 19 | 0.726 | 0.481 |
| 20 | 0.733 | 0.471 |
| 21 | 0.693 | 0.459 |
| 22 | 0.667 | 0.468 |
| 23 | 0.607 | 0.474 |
| 24 | 0.567 | 0.490 |
| 25 | 0.515 | 0.495 |
| 26 | 0.466 | 0.493 |
| 27 | 0.440 | 0.478 |
| 28 | 0.407 | 0.471 |

## Two-hop questions: decoding the hidden middle entity (3,405 test questions, 1,631 candidate names, first hop known 0.6264)

| Layer | Middle entity top-1 | ...first hop known | Named first subject top-1 (control) |
|---|---|---|---|
| 0 | 0.001 | 0.0 | 0.001 |
| 1 | 0.004 | 0.0042 | 0.852 |
| 2 | 0.004 | 0.0042 | 0.198 |
| 3 | 0.003 | 0.0047 | 0.151 |
| 4 | 0.004 | 0.0042 | 0.099 |
| 5 | 0.004 | 0.0042 | 0.077 |
| 6 | 0.013 | 0.0141 | 0.074 |
| 7 | 0.006 | 0.0066 | 0.075 |
| 8 | 0.009 | 0.007 | 0.119 |
| 9 | 0.022 | 0.0239 | 0.101 |
| 10 | 0.025 | 0.0281 | 0.096 |
| 11 | 0.022 | 0.022 | 0.082 |
| 12 | 0.020 | 0.0234 | 0.056 |
| 13 | 0.017 | 0.0197 | 0.069 |
| 14 | 0.020 | 0.0239 | 0.076 |
| 15 | 0.021 | 0.022 | 0.064 |
| 16 | 0.021 | 0.0248 | 0.066 |
| 17 | 0.022 | 0.0244 | 0.064 |
| 18 | 0.018 | 0.0183 | 0.056 |
| 19 | 0.014 | 0.0141 | 0.062 |
| 20 | 0.018 | 0.0197 | 0.046 |
| 21 | 0.019 | 0.0202 | 0.040 |
| 22 | 0.022 | 0.0248 | 0.110 |
| 23 | 0.025 | 0.0281 | 0.276 |
| 24 | 0.019 | 0.0197 | 0.336 |
| 25 | 0.034 | 0.0319 | 0.330 |
| 26 | 0.041 | 0.0398 | 0.271 |
| 27 | 0.024 | 0.0248 | 0.205 |
| 28 | 0.024 | 0.0258 | 0.155 |

Total time: 31.4 min
