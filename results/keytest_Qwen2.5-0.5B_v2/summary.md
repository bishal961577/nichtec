# Native-key test — Qwen/Qwen2.5-0.5B (v2)

2,580 stored test subjects, 8,082 unseen sentences; thresholds from 382 calibration subjects (false-fire allowance 2%). 263 names contain a stored name as part of a longer name.

**Native addressing: closed** (best variant: write sentence->sentence). **Two-hop middle entity: none** (top-1 0.0194 at layer 24, chance 0.00061).

## Native scan (no position given; layer chosen on calibration)

| Stored key | Layer | Hit on unseen sentences | False fire, never-stored names | Fire on longer names containing a stored one |
|---|---|---|---|---|
| alone->sentence | 2 | 0.569 | 0.062 | 0.3549 |
| write sentence->sentence | 3 | 0.628 | 0.029 | 0.5392 |

## Name position given (layer chosen on calibration; best test layer for reference)

| Variant | Calib. layer | Hit | Top-1 | False fire | Best test hit (layer) |
|---|---|---|---|---|---|
| alone->sentence / last | 2 | 0.592 | 0.902 | 0.044 | 0.592 (2) |
| alone->sentence / mean | 1 | 0.993 | 0.999 | 0.031 | 0.993 (1) |
| write sentence->sentence / last | 3 | 0.777 | 0.945 | 0.050 | 0.777 (3) |
| write sentence->sentence / mean | 1 | 0.899 | 0.986 | 0.036 | 0.899 (1) |

## A name inside a sentence vs the same name alone (median cosine, whitened, last token)

| Layer | Same name | Nearest other name |
|---|---|---|
| 0 | 1.000 | 0.819 |
| 1 | 0.996 | 0.771 |
| 2 | 0.987 | 0.705 |
| 3 | 0.970 | 0.579 |
| 4 | 0.944 | 0.558 |
| 5 | 0.914 | 0.560 |
| 6 | 0.907 | 0.577 |
| 7 | 0.892 | 0.554 |
| 8 | 0.882 | 0.543 |
| 9 | 0.869 | 0.520 |
| 10 | 0.846 | 0.516 |
| 11 | 0.835 | 0.507 |
| 12 | 0.816 | 0.506 |
| 13 | 0.810 | 0.502 |
| 14 | 0.783 | 0.494 |
| 15 | 0.737 | 0.482 |
| 16 | 0.716 | 0.487 |
| 17 | 0.720 | 0.496 |
| 18 | 0.659 | 0.479 |
| 19 | 0.608 | 0.469 |
| 20 | 0.568 | 0.455 |
| 21 | 0.531 | 0.497 |
| 22 | 0.496 | 0.509 |
| 23 | 0.468 | 0.495 |
| 24 | 0.420 | 0.481 |

## Two-hop questions: decoding the hidden middle entity (3,405 test questions, 1,631 candidate names, first hop known 0.3181)

| Layer | Middle entity top-1 | ...first hop known | Named first subject top-1 (control) |
|---|---|---|---|
| 0 | 0.000 | 0.0 | 0.000 |
| 1 | 0.002 | 0.0018 | 0.188 |
| 2 | 0.004 | 0.0037 | 0.197 |
| 3 | 0.006 | 0.0092 | 0.104 |
| 4 | 0.008 | 0.0111 | 0.082 |
| 5 | 0.012 | 0.0203 | 0.141 |
| 6 | 0.014 | 0.0212 | 0.116 |
| 7 | 0.009 | 0.0102 | 0.095 |
| 8 | 0.011 | 0.0148 | 0.069 |
| 9 | 0.017 | 0.0259 | 0.156 |
| 10 | 0.015 | 0.0203 | 0.127 |
| 11 | 0.016 | 0.0249 | 0.125 |
| 12 | 0.017 | 0.0185 | 0.096 |
| 13 | 0.015 | 0.0222 | 0.067 |
| 14 | 0.016 | 0.0231 | 0.071 |
| 15 | 0.018 | 0.0212 | 0.103 |
| 16 | 0.021 | 0.0259 | 0.120 |
| 17 | 0.015 | 0.0222 | 0.057 |
| 18 | 0.016 | 0.0277 | 0.086 |
| 19 | 0.012 | 0.0194 | 0.075 |
| 20 | 0.015 | 0.0249 | 0.076 |
| 21 | 0.010 | 0.0148 | 0.438 |
| 22 | 0.012 | 0.0203 | 0.365 |
| 23 | 0.016 | 0.0268 | 0.494 |
| 24 | 0.012 | 0.0194 | 0.195 |

Total time: 29.0 min
