# A shared starting point plus a personal difference

Pre-registration: section 9 of reports/Real data test 2026.md. 300 MovieLens-1M people, 80 ratings learned, next 40 tested; mean per-person AUC (0.5 = guessing). No model calls: saved answers only.

Sanity check: E rebuilt from the saved 4B answers reproduces the 4B run for all 300 people (within 0.001).

| Agent | Mean AUC | Minus P [95% interval] |
|---|---|---|
| P, shared list (popularity) | 0.679 | — |
| P+E, personal questions learned on top of P (4B) | 0.666 | -0.012 [-0.021, -0.004] |
| P+genres, dataset tags on top of P (no model) | 0.679 | +0.000 [-0.008, +0.009] |
| P+A, P's rank + history-reading rank (4B) | 0.672 | -0.007 [-0.015, +0.000] |
| P+A, P's rank + history-reading rank (27B) | 0.687 | +0.008 [-0.001, +0.017] |
| P+W, P's rank + taste-in-words rank (4B) | 0.659 | -0.020 [-0.030, -0.010] |
| P+W, P's rank + taste-in-words rank (27B) | 0.677 | -0.002 [-0.012, +0.009] |
| Collaborative filtering, everyone's data (reference) | 0.740 | +0.061 [+0.051, +0.072] |

P+E minus P+genres: -0.013 [-0.022, -0.003].
P+E kept popularity in its model for 289 of 300 people; mean personal questions with weight: 3.8.

## Verdict

- Main (P+E against P): STOP: -0.012 [-0.021, -0.004].
- Secondary, P+A (4B) against P: STOP: -0.007 [-0.015, +0.000].
- Secondary, P+A (27B) against P: STOP: +0.008 [-0.001, +0.017].
