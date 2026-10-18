# Agent D, the scientist: test 1

Model Qwen3.5-27B-FP8, 12 of 12 tasks done. Pre-registration: reports/Learning on the job from first principles 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of (random - decoy first) and (random - decoy last).

| Agent | Random orders | Decoy first | Decoy last | Learning | Order gap | Final rule good |
|---|---|---|---|---|---|---|
| D, scientist | 89.3 +/- 3.0 | 91.7 | 91.5 | 39.3 | -2.2 | 43% |
| D-count (same guesses, judged by count only) | 92.4 +/- 1.5 | 91.7 | 94.2 | 42.4 | 0.8 | 43% |

Comparison on Qwen3.5-27B (order-gap test, random orders): A 83.7, B 81.2. Brute force over 2,578 simple rules with the same judging: 99.4.
Streams whose guesses include a good rule: 43%. Guesses that contradict the cases they were shown: 37% of 1,540 (order-gap test: 61%). Conditions per guess: 3.9 (order-gap test: 4.0). D's late online accuracy (last 20 cases, random orders): 80.3.

| Task | Rule | D random / first / last | D-count random / first / last | Brute force | D's rule (random1) |
|---|---|---|---|---|---|
| rule1_draw1 | `size >= 5 and color != 'red'` | 86 / 80 / 100 | 88 / 62 / 100 | 100 | `color != 'red' and size >= 5` |
| rule1_draw2 | `size >= 5 and color != 'red'` | 90 / 100 / 92 | 90 / 100 / 92 | 92 | `color != 'red' and size > 3` |
| rule2_draw1 | `shape == 'circle' or material == 'metal'` | 90 / 62 / 100 | 92 / 72 / 100 | 100 | `shape == 'circle' or material == 'metal'` |
| rule2_draw2 | `shape == 'circle' or material == 'metal'` | 100 / 100 / 100 | 100 / 100 / 100 | 100 | `shape == 'circle' or material == 'metal'` |
| rule3_draw1 | `color in ('blue', 'green') and size <= 6` | 95 / 98 / 85 | 95 / 98 / 92 | 100 | `color in ['blue', 'green'] and size not in [8, 9]` |
| rule3_draw2 | `color in ('blue', 'green') and size <= 6` | 71 / 95 / 58 | 84 / 95 / 82 | 100 | `color == 'blue' and not (shape == 'square' and size == 9) or (color == 'green' and (shape == 'triangle' or shape == 'star'))` |
| rule4_draw1 | `material != 'wood' and shape != 'star'` | 88 / 78 / 95 | 89 / 85 / 92 | 100 | `shape in ('square', 'triangle') or (shape == 'circle' and material in ('plastic', 'glass'))` |
| rule4_draw2 | `material != 'wood' and shape != 'star'` | 68 / 90 / 100 | 87 / 90 / 100 | 100 | `material != 'wood' and (size % 2 == 1 or color == 'red')` |
| rule5_draw1 | `size > 3 and shape in ('square', 'triangle')` | 98 / 100 / 100 | 98 / 100 / 100 | 100 | `shape in ('square', 'triangle') and size > 3` |
| rule5_draw2 | `size > 3 and shape in ('square', 'triangle')` | 100 / 98 / 98 | 100 / 98 / 98 | 100 | `shape in ['square', 'triangle'] and size >= 4` |
| rule6_draw1 | `color == 'yellow' or size >= 7` | 92 / 100 / 92 | 92 / 100 / 92 | 100 | `shape == 'star' and color != 'red' or size >= 9 or color == 'yellow'` |
| rule6_draw2 | `color == 'yellow' or size >= 7` | 93 / 100 / 78 | 94 / 100 / 80 | 100 | `color == 'yellow' or size >= 7` |

## Verdict

- STOP: D scores 89.3 < 90. The loop does not fix induction; next is a diagnosis of the guesses, not a variant.

Check before the run: {'model': 'order-gap-27b', 'replies': ["size == 5 and material == 'wood'", "shape == 'square' and size == 5 and color == 'yellow'"], 'parsed_rules': ["size == 5 and material == 'wood'", "shape == 'square' and size == 5 and color == 'yellow'"], 'writes_rules': True}
