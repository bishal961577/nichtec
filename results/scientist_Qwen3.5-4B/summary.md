# Agent D, the scientist: test 2

Model Qwen3.5-4B, 12 of 12 tasks done. Pre-registration: reports/Learning on the job from first principles 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of (random - decoy first) and (random - decoy last).

| Agent | Random orders | Decoy first | Decoy last | Learning | Order gap | Final rule good |
|---|---|---|---|---|---|---|
| D, scientist | 75.2 +/- 3.4 | 65.0 | 76.0 | 25.2 | 10.2 | 5% |
| D-count (same guesses, judged by count only) | 79.0 +/- 2.3 | 65.4 | 77.3 | 29.0 | 13.6 | 7% |

Comparison on Qwen3.5-27B (order-gap test, random orders): A 83.7, B 81.2. Brute force over 2,578 simple rules with the same judging: 99.4.
Streams whose guesses include a good rule: 7%. Guesses that contradict the cases they were shown: 96% of 1,472 (order-gap test: 61%). Conditions per guess: 3.5 (order-gap test: 4.0). D's late online accuracy (last 20 cases, random orders): 66.1.

| Task | Rule | D random / first / last | D-count random / first / last | Brute force | D's rule (random1) |
|---|---|---|---|---|---|
| rule1_draw1 | `size >= 5 and color != 'red'` | 61 / 88 / 90 | 73 / 88 / 90 | 100 | `size > 6` |
| rule1_draw2 | `size >= 5 and color != 'red'` | 88 / 70 / 80 | 82 / 82 / 80 | 92 | `size >= 5 or size == 9` |
| rule2_draw1 | `shape == 'circle' or material == 'metal'` | 79 / 62 / 70 | 79 / 68 / 70 | 100 | `color == 'blue' or shape == 'circle' or size > 7` |
| rule2_draw2 | `shape == 'circle' or material == 'metal'` | 68 / 62 / 65 | 75 / 75 / 62 | 100 | `color == 'green' or shape == 'circle' or size == 9 or material == 'metal'` |
| rule3_draw1 | `color in ('blue', 'green') and size <= 6` | 83 / 48 / 82 | 88 / 40 / 88 | 100 | `color == 'blue' or (color == 'green' and (shape == 'circle' or shape == 'square' or shape == 'triangle' or shape == 'star')) and size <= 6` |
| rule3_draw2 | `color in ('blue', 'green') and size <= 6` | 72 / 58 / 58 | 76 / 62 / 60 | 100 | `color == 'blue' and size <= 6 or color == 'green'` |
| rule4_draw1 | `material != 'wood' and shape != 'star'` | 61 / 70 / 70 | 62 / 80 / 70 | 100 | `size >= 5` |
| rule4_draw2 | `material != 'wood' and shape != 'star'` | 85 / 60 / 50 | 85 / 60 / 60 | 100 | `shape == 'square' or shape == 'triangle' or shape == 'circle' and material != 'wood'` |
| rule5_draw1 | `size > 3 and shape in ('square', 'triangle')` | 71 / 55 / 100 | 72 / 70 / 100 | 100 | `(color == 'yellow') or (shape == 'square') or (size == 8) or (material == 'wood')` |
| rule5_draw2 | `size > 3 and shape in ('square', 'triangle')` | 88 / 62 / 72 | 88 / 62 / 72 | 100 | `shape == 'square' or shape == 'triangle'` |
| rule6_draw1 | `color == 'yellow' or size >= 7` | 57 / 90 / 75 | 80 / 42 / 75 | 100 | `shape == 'star' or size >= 4` |
| rule6_draw2 | `color == 'yellow' or size >= 7` | 89 / 55 / 100 | 89 / 55 / 100 | 100 | `color == 'yellow' or (shape == 'square' and size >= 7 and material != 'plastic') or (shape != 'square' and size == 7)` |

## Verdict

- STOP: the phone-sized model with the habits scores 75.2 < 83.7, below the 27B model without them.

Check before the run: {'model': 'order-gap-4b', 'replies': ["color == 'yellow' or size > 3", "shape == 'circle' or color == 'blue' or size > 3 or material == 'wood' or material == 'metal' or material == 'glass'"], 'parsed_rules': ["color == 'yellow' or size > 3", "shape == 'circle' or color == 'blue' or size > 3 or material == 'wood' or material == 'metal' or material == 'glass'"], 'writes_rules': True}
