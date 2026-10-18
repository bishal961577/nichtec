# Order gap: does the order of experience change what an agent learns?

Model Qwen3.5-27B-FP8, standard rules, 12 of 12 tasks done. Pre-registration: reports/Order gap 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of (random - decoy first) and (random - decoy last). Learning = random-order accuracy - 50.

| Agent | Random orders | Decoy first | Decoy last | Learning | Gap first | Gap last | Order gap | Spread over random orders | Late online |
|---|---|---|---|---|---|---|---|---|---|
| A, history | 83.7 | 81.2 | 82.5 | 33.7 +/- 1.8 | 2.5 +/- 1.7 | 1.2 +/- 1.7 | 2.5 | 8.8 | 65.8 |
| B, ledger | 81.2 | 86.2 | 81.2 | 31.2 +/- 2.7 | -5.1 +/- 3.3 | -0.1 +/- 4.2 | -0.1 | 16.7 | 75.7 |
| C, revise | 66.8 | 64.8 | 58.1 | 16.8 +/- 3.4 | 2.0 +/- 5.0 | 8.7 +/- 4.5 | 8.7 | 21.9 | 64.9 |

A with no history: 52.1. A with the history sorted: 86.2. A at threshold zero instead of the median, random orders: 72.6.
A's answers that were neither yes nor no: 0.0% of 5,760 calls. Proposals that are valid rules: 85%. Streams whose proposals include a good rule: 12%.
Final rule good / decoy: B 12% / 0%, C 7% / 0%. Decoy kept after a decoy-first stream: B 0%, C 0%.

| Task | Rule | A random / first / last | B random / first / last | C random / first / last |
|---|---|---|---|---|
| rule1_draw1 | `size >= 5 and color != 'red'` | 77 / 85 / 85 | 66 / 95 / 85 | 53 / 60 / 62 |
| rule1_draw2 | `size >= 5 and color != 'red'` | 88 / 80 / 85 | 76 / 88 / 52 | 75 / 62 / 58 |
| rule2_draw1 | `shape == 'circle' or material == 'metal'` | 80 / 80 / 80 | 100 / 92 / 92 | 96 / 75 / 48 |
| rule2_draw2 | `shape == 'circle' or material == 'metal'` | 73 / 65 / 65 | 97 / 92 / 68 | 65 / 92 / 52 |
| rule3_draw1 | `color in ('blue', 'green') and size <= 6` | 83 / 85 / 85 | 79 / 82 / 80 | 58 / 50 / 48 |
| rule3_draw2 | `color in ('blue', 'green') and size <= 6` | 85 / 90 / 85 | 76 / 88 / 82 | 61 / 58 / 58 |
| rule4_draw1 | `material != 'wood' and shape != 'star'` | 97 / 90 / 85 | 82 / 70 / 95 | 72 / 62 / 78 |
| rule4_draw2 | `material != 'wood' and shape != 'star'` | 87 / 90 / 85 | 78 / 85 / 70 | 65 / 68 / 60 |
| rule5_draw1 | `size > 3 and shape in ('square', 'triangle')` | 87 / 80 / 80 | 81 / 95 / 88 | 61 / 95 / 50 |
| rule5_draw2 | `size > 3 and shape in ('square', 'triangle')` | 82 / 75 / 80 | 83 / 82 / 88 | 78 / 62 / 60 |
| rule6_draw1 | `color == 'yellow' or size >= 7` | 78 / 70 / 85 | 79 / 78 / 90 | 56 / 42 / 58 |
| rule6_draw2 | `color == 'yellow' or size >= 7` | 88 / 85 / 90 | 78 / 88 / 85 | 62 / 50 / 68 |

## Verdict

- Check A learns (learning >= 5): yes (33.7).
- Check the proposer (good rule in >= 25% of streams): no (12%).
- STOP: A's order gap is 2.5 < 5. Order is not what holds back a full-history agent here; the route ends.
- Trivial cure: A with the history in a fixed order scores 86.2 against 83.7 for random orders, so a fixed order removes A's order damage for free.

Check before the run: {'served_models': ['order-gap-27b'], 'model': 'order-gap-27b', 'first_token': 'no', 'answers_yes_no': True, 'margin': -1.5751565291707146, 'top_tokens': [('no', -0.208), ('yes', -1.833), ('Yes', -4.583), ('No', -4.583), ('To', -5.458), ('Let', -6.958)], 'proposals': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"], 'parsed_rules': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"]}
