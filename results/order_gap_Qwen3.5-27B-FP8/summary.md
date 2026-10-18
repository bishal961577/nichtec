# Order gap: does the order of experience change what an agent learns?

Model Qwen3.5-27B-FP8, standard rules, 2 of 12 tasks done. Pre-registration: reports/Order gap 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of (random - decoy first) and (random - decoy last). Learning = random-order accuracy - 50.

| Agent | Random orders | Decoy first | Decoy last | Learning | Gap first | Gap last | Order gap | Spread over random orders | Late online |
|---|---|---|---|---|---|---|---|---|---|
| A, history | 82.5 | 82.5 | 85.0 | 32.5 +/- 5.8 | 0.0 +/- 8.3 | -2.5 +/- 5.8 | 0.0 | 7.5 | 62.5 |
| B, ledger | 70.8 | 91.2 | 68.8 | 20.8 +/- 5.0 | -20.4 +/- 8.8 | 2.1 +/- 21.2 | 2.1 | 35.0 | 71.7 |
| C, revise | 64.2 | 61.2 | 60.0 | 14.2 +/- 10.8 | 2.9 +/- 9.6 | 4.2 +/- 13.3 | 4.2 | 32.5 | 69.2 |

A with no history: 57.5. A with the history sorted: 90.0. A at threshold zero instead of the median, random orders: 67.1.
A's answers that were neither yes nor no: 0.0% of 960 calls. Proposals that are valid rules: 82%. Streams whose proposals include a good rule: 10%.
Final rule good / decoy: B 10% / 0%, C 10% / 0%. Decoy kept after a decoy-first stream: B 0%, C 0%.

| Task | Rule | A random / first / last | B random / first / last | C random / first / last |
|---|---|---|---|---|
| rule1_draw1 | `size >= 5 and color != 'red'` | 77 / 85 / 85 | 66 / 95 / 85 | 53 / 60 / 62 |
| rule1_draw2 | `size >= 5 and color != 'red'` | 88 / 80 / 85 | 76 / 88 / 52 | 75 / 62 / 58 |

## Verdict (provisional: not all tasks done)

- Check A learns (learning >= 5): yes (32.5).
- Check the proposer (good rule in >= 25% of streams): no (10%).
- STOP: A's order gap is 0.0 < 5. Order is not what holds back a full-history agent here; the route ends.
- Trivial cure: A with the history in a fixed order scores 90.0 against 82.5 for random orders, so a fixed order removes A's order damage for free.

Check before the run: {'served_models': ['order-gap-27b'], 'model': 'order-gap-27b', 'first_token': 'no', 'answers_yes_no': True, 'margin': -1.5751565291707146, 'top_tokens': [('no', -0.208), ('yes', -1.833), ('Yes', -4.583), ('No', -4.583), ('To', -5.458), ('Let', -6.958)], 'proposals': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"], 'parsed_rules': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"]}
