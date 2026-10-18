# Order gap: does the order of experience change what an agent learns?

Model Qwen3.5-27B-FP8, standard rules, 1 of 12 tasks done. Pre-registration: reports/Order gap 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of (random - decoy first) and (random - decoy last). Learning = random-order accuracy - 50.

| Agent | Random orders | Decoy first | Decoy last | Learning | Gap first | Gap last | Order gap | Spread over random orders | Late online |
|---|---|---|---|---|---|---|---|---|---|
| A, history | 76.7 | 85.0 | 85.0 | 26.7 +/- nan | -8.3 +/- nan | -8.3 +/- nan | -8.3 | 5.0 | 65.0 |
| B, ledger | 65.8 | 95.0 | 85.0 | 15.8 +/- nan | -29.2 +/- nan | -19.2 +/- nan | -19.2 | 20.0 | 66.7 |
| C, revise | 53.3 | 60.0 | 62.5 | 3.3 +/- nan | -6.7 +/- nan | -9.2 +/- nan | -6.7 | 15.0 | 61.7 |

A with no history: 60.0. A with the history sorted: 85.0. A at threshold zero instead of the median, random orders: 73.3.
A's answers that were neither yes nor no: 0.0% of 480 calls. Proposals that are valid rules: 84%. Streams whose proposals include a good rule: 0%.
Final rule good / decoy: B 0% / 0%, C 0% / 0%. Decoy kept after a decoy-first stream: B 0%, C 0%.

| Task | Rule | A random / first / last | B random / first / last | C random / first / last |
|---|---|---|---|---|
| rule1_draw1 | `size >= 5 and color != 'red'` | 77 / 85 / 85 | 66 / 95 / 85 | 53 / 60 / 62 |

## Verdict (provisional: not all tasks done)

- Check A learns (learning >= 5): yes (26.7).
- Check the proposer (good rule in >= 25% of streams): no (0%).
- STOP: A's order gap is -8.3 < 5. Order is not what holds back a full-history agent here; the route ends.
- Trivial cure: A with the history in a fixed order scores 85.0 against 76.7 for random orders, so a fixed order removes A's order damage for free.

Check before the run: {'served_models': ['order-gap-27b'], 'model': 'order-gap-27b', 'first_token': 'no', 'answers_yes_no': True, 'margin': -1.5751565291707146, 'top_tokens': [('no', -0.208), ('yes', -1.833), ('Yes', -4.583), ('No', -4.583), ('To', -5.458), ('Let', -6.958)], 'proposals': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"], 'parsed_rules': ["shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and material == 'wood' and size == 5", "shape == 'circle' and material == 'metal' and size > 5 or shape == 'square' and color == 'yellow' and size == 5 and material == 'wood'"]}
