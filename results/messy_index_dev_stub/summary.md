# Messy chat index test: dev (seed 101)

Pre-registration: reports/Messy chat index test 2026.md (with amendment 1). Accuracy in %; hard set = role, nick, fname_ctx, desc, who, two, upd, retract, deleted, event_last, event_what, event_who

| Kind | n | pages think | B think | oracle think | nolinks think | C think | Ccues think | pages off | B off | C off | Ccues off | Bw think | Bw off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amb | 30 | 100.0 | 96.7 | 100.0 | 96.7 | 30.0 | 80.0 | 100.0 | 96.7 | 30.0 | 80.0 | 93.3 | 93.3 |
| deleted | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| desc | 150 | 100.0 | 80.7 | 86.7 | 80.7 | 2.0 | 1.3 | 100.0 | 80.7 | 2.0 | 1.3 | 81.3 | 81.3 |
| event_last | 170 | 100.0 | 95.9 | 98.8 | 95.9 | 84.7 | 84.1 | 100.0 | 95.9 | 84.7 | 84.1 | 97.1 | 97.1 |
| event_what | 166 | 100.0 | 91.6 | 96.4 | 91.6 | 85.5 | 85.5 | 100.0 | 91.6 | 85.5 | 85.5 | 94.6 | 94.6 |
| event_who | 74 | 100.0 | 91.9 | 97.3 | 91.9 | 100.0 | 98.6 | 100.0 | 91.9 | 100.0 | 98.6 | 93.2 | 93.2 |
| fname_ctx | 91 | 100.0 | 80.2 | 81.3 | 80.2 | 75.8 | 75.8 | 100.0 | 80.2 | 75.8 | 75.8 | 75.8 | 75.8 |
| named | 150 | 100.0 | 98.7 | 100.0 | 98.7 | 80.0 | 78.7 | 100.0 | 98.7 | 80.0 | 78.7 | 97.3 | 97.3 |
| never | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| nick | 119 | 100.0 | 92.4 | 100.0 | 92.4 | 21.0 | 15.1 | 100.0 | 92.4 | 21.0 | 15.1 | 91.6 | 91.6 |
| retract | 114 | 100.0 | 97.4 | 100.0 | 97.4 | 55.3 | 50.9 | 100.0 | 97.4 | 55.3 | 50.9 | 93.9 | 93.9 |
| role | 50 | 100.0 | 92.0 | 100.0 | 92.0 | 40.0 | 34.0 | 100.0 | 92.0 | 40.0 | 34.0 | 86.0 | 86.0 |
| two | 180 | 100.0 | 92.8 | 100.0 | 3.3 | 7.8 | 4.4 | 100.0 | 92.8 | 7.8 | 4.4 | 90.6 | 90.6 |
| upd | 180 | 100.0 | 97.8 | 100.0 | 97.8 | 62.8 | 58.9 | 100.0 | 97.8 | 62.8 | 58.9 | 98.9 | 98.9 |
| who | 150 | 100.0 | 91.3 | 96.0 | 91.3 | 61.3 | 53.3 | 100.0 | 91.3 | 61.3 | 53.3 | 92.0 | 92.0 |
| HARD | 1544 | 100.0 | 92.2 | 96.6 | 81.8 | 55.6 | 52.8 | 100.0 | 92.2 | 55.6 | 52.8 | 92.0 | 92.0 |
| ASSOC | 480 | 100.0 | 88.5 | 94.6 | 55.0 | 22.7 | 18.8 | 100.0 | 88.5 | 22.7 | 18.8 | 88.1 | 88.1 |
| EVENTS | 410 | 100.0 | 93.4 | 97.6 | 93.4 | 87.8 | 87.3 | 100.0 | 93.4 | 87.8 | 87.3 | 95.4 | 95.4 |
| held-out | 0 | - | - | - | - | - | - | - | - | - | - | - | - |

## Verdict

- **BETWEEN: neither line met**
- B hard 92.2% (>= 85.0); oracle 96.6% (read-limited below 85.0); B/oracle 0.96 (>= 0.9); desc 80.7%, who 91.3%, two 92.8% (each >= 75.0)
- Fast path: B with thinking off 92.2% on the hard set (>= 80.0)
- C: hard 55.6%, desc, who, two 22.7% (desc 2.0, who 61.3, two 7.8); B minus C on them 65.8 points [60.7, 70.7] (people resampled, n = 480) <- the stronger search, used for the margin
- Ccues: hard 52.8%, desc, who, two 18.8% (desc 1.3, who 53.3, two 4.4); B minus Ccues on them 69.8 points [64.9, 74.5] (people resampled, n = 480)
- Filing: wrong merges 0.91% (<= 1.0; writer-limited above 3.0); hard cases 0.60% of 335 (<= 5.0; writer-limited above 10.0); facts filed 93.8%; filed right by mention style {"fname": 99.5, "fname_ctx": 99.6, "full": 99.2, "nick": 100.0, "old_role": 100.0, "pronoun": 99.5, "pronoun_prev": 98.2, "role": 100.0, "via": 94.4} (n {"fname": 367, "fname_ctx": 223, "full": 744, "nick": 42, "old_role": 1, "pronoun": 184, "pronoun_prev": 112, "role": 112, "via": 107}); link precision 86.4% of 784; lure intrusion 0.0%; 303 index people for 260, 8 split; 42 rows about me
- Forgetting a whole person: 98.0% of their messages removed (>= 98.0); worst person 88.5%; collateral 0.3% of removed
- Deletion: rows elsewhere still carrying a deleted value 0; deleted values shown to a reader 0; answers naming a deleted value (thinking on) pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 0.0%, Ccues 0.0%, Bw 0.0%; guessing rate on never-stated facts pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 0.0%, Ccues 0.0%, Bw 0.0%
- Writing cost: 11 completion and 98 prompt tokens per message (<= 150 and <= 500)
- Events (reported line): B 93.4% (>= 70.0)
- Conditions for PASS: {"B_hard": true, "oracle_ratio": true, "assoc_each": true, "margin": true, "merges": true, "hard_merges": true, "fast_path": true, "write_cost": true, "forget": false}
- B: found every needed message for 85.5% of questions; median completion 7 tokens; cut off 0.0%
- oracle: found every needed message for 91.8% of questions; median completion 7 tokens; cut off 0.0%
- nolinks: found every needed message for 76.6% of questions; median completion 8 tokens; cut off 0.0%
- C: found every needed message for 44.2% of questions; median completion 10 tokens; cut off 0.0%
- Ccues: found every needed message for 45.3% of questions; median completion 8 tokens; cut off 0.0%
- Thinking writer: B-w hard 92.0% (off 92.0%); merges 0.91%, hard 0.87%; forgetting 97.9%; 11 completion tokens per message
