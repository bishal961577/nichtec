# Messy chat index test: dev (seed 101)

Pre-registration: reports/Messy chat index test 2026.md (with amendments 1 and 2). Accuracy in %; hard set = role, nick, fname_ctx, desc, who, two, upd, retract, deleted, event_last, event_what, event_who

| Kind | n | pages think | B think | oracle think | nolinks think | C think | Ccues think | pages off | B off | C off | Ccues off | Bw think | Bw off |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| amb | 30 | 100.0 | 63.3 | 100.0 | 63.3 | 30.0 | 80.0 | 100.0 | 63.3 | 30.0 | 80.0 | 66.7 | 66.7 |
| deleted | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| desc | 150 | 100.0 | 63.3 | 86.7 | 63.3 | 2.0 | 1.3 | 100.0 | 63.3 | 2.0 | 1.3 | 58.0 | 58.0 |
| event_last | 170 | 100.0 | 87.1 | 98.8 | 87.1 | 84.7 | 84.1 | 100.0 | 87.1 | 84.7 | 84.1 | 85.3 | 85.3 |
| event_what | 166 | 100.0 | 84.9 | 96.4 | 84.9 | 85.5 | 85.5 | 100.0 | 84.9 | 85.5 | 85.5 | 83.7 | 83.7 |
| event_who | 74 | 100.0 | 94.6 | 97.3 | 94.6 | 100.0 | 98.6 | 100.0 | 94.6 | 100.0 | 98.6 | 97.3 | 97.3 |
| fname_ctx | 91 | 100.0 | 75.8 | 86.8 | 75.8 | 75.8 | 75.8 | 100.0 | 75.8 | 75.8 | 75.8 | 81.3 | 81.3 |
| named | 150 | 100.0 | 88.7 | 100.0 | 88.7 | 80.0 | 78.7 | 100.0 | 88.7 | 80.0 | 78.7 | 88.7 | 88.7 |
| never | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| nick | 119 | 100.0 | 72.3 | 100.0 | 72.3 | 21.0 | 15.1 | 100.0 | 72.3 | 21.0 | 15.1 | 77.3 | 77.3 |
| retract | 114 | 100.0 | 80.7 | 100.0 | 80.7 | 55.3 | 50.9 | 100.0 | 80.7 | 55.3 | 50.9 | 78.1 | 78.1 |
| role | 50 | 100.0 | 86.0 | 100.0 | 86.0 | 40.0 | 34.0 | 100.0 | 86.0 | 40.0 | 34.0 | 90.0 | 90.0 |
| two | 180 | 100.0 | 63.3 | 100.0 | 2.8 | 7.8 | 4.4 | 100.0 | 63.3 | 7.8 | 4.4 | 58.9 | 58.9 |
| upd | 180 | 100.0 | 73.3 | 100.0 | 73.3 | 62.8 | 58.9 | 100.0 | 73.3 | 62.8 | 58.9 | 80.0 | 80.0 |
| who | 150 | 100.0 | 75.3 | 96.0 | 75.3 | 61.3 | 53.3 | 100.0 | 75.3 | 61.3 | 53.3 | 75.3 | 75.3 |
| HARD | 1544 | 100.0 | 77.9 | 96.9 | 70.9 | 55.6 | 52.8 | 100.0 | 77.9 | 55.6 | 52.8 | 78.1 | 78.1 |
| ASSOC | 480 | 100.0 | 67.1 | 94.6 | 44.4 | 22.7 | 18.8 | 100.0 | 67.1 | 22.7 | 18.8 | 63.7 | 63.7 |
| EVENTS | 410 | 100.0 | 87.6 | 97.6 | 87.6 | 87.8 | 87.3 | 100.0 | 87.6 | 87.8 | 87.3 | 86.8 | 86.8 |
| held-out | 0 | - | - | - | - | - | - | - | - | - | - | - | - |

## Verdict

- **WRITER-LIMITED: the reading works on a perfect index, the 4B's filing does not; next is the thinking writer (pre-registered), a bigger writer only if that fails too**
- B hard 77.9% (>= 85.0); oracle 96.9% (read-limited below 85.0); B/oracle 0.80 (>= 0.9); desc 63.3%, who 75.3%, two 63.3% (each >= 75.0)
- Fast path: B with thinking off 77.9% on the hard set (>= 80.0)
- C: hard 55.6%, desc, who, two 22.7% (desc 2.0, who 61.3, two 7.8); B minus C on them 44.4 points [38.2, 50.2] (people resampled, n = 480) <- the stronger search, used for the margin
- Ccues: hard 52.8%, desc, who, two 18.8% (desc 1.3, who 53.3, two 4.4); B minus Ccues on them 48.3 points [42.3, 54.0] (people resampled, n = 480)
- Filing: wrong merges 3.47% (<= 1.0; writer-limited above 3.0); hard cases 10.37% of 482 (<= 5.0; writer-limited above 10.0); facts filed 91.4%; filed right by mention style {"fname": 97.1, "fname_ctx": 93.2, "full": 98.8, "nick": 100.0, "old_role": 100.0, "pronoun": 98.3, "pronoun_prev": 98.3, "role": 99.1, "via": 77.8} (n {"fname": 373, "fname_ctx": 219, "full": 739, "nick": 43, "old_role": 2, "pronoun": 181, "pronoun_prev": 115, "role": 111, "via": 108}); link precision 93.5% of 677; lure intrusion 0.3%; 707 index people for 260, 109 split; 49 rows about me
- Forgetting a whole person: 93.0% of their messages removed on average; worst person 54.5% (every person >= 95.0); collateral 0.5% of removed
- Deletion: rows elsewhere still carrying a deleted value 0; deleted values shown to a reader 0; answers naming a deleted value (thinking on) pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 0.0%, Ccues 0.0%, Bw 0.0%; guessing rate on never-stated facts pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 0.0%, Ccues 0.0%, Bw 0.0%
- Writing cost: 11 completion and 141 prompt tokens per message (<= 150 and <= 500); about 2.1 phone-minutes per 100 messages at 20 tokens/s writing and 200 tokens/s reading (assumed)
- Events (reported line): B 87.6% (>= 70.0)
- Conditions for PASS: {"B_hard": false, "oracle_ratio": false, "assoc_each": false, "margin": true, "merges": false, "hard_merges": false, "fast_path": false, "write_cost": true, "forget": false}
- B: found every needed message for 67.1% of questions; median completion 8 tokens; cut off 0.0%
- oracle: found every needed message for 92.1% of questions; median completion 7 tokens; cut off 0.0%
- nolinks: found every needed message for 61.9% of questions; median completion 8 tokens; cut off 0.0%
- C: found every needed message for 44.2% of questions; median completion 10 tokens; cut off 0.0%
- Ccues: found every needed message for 45.3% of questions; median completion 8 tokens; cut off 0.0%
- Thinking writer verdict: the thinking writer does not pass either (conditions {"B_hard": false, "oracle_ratio": false, "assoc_each": false, "margin": true, "merges": false, "hard_merges": false, "fast_path": false, "write_cost": true, "forget": false, "no_rows_left": true})
- Thinking writer: B-w hard 78.1% (off 78.1%); merges 3.72%, hard 12.91%; forgetting 94.6%; 11 completion tokens per message
