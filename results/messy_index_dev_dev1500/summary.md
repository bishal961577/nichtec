# Messy chat index test: dev (seed 101)

Pre-registration: reports/Messy chat index test 2026.md (with amendments 1 and 2). Accuracy in %; hard set = role, nick, fname_ctx, desc, who, two, upd, retract, deleted, event_last, event_what, event_who

| Kind | n | pages think | B think | oracle think | nolinks think | C think | Ccues think | pages off | B off | C off | Ccues off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| amb | 3 | 66.7 | 66.7 | 100.0 | 66.7 | 0.0 | 33.3 | 33.3 | 33.3 | 0.0 | 66.7 |
| deleted | 41 | 90.2 | 90.2 | 90.2 | 92.7 | 70.7 | 65.9 | 92.7 | 95.1 | 46.3 | 63.4 |
| desc | 26 | 100.0 | 46.2 | 80.8 | 42.3 | 30.8 | 0.0 | 76.9 | 26.9 | 0.0 | 0.0 |
| event_last | 9 | 88.9 | 77.8 | 88.9 | 66.7 | 55.6 | 77.8 | 33.3 | 33.3 | 33.3 | 33.3 |
| event_what | 22 | 100.0 | 81.8 | 86.4 | 81.8 | 86.4 | 77.3 | 90.9 | 68.2 | 68.2 | 68.2 |
| event_who | 20 | 95.0 | 75.0 | 70.0 | 65.0 | 70.0 | 90.0 | 85.0 | 50.0 | 75.0 | 45.0 |
| fname_ctx | 26 | 100.0 | 76.9 | 100.0 | 76.9 | 88.5 | 73.1 | 65.4 | 53.8 | 61.5 | 26.9 |
| named | 37 | 89.2 | 70.3 | 89.2 | 75.7 | 62.2 | 56.8 | 56.8 | 45.9 | 59.5 | 29.7 |
| never | 40 | 95.0 | 90.0 | 95.0 | 90.0 | 55.0 | 60.0 | 100.0 | 100.0 | 27.5 | 52.5 |
| nick | 34 | 97.1 | 70.6 | 94.1 | 73.5 | 61.8 | 8.8 | 58.8 | 55.9 | 14.7 | 5.9 |
| retract | 15 | 100.0 | 86.7 | 93.3 | 93.3 | 80.0 | 53.3 | 26.7 | 33.3 | 33.3 | 33.3 |
| role | 12 | 100.0 | 83.3 | 91.7 | 83.3 | 41.7 | 41.7 | 33.3 | 25.0 | 41.7 | 16.7 |
| two | 17 | 94.1 | 35.3 | 94.1 | 0.0 | 52.9 | 5.9 | 23.5 | 5.9 | 0.0 | 5.9 |
| upd | 34 | 88.2 | 67.6 | 85.3 | 67.6 | 61.8 | 67.6 | 52.9 | 41.2 | 64.7 | 29.4 |
| who | 32 | 96.9 | 53.1 | 65.6 | 50.0 | 46.9 | 50.0 | 84.4 | 46.9 | 25.0 | 18.8 |
| HARD | 288 | 95.5 | 70.1 | 86.1 | 67.4 | 62.8 | 50.0 | 66.7 | 50.3 | 39.2 | 29.9 |
| ASSOC | 75 | 97.3 | 46.7 | 77.3 | 36.0 | 42.7 | 22.7 | 68.0 | 30.7 | 10.7 | 9.3 |
| EVENTS | 51 | 96.1 | 78.4 | 80.4 | 72.5 | 74.5 | 82.4 | 78.4 | 54.9 | 64.7 | 52.9 |
| held-out | 0 | - | - | - | - | - | - | - | - | - | - |

## Verdict

- **WRITER-LIMITED: the reading works on a perfect index, the 4B's filing does not; next is the thinking writer (pre-registered), a bigger writer only if that fails too**
- B hard 70.1% (>= 85.0); oracle 86.1% (read-limited below 85.0); B/oracle 0.81 (>= 0.9); desc 46.2%, who 53.1%, two 35.3% (each >= 75.0)
- Fast path: B with thinking off 50.3% on the hard set (>= 80.0)
- C: hard 62.8%, desc, who, two 42.7% (desc 30.8, who 46.9, two 52.9); B minus C on them 4.0 points [-9.1, 16.2] (people resampled, n = 75) <- the stronger search, used for the margin
- Ccues: hard 50.0%, desc, who, two 22.7% (desc 0.0, who 50.0, two 5.9); B minus Ccues on them 24.0 points [11.0, 36.2] (people resampled, n = 75)
- Filing: wrong merges 6.27% (<= 1.0; writer-limited above 3.0); hard cases 33.33% of 90 (<= 5.0; writer-limited above 10.0); facts filed 78.2%; filed right by mention style {"fname": 99.1, "fname_ctx": 90.6, "full": 97.6, "nick": 100.0, "pronoun": 97.9, "pronoun_prev": 72.4, "role": 100.0, "via": 11.1} (n {"fname": 112, "fname_ctx": 53, "full": 212, "nick": 9, "pronoun": 48, "pronoun_prev": 29, "role": 56, "via": 18}); link precision 57.1% of 91; lure intrusion 51.9%; 132 index people for 260, 19 split; 60 rows about me
- Forgetting a whole person: 89.0% of their messages removed on average; worst person 62.2% (every person >= 95.0); collateral 3.9% of removed
- Deletion: rows elsewhere still carrying a deleted value 0; deleted values shown to a reader 0; answers naming a deleted value (thinking on) pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 0.0%, Ccues 0.0%; guessing rate on never-stated facts pages 0.0%, B 0.0%, oracle 0.0%, nolinks 0.0%, C 37.5%, Ccues 35.0%
- Writing cost: 15 completion and 139 prompt tokens per message (<= 150 and <= 500); about 2.4 phone-minutes per 100 messages at 20 tokens/s writing and 200 tokens/s reading (assumed)
- Events (reported line): B 78.4% (>= 70.0)
- Conditions for PASS: {"B_hard": false, "oracle_ratio": false, "assoc_each": false, "margin": false, "merges": false, "hard_merges": false, "fast_path": false, "write_cost": true, "forget": false}
- B: found every needed message for 71.7% of questions; median completion 4172 tokens; cut off 0.0%
- oracle: found every needed message for 92.7% of questions; median completion 4449 tokens; cut off 0.3%
- nolinks: found every needed message for 69.8% of questions; median completion 4184 tokens; cut off 0.0%
- C: found every needed message for 58.2% of questions; median completion 4665 tokens; cut off 0.3%
- Ccues: found every needed message for 54.6% of questions; median completion 5764 tokens; cut off 0.0%
