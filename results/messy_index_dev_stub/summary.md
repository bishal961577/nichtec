# Messy chat index test: dev (seed 101)

Pre-registration: reports/Messy chat index test 2026.md. Accuracy in %; hard set = role, nick, fname_ctx, desc, who, two, upd, retract, deleted, event_last, event_what, event_who

| Kind | n | pages think | B think | oracle think | nolinks think | C think | pages off | B off | C off |
|---|---|---|---|---|---|---|---|---|---|
| amb | 29 | 100.0 | 86.2 | 100.0 | 86.2 | 24.1 | 100.0 | 86.2 | 24.1 |
| deleted | 100 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| desc | 150 | 100.0 | 69.3 | 92.0 | 69.3 | 0.0 | 100.0 | 69.3 | 0.0 |
| event_last | 170 | 100.0 | 94.1 | 100.0 | 94.1 | 79.4 | 100.0 | 94.1 | 79.4 |
| event_what | 152 | 100.0 | 94.7 | 100.0 | 94.7 | 71.1 | 100.0 | 94.7 | 71.1 |
| event_who | 70 | 100.0 | 94.3 | 98.6 | 94.3 | 68.6 | 100.0 | 94.3 | 68.6 |
| fname_ctx | 93 | 100.0 | 73.1 | 81.7 | 73.1 | 60.2 | 100.0 | 73.1 | 60.2 |
| named | 150 | 100.0 | 96.0 | 100.0 | 96.0 | 84.7 | 100.0 | 96.0 | 84.7 |
| nick | 119 | 100.0 | 91.6 | 100.0 | 91.6 | 16.0 | 100.0 | 91.6 | 16.0 |
| retract | 114 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| role | 53 | 100.0 | 83.0 | 100.0 | 83.0 | 34.0 | 100.0 | 83.0 | 34.0 |
| two | 180 | 100.0 | 89.4 | 100.0 | 0.0 | 1.7 | 100.0 | 89.4 | 1.7 |
| upd | 180 | 100.0 | 87.8 | 100.0 | 87.8 | 56.1 | 100.0 | 87.8 | 56.1 |
| who | 150 | 100.0 | 74.7 | 93.3 | 74.7 | 18.7 | 100.0 | 74.7 | 18.7 |
| HARD | 1531 | 100.0 | 87.5 | 97.4 | 77.0 | 47.7 | 100.0 | 87.5 | 47.7 |
| ASSOC | 480 | 100.0 | 78.5 | 95.4 | 45.0 | 6.5 | 100.0 | 78.5 | 6.5 |
| EVENTS | 392 | 100.0 | 94.4 | 99.7 | 94.4 | 74.2 | 100.0 | 94.4 | 74.2 |
| held-out | 0 | - | - | - | - | - | - | - | - |

## Verdict

- **BETWEEN (neither line met)**
- B hard 87.5% (pass >= 85.0, stop < 70.0); oracle 97.4%, ratio 0.90 (>= 0.9); desc 69.3%, who 74.7%, two 89.4% (each >= 75.0)
- B minus C on desc, who, two pooled: 72.1 points [67.7, 76.2] (n = 480; pass >= 5.0 with the interval above 0)
- Filing: wrong merges 0.81% (<= 1.0), near names 0.52% (<= 3.0); facts filed 93.7%; 260 index people for 260 real; 0 people split into several ids; lure rows 0 of 604 lure messages
- Deletion: answers naming a deleted value 0; rows elsewhere still carrying a deleted value 0
- Writing cost: 6 completion tokens per message (<= 1000); 82 prompt tokens per message
- Events (reported line): B 94.4% on event_last, event_what, event_who (>= 70.0)
- Fast path (reported line): B with thinking off 87.5% on the hard set (>= 85.0)
- Conditions: {"hard": true, "oracle_ratio": false, "assoc_each": false, "margin": true, "merges": true, "no_leak": true, "write_cost": true}
- Found every needed message (retrieval, thinking on): B 81.2%, oracle 97.5%, nolinks 73.2%, C 39.2%
- Cost B think: median completion 7 tokens, cut off 0.0%
- Cost C think: median completion 10 tokens, cut off 0.0%
- Cost pages think: median completion 7 tokens, cut off 0.0%
- Cost B nothink: median completion 7 tokens, cut off 0.0%
