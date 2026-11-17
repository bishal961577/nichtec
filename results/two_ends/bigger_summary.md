# Step 4: the bigger describer (Qwen3.5-27B-FP8) on the dev world

Pre-registration: reports/Two ends check 2026.md, step 4. Describer v3, thinking off, memory-blind; the same resolver; the reporting-verb filter on (registered before the run).

Messages touched by the practice world's pronoun quirk (left out of the split measure): 66.

| Measure | 4B own | 4B A | 4B B | 27B own | 27B A | 27B B | Line |
|---|---|---|---|---|---|---|---|
| Wrong merges, everyday facts (primary) | 8.17% | 6.57% | 6.83% | 4.02% | 3.44% | 3.72% | <= 1.5% (stop > 3%) |
| Same, without the pronoun quirk | 5.11% | 3.23% | 3.51% | 2.92% | 1.98% | 2.29% | reported (interpretation) |
| Wrong merges, registered count | 7.83% | 5.66% | 6.01% | 4.02% | 2.92% | 3.21% | reported |
| "X's flatmate" filed right | 45.0% | 65.0% | 65.0% | 81.0% | 100.0% | 100.0% | >= 75% |
| Worst-person forgetting | 49.2% | 92.1% | 92.1% | 42.9% | 100.0% | 100.0% | >= 90% |
| Facts filed right | 78.8% | 80.8% | 80.7% | 88.0% | 89.1% | 88.9% | >= 85% |
| Link precision | 86.1% | 100.0% | 100.0% | 90.1% | 100.0% | 100.0% | reported |
| Mentions left unresolved | 9.2% | 4.3% | 3.2% | 22.6% | 1.4% | 0.6% | reported |
| Owner questions | – | 204 | 281 | – | 204 | 332 | B <= 2 a day |
| Verdict (dev) | STOP | STOP | STOP | STOP | STOP | STOP |  |

Wrong everyday rows by how the person the fact is about was mentioned:

- 4B none: {"pronoun": 21, "role": 12, "via": 10, "fname": 9, "full": 7, "fname_ctx": 4, "pronoun_prev": 1}
- 4B A: {"pronoun": 23, "via": 10, "fname": 9, "full": 7, "role": 2, "fname_ctx": 1}
- 4B B: {"pronoun": 23, "via": 10, "fname": 9, "full": 7, "fname_ctx": 3, "role": 2}
- 27B none: {"pronoun": 13, "full": 8, "fname_ctx": 5, "fname": 1, "via": 1, "pronoun_prev": 1}
- 27B A: {"pronoun": 14, "full": 8, "role": 2, "fname": 1}
- 27B B: {"pronoun": 14, "full": 8, "fname_ctx": 2, "role": 2, "fname": 1}

Without the filter, the 27B: none 4.02%; A 3.44%; B 3.72%

Writing end on the 300 sample (corrected scorer), 4B -> 27B:

- attached_easy: 96.2% -> 100.0%
- attached_hard: 89.8% -> 97.1%
- rel_direction: 64.7% -> 100.0%
- lures_tagged: 93.3% -> 95.0%
- definitions: 92.3% -> 84.6%
- fact_recall: 96.1% -> 98.4%
- parsed: 100.0% -> 100.0%

27B: 0 messages without valid JSON; tokens per message 145 out, 1452 in.
