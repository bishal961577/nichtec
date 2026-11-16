# Step 2: the whole chain on the dev world (first 1,500 messages)

Pre-registration: reports/Two ends check 2026.md, step 2.

| Measure | Chain | Line | B on dev |
|---|---|---|---|
| Wrong merges | 9.42% | <= 1.5% | 6.27% |
| "X's flatmate" filed right | 45.0% | >= 75% | 11.1% |
| Worst-person forgetting | 49.2% (average 83.9%) | >= 90% | 62.2% |
| Facts filed right | 79.1% | >= 85% | 78.2% |
| Hard-case merges | 32.8% | reported | 33.3% |
| Link precision | 86.1% | reported | 57.1% |
| Mentions left unresolved | 9.2% | reported | - |
| Filed right by mention style | {"fname": 96.6, "fname_ctx": 96.2, "full": 98.6, "nick": 91.7, "pronoun": 66.7, "pronoun_prev": 97.1, "role": 81.6, "via": 45.0} | reported | - |
| Hard set, chain, thinking off | 54.9% | >= 77.3% | B thinking on 72.3%, off 50.3% |
| Chain minus B thinking on (people-resampled 95% interval) | -15.3 [-22.7, -8.0] | interval above 0 | - |
| Median answer | 22 tokens | <= 64 | about 4,200 |
| Found every needed message | 75.7% | reported | 71.7% |
| By kind | {"deleted": 85.4, "desc": 50.0, "event_last": 22.2, "event_what": 72.7, "event_who": 25.0, "fname_ctx": 69.2, "nick": 61.8, "retract": 40.0, "role": 33.3, "two": 35.3, "upd": 55.9, "who": 40.6} | reported | - |

**Verdict (dev, as registered): STOP**

Checks by construction: shuffled input order gives the same view: True; bindings that change when the 35 deleted messages are removed: 3; rows left carrying a deleted value: 0.
