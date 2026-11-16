# Two ends check: results (dev world, first 1,500 messages)

Pre-registration: reports/Two ends check 2026.md

## Writing end (300 dev messages, memory-blind, thinking off)

| Measure | Result | Line |
|---|---|---|
| Claims on the right mention, easy | 87.0% of 23 | >= 98 |
| Claims on the right mention, hard | 92.0% of 88 | >= 93 (stop < 85 after a week) |
| Relation direction (one-way, of captured) | 45.5% (captured 64.7% of 17) | >= 95 |
| Lures tagged | 93.3% of 60 | >= 90 |
| Definitions captured | 52.3% of 65 | >= 98 |
| Fact recall | 86.0% of 129 | reported (>= 90 suggested) |
| Valid JSON | 100.0% | - |
| Tokens per message | 165 out, 1324 in | - |

**Writing-end lines met:** {"attached_easy": false, "attached_hard": false, "rel_direction": false, "lures_tagged": true, "definitions": false}

## Writing end (300 dev messages, memory-blind, thinking off) - corrected scorer

| Measure | Result | Line |
|---|---|---|
| Claims on the right mention, easy | 91.3% of 23 | >= 98 |
| Claims on the right mention, hard | 92.0% of 88 | >= 93 (stop < 85 after a week) |
| Relation direction (one-way, of captured) | 46.7% (captured 88.2% of 17) | >= 95 |
| Lures tagged | 95.0% of 60 | >= 90 |
| Definitions captured | 72.3% of 65 | >= 98 |
| Fact recall | 86.0% of 129 | reported (>= 90 suggested) |
| Valid JSON | 100.0% | - |
| Tokens per message | 165 out, 1324 in | - |

**Writing-end lines met:** {"attached_easy": false, "attached_hard": false, "rel_direction": false, "lures_tagged": true, "definitions": false}
