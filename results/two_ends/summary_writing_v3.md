# Two ends check: results (dev world, first 1,500 messages)

Pre-registration: reports/Two ends check 2026.md

## Writing end (300 dev messages, memory-blind, thinking off)

| Measure | Result | Line |
|---|---|---|
| Claims on the right mention, easy | 96.2% of 26 | >= 98 |
| Claims on the right mention, hard | 88.8% of 98 | >= 93 (stop < 85 after a week) |
| Relation direction (one-way, of captured) | 78.6% (captured 82.4% of 17) | >= 95 |
| Lures tagged | 91.7% of 60 | >= 90 |
| Definitions captured | 70.8% of 65 | >= 98 |
| Fact recall | 96.1% of 129 | reported (>= 90 suggested) |
| Valid JSON | 100.0% | - |
| Tokens per message | 168 out, 1453 in | - |

**Writing-end lines met:** {"attached_easy": false, "attached_hard": false, "rel_direction": false, "lures_tagged": true, "definitions": false}

## Writing end (300 dev messages, memory-blind, thinking off) - corrected scorer

| Measure | Result | Line |
|---|---|---|
| Claims on the right mention, easy | 96.2% of 26 | >= 98 |
| Claims on the right mention, hard | 89.8% of 98 | >= 93 (stop < 85 after a week) |
| Relation direction (one-way, of captured) | 64.7% (captured 100.0% of 17) | >= 95 |
| Lures tagged | 93.3% of 60 | >= 90 |
| Definitions captured | 92.3% of 65 | >= 98 |
| Fact recall | 96.1% of 129 | reported (>= 90 suggested) |
| Valid JSON | 100.0% | - |
| Tokens per message | 168 out, 1453 in | - |

**Writing-end lines met:** {"attached_easy": false, "attached_hard": false, "rel_direction": false, "lures_tagged": true, "definitions": false}
