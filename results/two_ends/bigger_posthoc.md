# Step 4, post-hoc: the antecedent rule fixed (dev; not a judged run)

Condition A (every definition confirmed), reporting-verb filter on. 'Fixed': the describer's "previous" may name someone in the same message, the teller included, and names match as whole words.

| Measure | 4B | 4B fixed | 27B | 27B fixed | Line |
|---|---|---|---|---|---|
| Wrong merges, everyday facts | 6.57% | 5.56% | 3.44% | 1.52% | <= 1.5% (stop > 3%) |
| Same, without the pronoun quirk | 3.23% | 2.66% | 1.98% | 0.61% | - |
| Wrong rows | 52 | 44 | 25 | 11 | - |
| "X's flatmate" filed right | 65.0% | 65.0% | 100.0% | 100.0% | >= 75% |
| Worst-person forgetting | 92.1% | 92.1% | 100.0% | 100.0% | >= 90% |
| Facts filed right | 80.8% | 81.7% | 89.1% | 90.5% | >= 85% |

Condition B (asking policy), wrong merges on everyday facts: 4B 6.83% (without the quirk 3.51%, 281 questions); 4B fixed 5.82% (without the quirk 2.95%, 281 questions); 27B 3.72% (without the quirk 2.29%, 332 questions); 27B fixed 1.79% (without the quirk 0.92%, 332 questions)

Wrong rows by how the person was mentioned, A: 4B {"pronoun": 23, "via": 10, "fname": 9, "full": 7, "role": 2, "fname_ctx": 1}; 4B fixed {"pronoun": 16, "via": 10, "fname": 9, "full": 7, "role": 1, "fname_ctx": 1}; 27B {"pronoun": 14, "full": 8, "role": 2, "fname": 1}; 27B fixed {"pronoun": 6, "full": 4, "fname": 1}
