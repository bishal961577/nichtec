# Agent D, the scientist, with thinking on: claude-haiku-5-5

Pre-registration: reports/Notebook reading at shipping size 2026.md (Part B). Order random1, 12 tasks, same seeds as the original run; sampling {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'max_new_tokens': 16384}.

| | Proposals | Readable rule | Guesses contradicting shown cases | Good rule in pool | D test accuracy | D's final rule good |
|---|---|---|---|---|---|---|
| thinking on | 384 | 100% | 0.0% of 384 | 100% | 99.4 | 92% |
| original (thinking off), random1 | 384 | 5% | 0.0% of 18 | 0% | 55.0 | 0% |

Original run, all five orders: 0.0% of guesses contradicted.

## Verdict

- WITHDRAWN: with thinking on, 0.0% of guesses contradict the cases shown (<= 10%). Conclusion (b) does not hold for this model in thinking mode.
