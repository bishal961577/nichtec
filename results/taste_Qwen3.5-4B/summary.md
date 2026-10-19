# Real-data test: Qwen3.5-4B

Pre-registration: reports/Real data test 2026.md. Mean per-person test AUC over 300 MovieLens-1M users (80 ratings learned, next 40 tested). 0.5 = guessing.

| Agent | Mean AUC |
|---|---|
| A, history in the prompt | 0.614 |
| W, taste written in words | 0.572 |
| **E, model names questions, code learns the person** | **0.537** |
| Genres + decades, same learner (no model) | 0.552 |
| Popularity before the test window (not personal) | 0.679 |
| Collaborative filtering, everyone's data (reference) | 0.740 |

Paired differences, mean [95% bootstrap interval over people]:
- E - A: -0.078 [-0.093, -0.061]
- E - W: -0.035 [-0.052, -0.017]
- E - genres: -0.015 [-0.031, -0.000]
- E - popularity: -0.142 [-0.161, -0.123]

Answer calls: 396,446; neither yes nor no: 0.0%; 10.6 min.
Questions most often given weight: Is the movie a comedy? (38); Is the movie a horror film? (24); Is the movie a romantic comedy? (14); Is the movie a science fiction film? (13); Is the movie based on a book or novel? (12); Is the movie a sequel to a previous film? (11); Is the movie a comedy or a comedy-drama? (10); Is the movie a drama? (9); Is the movie a musical? (8); Is the movie a heist or crime thriller? (7); Is the movie a James Bond film? (7); Is the movie a thriller or crime drama? (7); Is the movie a heist or crime drama? (7); Is the movie a remake of an older film? (6); Does the movie feature a strong ensemble cast? (6)

## Verdict (test 1)

- K1, answering: 0.0% of answers neither yes nor no (line <= 10): ok.
- K2, signal: collaborative filtering mean AUC 0.740 (line >= 0.60): ok.
- STOP: E minus the better of A and W (A) is -0.078 [-0.093, -0.061]; the interval includes or lies below 0. This line of work ends.

Check before the run: {'reading_first_token': 'No', 'reading_answers_yes_no': True, 'questions_named': ['Is the movie a drama?', 'Is the movie based on a true story?', 'Does the movie feature a strong female protagonist?', 'Is the movie directed by a male director?', 'Is the movie about a famous historical figure?', 'Does the movie have a complex or tragic plot?', 'Is the movie a remake of an earlier film?', 'Does the movie focus on crime or law enforcement?', 'Is the movie set in the 1990s or earlier?', 'Does the movie explore themes of redemption or social justice?', 'Is the movie considered a cult classic by this user?', 'Does the movie feature a famous actor in a lead role?']}
