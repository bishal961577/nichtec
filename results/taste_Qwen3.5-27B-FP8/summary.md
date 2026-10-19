# Real-data test: Qwen3.5-27B-FP8

Pre-registration: reports/Real data test 2026.md. Mean per-person test AUC over 300 MovieLens-1M users (80 ratings learned, next 40 tested). 0.5 = guessing.

| Agent | Mean AUC |
|---|---|
| A, history in the prompt | 0.635 |
| W, taste written in words | 0.605 |
| E, model names questions, code learns the person | not run on this model |
| Genres + decades, same learner (no model) | 0.552 |
| Popularity before the test window (not personal) | 0.679 |
| Collaborative filtering, everyone's data (reference) | 0.740 |

Paired differences, mean [95% bootstrap interval over people]:
- W - A: -0.030 [-0.042, -0.019]
- A - popularity: -0.044 [-0.062, -0.026]

Answer calls: 1; neither yes nor no: 0.0%; 0.1 min.
Questions most often given weight: 

## Verdict (test 1)

- K1, answering: 0.0% of answers neither yes nor no (line <= 10): ok.
- K2, signal: collaborative filtering mean AUC 0.740 (line >= 0.60): ok.
- Test 1's line is not applied on this model: agent E was not run here (see the report).

Check before the run: {'reading_first_token': 'No', 'reading_answers_yes_no': True, 'questions_named': ['Is the movie released in the 1990s?', 'Is the movie directed by someone other than Stanley Kubrick or Spike Lee?', 'Does the movie feature a strong ensemble cast rather than a single protagonist?', 'Is the movie a drama with elements of hope or redemption?', 'Is the movie based on a true story?', 'Does the movie avoid being a heavy-handed political thriller or historical epic?', 'Is the movie a character study focusing on personal struggles or relationships?', 'Does the movie have a dark or quirky sense of humor?', 'Is the movie a romantic period piece with a happy or bittersweet ending?', 'Does the movie avoid excessive surrealism or abstract horror elements?', 'Is the movie widely acclaimed by critics and audiences alike?', 'Does the movie avoid being a standard Hollywood action blockbuster?']}
