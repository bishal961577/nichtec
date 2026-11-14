# Notebook reading at shipping size: claude-haiku-5-5

Pre-registration: reports/Notebook reading at shipping size 2026.md (amendment 1 = section 7). Accuracy in % with 95% bootstrap intervals over questions (10,000 resamples).

## Verdict

- WHOLE NOTEBOOK: hard set (695 questions), thinking on: whole 5,000-note pile 94.0%, two-round search 67.9%, perfect page selection 97.7%; whole/selected 0.96 [0.95, 0.98], search/selected 0.70 [0.66, 0.73] (viable >= 0.8); warm-cache answer at 5,000 notes 1.4 s (line 10 s). Version 1 in the cloud = the whole notebook in the prompt; two-round search is not enough.
- Easy reference: reworded single-hop at 5,000 notes 99.3% [98.0, 100.0] (first design: pass >= 85, stop < 70); two-hop 5,000/10 ratio 0.98 [0.96, 1.00] (pass >= 0.8, stop < 0.5); reported, not deciding

## Personal notebook by question kind

| Kind | Mode | 10 (pages) | 100 | 1,000 | 5,000 | BM25 top-20 | Two-round search |
|---|---|---|---|---|---|---|---|
| A: relation paraphrase, name kept | think | 99.3 [98-100] n=150 | 100.0 [100-100] n=150 | 100.0 [100-100] n=150 | 99.3 [98-100] n=150 | 100.0 [100-100] n=150 | 100.0 [100-100] n=150 |
| A: relation paraphrase, name kept | nothink | 99.3 [98-100] n=150 | - | - | 52.0 [44-60] n=150 | - | 100.0 [100-100] n=150 |
| A: note's own words | think | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 |
| A: note's own words | nothink | 100.0 [100-100] n=100 | - | - | 62.0 [52-71] n=100 | - | 100.0 [100-100] n=100 |
| A: two-hop | think | 100.0 [100-100] n=200 | 99.5 [98-100] n=200 | 99.5 [98-100] n=200 | 98.5 [96-100] n=200 | 0.5 [0-2] n=200 | 68.5 [62-75] n=200 |
| A: two-hop | nothink | 99.5 [98-100] n=200 | - | - | 8.5 [5-12] n=200 | - | 53.5 [46-60] n=200 |
| A: latest value | think | 100.0 [100-100] n=135 | 98.5 [96-100] n=135 | 100.0 [100-100] n=135 | 94.8 [91-99] n=135 | 100.0 [100-100] n=135 | 100.0 [100-100] n=135 |
| A: latest value | nothink | 100.0 [100-100] n=135 | - | - | 46.7 [39-55] n=135 | - | 94.8 [91-99] n=135 |
| A: value before the change | think | 100.0 [100-100] n=20 | 95.0 [85-100] n=20 | 100.0 [100-100] n=20 | 95.0 [85-100] n=20 | 100.0 [100-100] n=20 | 100.0 [100-100] n=20 |
| A: value before the change | nothink | 100.0 [100-100] n=20 | - | - | 45.0 [25-65] n=20 | - | 100.0 [100-100] n=20 |
| A: retracted (says none) | think | 50.0 [33-67] n=30 | 26.7 [13-43] n=30 | 13.3 [3-27] n=30 | 0.0 [0-0] n=30 | 33.3 [17-50] n=30 | 40.0 [23-57] n=30 |
| A: retracted (says none) | nothink | 23.3 [10-40] n=30 | - | - | 3.3 [0-10] n=30 | - | 10.0 [0-23] n=30 |
| A3: described person | think | 99.2 [98-100] n=120 | - | - | 100.0 [100-100] n=120 | 0.0 [0-0] n=120 | 15.0 [9-22] n=120 |
| A3: described person | nothink | 100.0 [100-100] n=120 | - | - | 0.0 [0-0] n=120 | - | 9.2 [4-15] n=120 |
| A3: who is it | think | 100.0 [100-100] n=60 | - | - | 98.3 [95-100] n=60 | 18.3 [10-28] n=60 | 33.3 [22-45] n=60 |
| A3: who is it | nothink | 98.3 [95-100] n=60 | - | - | 1.7 [0-5] n=60 | - | 26.7 [17-38] n=60 |
| A3: first name + detail | think | 100.0 [100-100] n=60 | - | - | 98.3 [95-100] n=60 | 45.0 [32-58] n=60 | 100.0 [100-100] n=60 |
| A3: first name + detail | nothink | 100.0 [100-100] n=60 | - | - | 3.3 [0-8] n=60 | - | 91.7 [85-98] n=60 |
| A3: ambiguous first name flagged | think | - | - | - | 90.0 [77-100] n=30 | 100.0 [100-100] n=30 | 100.0 [100-100] n=30 |
| A3: ambiguous first name flagged | nothink | - | - | - | 30.0 [13-47] n=30 | - | 100.0 [100-100] n=30 |
| A3: close person with lures | think | 100.0 [100-100] n=90 | - | - | 100.0 [100-100] n=90 | 100.0 [100-100] n=90 | 100.0 [100-100] n=90 |
| A3: close person with lures | nothink | 90.0 [83-96] n=90 | - | - | 50.0 [40-60] n=90 | - | 96.7 [92-100] n=90 |

Updates by form, thinking on (latest value right / stale):

- 10 notes: explicit 100% / 0%, plain 100% / 0%, correction 100% / 0%, two changes 100%
- 100 notes: explicit 100% / 0%, plain 96% / 0%, correction 100% / 0%, two changes 97%
- 1000 notes: explicit 100% / 0%, plain 100% / 0%, correction 100% / 0%, two changes 100%
- 5000 notes: explicit 98% / 2%, plain 87% / 13%, correction 100% / 0%, two changes 93%
- bm25 notes: explicit 100% / 0%, plain 100% / 0%, correction 100% / 0%, two changes 100%
- search2 notes: explicit 100% / 0%, plain 100% / 0%, correction 100% / 0%, two changes 100%

Where wrong answers came from, thinking on (hard set):

- 10: 16 wrong: old value 15, group partner 1
- 100: 25 wrong: old value 24, no value 1
- 1000: 27 wrong: old value 26, no value 1
- 5000: 42 wrong: old value 37, no value 3, another person's value 1, no name 1
- bm25: 421 wrong: no value 335, no name 42, old value 20, the asker's own value (A) 11, partial match 7, same first name 3, same last name 1, another person's value 1, group partner 1
- search2: 223 wrong: no value 159, no name 31, old value 18, partial match 9, the asker's own value (A) 5, another person's value 1

Cost per answer, thinking on (mean prompt tokens, completion tokens, seconds under load; cut off):

- 10: prompt 345, completion 27, 0.9 s, cut off 0.0%
- 100: prompt 2679, completion 64, 1.1 s, cut off 0.0%
- 1000: prompt 20157, completion 143, 1.5 s, cut off 0.0%
- 5000: prompt 100543, completion 346, 2.5 s, cut off 0.0%
- bm25: prompt 488, completion 90, 1.2 s, cut off 0.0%
- search2: prompt 588, completion 100, 1.2 s, cut off 0.0%

Preflight (single requests, not scored): {"served": ["claude-haiku-5-5"], "model": "claude-haiku-5-5", "cold": {"prompt_tokens": 100530, "cached_tokens": 0, "completion_tokens": 30, "sec": 1.86, "truncated": false}, "warm_other_question": {"prompt_tokens": 100530, "cached_tokens": 100501, "completion_tokens": 26, "sec": 1.41, "truncated": false}, "warm_same": {"prompt_tokens": 100530, "cached_tokens": 100501, "completion_tokens": 26, "sec": 1.26, "truncated": false}, "warm_thinking": {"prompt_tokens": 100532, "cached_tokens": 0, "completion_tokens": 49, "sec": 1.21, "truncated": false}, "bm25_pages": {"prompt_tokens": 431, "sec": 1.05}, "rate_limits": {"anthropic-ratelimit-input-tokens-limit": "2000000", "anthropic-ratelimit-input-tokens-remaining": "2000000", "anthropic-ratelimit-input-tokens-reset": "2026-11-14T10:50:02Z", "anthropic-ratelimit-output-tokens-limit": "400000", "anthropic-ratelimit-output-tokens-remaining": "400000", "anthropic-ratelimit-output-tokens-reset": "2026-11-14T10:50:02Z", "anthropic-ratelimit-requests-limit": "4000", "anthropic-ratelimit-requests-remaining": "3999", "anthropic-ratelimit-requests-reset": "2026-11-14T10:50:00Z", "anthropic-ratelimit-tokens-limit": "2400000", "anthropic-ratelimit-tokens-remaining": "2400000", "anthropic-ratelimit-tokens-reset": "2026-11-14T10:50:02Z"}}

## Conflicting world edits (MQuAKE-CF-3k-v2)

| Question | Arm, mode | 10 edits | 100 | 1,000 | all | Own facts, no notes |
|---|---|---|---|---|---|---|
| single-hop, new answer | world, think | 100.0 [100-100] old 0% | 100.0 [100-100] old 0% | 100.0 [100-100] old 0% | 100.0 [100-100] old 0% | - |
| single-hop, new answer | world_noinst, think | - | - | - | 100.0 [100-100] old 0% | - |
| single-hop, new answer | world, nothink | 99.3 [98-100] old 0% | - | - | 95.3 [92-99] old 1% | - |
| two-hop, new answer | world, think | 89.2 [83-94] old 3% | 85.8 [79-92] old 4% | 85.0 [78-91] old 5% | 81.7 [74-88] old 5% | 75.8 |
| two-hop, new answer | world_noinst, think | - | - | - | 80.8 [73-88] old 5% | - |
| two-hop, new answer | world, nothink | 76.7 [69-84] old 6% | - | - | 51.7 [42-61] old 6% | 65.8 |

World: {"people": 840, "groups": 420, "notes": 5025, "pile_notes": {"10": [9, 44], "100": [101, 207], "1000": [898, 1106], "5000": [5025, 5025]}, "pile_chars_5000": 210676, "personal_questions": {"reword": 150, "same": 100, "two": 200, "upd": 135, "prev": 20, "retract": 30, "desc": 120, "who": 60, "fname": 60, "amb": 30, "close": 90}, "hard_set": 695, "bm25_recall": {"reword": 1.0, "same": 1.0, "two": 0.005, "upd": 1.0, "prev": 1.0, "retract": 0.967, "desc": 0.0, "fname": 0.45, "close": 1.0}, "world_edits": 2764, "world_edit_clashes_skipped": 0, "world_questions": {"w1": 150, "w2": 120}, "world_piles": {"10": 69, "100": 7, "1000": 1, "all": 1}}
