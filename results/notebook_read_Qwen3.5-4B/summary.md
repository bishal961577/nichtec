# Notebook reading at shipping size: Qwen3.5-4B

Pre-registration: reports/Notebook reading at shipping size 2026.md (amendment 1 = section 7). Accuracy in % with 95% bootstrap intervals over questions (10,000 resamples).

## Verdict

- PHONE SELECTION NOT VIABLE: hard set (695 questions), thinking on: two-round search 50.2%, perfect page selection 99.9%, search/selected 0.50 [0.46, 0.54] (viable >= 0.8)
- Easy reference: not complete
- Whole 5,025-note notebook on the phone-sized model (amendment 4): not complete

## Personal notebook by question kind

| Kind | Mode | 10 (pages) | 100 | 1,000 | 5,000 | BM25 top-20 | Two-round search |
|---|---|---|---|---|---|---|---|
| A: relation paraphrase, name kept | think | 100.0 [100-100] n=150 | 99.3 [98-100] n=150 | - | 29.0 [13-45] n=31 | 99.3 [98-100] n=150 | 98.7 [97-100] n=150 |
| A: relation paraphrase, name kept | nothink | 99.3 [98-100] n=150 | 97.3 [95-99] n=150 | - | - | - | 96.7 [93-99] n=150 |
| A: relation paraphrase, name kept | nothink256 | 99.3 [98-100] n=150 | - | - | 92.0 [87-96] n=150 | - | - |
| A: note's own words | think | 99.0 [97-100] n=100 | 99.0 [97-100] n=100 | - | 36.4 [21-55] n=33 | 100.0 [100-100] n=100 | 100.0 [100-100] n=100 |
| A: note's own words | nothink | 100.0 [100-100] n=100 | 96.0 [92-99] n=100 | - | - | - | 88.0 [81-94] n=100 |
| A: note's own words | nothink256 | 100.0 [100-100] n=100 | - | - | 94.0 [89-98] n=100 | - | - |
| A: two-hop | think | 100.0 [100-100] n=200 | 99.5 [98-100] n=200 | - | 30.0 [10-50] n=20 | 0.5 [0-2] n=200 | 26.0 [20-32] n=200 |
| A: two-hop | nothink | 93.0 [89-96] n=200 | 40.0 [34-47] n=200 | - | - | - | 17.0 [12-22] n=200 |
| A: two-hop | nothink256 | 93.0 [90-96] n=200 | - | - | 5.0 [2-8] n=200 | - | - |
| A: latest value | think | 100.0 [100-100] n=135 | 99.3 [98-100] n=135 | - | 25.0 [10-45] n=20 | 99.3 [98-100] n=135 | 99.3 [98-100] n=135 |
| A: latest value | nothink | 89.6 [84-95] n=135 | 91.9 [87-96] n=135 | - | - | - | 91.9 [87-96] n=135 |
| A: latest value | nothink256 | 89.6 [84-95] n=135 | - | - | 42.2 [34-50] n=135 | - | - |
| A: value before the change | think | 100.0 [100-100] n=20 | 100.0 [100-100] n=20 | - | 55.0 [35-75] n=20 | 100.0 [100-100] n=20 | 100.0 [100-100] n=20 |
| A: value before the change | nothink | 100.0 [100-100] n=20 | 100.0 [100-100] n=20 | - | - | - | 95.0 [85-100] n=20 |
| A: value before the change | nothink256 | 100.0 [100-100] n=20 | - | - | 80.0 [60-95] n=20 | - | - |
| A: retracted (says none) | think | 96.7 [90-100] n=30 | 100.0 [100-100] n=30 | - | 40.0 [20-60] n=20 | 93.3 [83-100] n=30 | 90.0 [80-100] n=30 |
| A: retracted (says none) | nothink | 100.0 [100-100] n=30 | 76.7 [60-90] n=30 | - | - | - | 93.3 [83-100] n=30 |
| A: retracted (says none) | nothink256 | 100.0 [100-100] n=30 | - | - | 3.3 [0-10] n=30 | - | - |
| A3: described person | think | 100.0 [100-100] n=120 | - | - | 0.0 [0-0] n=20 | 0.0 [0-0] n=120 | 0.8 [0-2] n=120 |
| A3: described person | nothink | 99.2 [98-100] n=120 | - | - | - | - | 0.8 [0-2] n=120 |
| A3: described person | nothink256 | 99.2 [98-100] n=120 | - | - | 2.5 [0-6] n=120 | - | - |
| A3: who is it | think | 100.0 [100-100] n=60 | - | - | 0.0 [0-0] n=20 | 18.3 [10-28] n=60 | 20.0 [10-30] n=60 |
| A3: who is it | nothink | 100.0 [100-100] n=60 | - | - | - | - | 6.7 [2-13] n=60 |
| A3: who is it | nothink256 | 100.0 [100-100] n=60 | - | - | 3.3 [0-8] n=60 | - | - |
| A3: first name + detail | think | 100.0 [100-100] n=60 | - | - | 15.0 [0-30] n=20 | 45.0 [33-58] n=60 | 55.0 [42-67] n=60 |
| A3: first name + detail | nothink | 100.0 [100-100] n=60 | - | - | - | - | 35.0 [23-47] n=60 |
| A3: first name + detail | nothink256 | 100.0 [100-100] n=60 | - | - | 30.0 [18-42] n=60 | - | - |
| A3: ambiguous first name flagged | think | - | - | - | 0.0 [0-0] n=20 | 10.0 [0-20] n=30 | 20.0 [7-33] n=30 |
| A3: ambiguous first name flagged | nothink | - | - | - | - | - | 26.7 [10-43] n=30 |
| A3: ambiguous first name flagged | nothink256 | - | - | - | 0.0 [0-0] n=30 | - | - |
| A3: close person with lures | think | 100.0 [100-100] n=90 | - | - | 30.0 [10-50] n=20 | 100.0 [100-100] n=90 | 100.0 [100-100] n=90 |
| A3: close person with lures | nothink | 94.4 [89-99] n=90 | - | - | - | - | 93.3 [88-98] n=90 |
| A3: close person with lures | nothink256 | 94.4 [89-99] n=90 | - | - | 90.0 [83-96] n=90 | - | - |

Updates by form, thinking on (latest value right / stale):

- 10 notes: explicit 100% / 0%, plain 100% / 0%, correction 100% / 0%, two changes 100%
- 100 notes: explicit 100% / 0%, plain 98% / 2%, correction 100% / 0%, two changes 100%
- 5000 notes: explicit 0% / 33%, plain 0% / 29%, correction 71% / 0%, two changes 0%
- bm25 notes: explicit 100% / 0%, plain 100% / 0%, correction 98% / 2%, two changes 97%
- search2 notes: explicit 100% / 0%, plain 100% / 0%, correction 98% / 2%, two changes 97%

Where wrong answers came from, thinking on (hard set):

- 10: 1 wrong: no 'none' in answer 1
- 100: 2 wrong: old value 1, cut off 1
- 5000: 112 wrong: cut off 102, old value 8, another person's value 2
- bm25: 404 wrong: no value 307, no name 49, the asker's own value (A) 28, another person's value 6, same last name 6, same first name 5, old value 2, no 'none' in answer 1
- search2: 346 wrong: no value 251, no name 47, the asker's own value (A) 28, another person's value 6, same first name 5, same last name 3, old value 2, no 'none' in answer 2, cut off 1, partial match 1

Cost per answer, thinking on (mean prompt tokens, completion tokens, seconds under load; cut off):

- 10: prompt 221, completion 891, 23.1 s, cut off 0.1%
- 100: prompt 1616, completion 2327, 62.7 s, cut off 0.3%
- 5000: prompt 60188, completion 13278, 902.1 s, cut off 65.2%
- bm25: prompt 304, completion 3342, 99.1 s, cut off 0.0%
- search2: prompt 323, completion 3047, 99.4 s, cut off 0.1%

Preflight (single requests, not scored): {"served": ["Qwen/Qwen3.5-4B"], "model": "Qwen/Qwen3.5-4B", "cold": {"prompt_tokens": 60185, "cached_tokens": null, "completion_tokens": 10, "sec": 4.73, "truncated": false}, "warm_other_question": {"prompt_tokens": 60184, "cached_tokens": null, "completion_tokens": 11, "sec": 0.55, "truncated": false}, "warm_same": {"prompt_tokens": 60184, "cached_tokens": null, "completion_tokens": 11, "sec": 0.63, "truncated": false}, "warm_thinking": {"prompt_tokens": 60182, "cached_tokens": null, "completion_tokens": 638, "sec": 10.18, "truncated": false}, "bm25_pages": {"prompt_tokens": 277, "sec": 0.39}}

## Conflicting world edits (MQuAKE-CF-3k-v2)

| Question | Arm, mode | 10 edits | 100 | 1,000 | all | Own facts, no notes |
|---|---|---|---|---|---|---|
| single-hop, new answer | world, think | 100.0 [100-100] old 0% | 100.0 [100-100] old 0% | - | - | - |
| two-hop, new answer | world, think | 86.7 [80-92] old 3% | 80.8 [73-88] old 2% | - | - | 55.8 |

World: {"people": 840, "groups": 420, "notes": 5025, "personal_questions": {"reword": 150, "same": 100, "two": 200, "upd": 135, "prev": 20, "retract": 30, "desc": 120, "who": 60, "fname": 60, "amb": 30, "close": 90}, "hard_set": 695, "bm25_recall": {"reword": 1.0, "same": 1.0, "two": 0.005, "upd": 1.0, "prev": 1.0, "retract": 0.967, "desc": 0.0, "fname": 0.45, "close": 1.0}, "world_edits": 2764}
