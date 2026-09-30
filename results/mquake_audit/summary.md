# MQuAKE label audit: are the labels right when every edit is applied at once?

Tests 4 and 5 write all of a file's edits into one memory and score every case against its new answer. A
case's label is then wrong if another case edits a step of its chain that it does not edit itself
(contaminated), or two cases edit the same step to different answers (conflicting). Cases are matched by
their Wikidata fact chain, since case numbers differ between the files.

| Data | Cases | Distinct edits | Clean | Conflicting | Contaminated | Duplicate cases |
|---|---|---|---|---|---|---|
| MQuAKE-CF-3k (original) | 3,000 | 2,786 | 2,002 | 0 | 998 (33.3%) | 4 |
| MQuAKE-CF-3k-v2 (tests 4 and 5) | 3,000 | 2,764 | 3,000 | 0 | 0 (0.0%) | 16 |
| MQuAKE-CF (9,218 cases) | 9,218 | 7,253 | 6,370 | 0 | 2,848 (30.9%) | 41 |
| MQuAKE-T | 1,868 | 96 | 1,867 | 0 | 1 (0.1%) | 0 |

remastered_CF3k_error: ProxyError: 403 Forbidden

remastered_T_error: ProxyError: 403 Forbidden
