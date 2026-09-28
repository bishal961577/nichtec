# Real-LM memory test 3 (snapped canonical keys): Qwen/Qwen2.5-0.5B

Memory after block 14 of 24, 65,536 slots (1024 person codes x 64 relation codes) x 896 dims; 24,000 facts in nights of 3,000; 4 wordings written per fact.

No memory / target injected directly (first-night facts): {"train_none": 0.0245, "train_oracle": 1.0, "test_none": 0.02, "test_oracle": 1.0}

Relation classified correctly: {"written_wordings": 1.0, "unseen_wordings": 0.965}

Person key: {"block": 7, "theta": 0.98063, "per_block": {"3": {"same_name_min_cos": 0.99685, "different_names_max_cos": 0.97106, "different_names_mean_nn_cos": 0.8232}, "5": {"same_name_min_cos": 0.99513, "different_names_max_cos": 0.9734, "different_names_mean_nn_cos": 0.8105}, "7": {"same_name_min_cos": 0.9954, "different_names_max_cos": 0.96586, "different_names_mean_nn_cos": 0.7903}, "9": {"same_name_min_cos": 0.99433, "different_names_max_cos": 0.96523, "different_names_mean_nn_cos": 0.7917}}, "calib_same_name_pass": 1.0, "calib_different_name_pass": 0.0}

## First night's facts, top-1 accuracy after each night (no snap, no gate / snapped + gate)

| Facts | People | False merges | Constraints/slot | Distinct slots | delta_rule written | batch_ls written | joint_ls written | delta_rule unseen | batch_ls unseen | joint_ls unseen |
|---|---|---|---|---|---|---|---|---|---|---|
| 3,000 | 1,955 | 0 | 0.18 | 15,449 | 0.379 / 0.379 | 1.000 / 1.000 | 1.000 / 1.000 | 0.353 / 0.353 | 0.964 / 0.964 | 0.965 / 0.965 |
| 6,000 | 2,697 | 0 | 0.37 | 17,778 | 0.152 / 0.152 | 0.499 / 0.499 | 1.000 / 1.000 | 0.141 / 0.141 | 0.444 / 0.444 | 0.964 / 0.964 |
| 9,000 | 2,924 | 1 | 0.55 | 19,004 | 0.084 / 0.084 | 0.242 / 0.242 | 0.999 / 0.999 | 0.081 / 0.081 | 0.217 / 0.217 | 0.960 / 0.960 |
| 12,000 | 2,991 | 2 | 0.73 | 19,821 | 0.047 / 0.047 | 0.177 / 0.177 | 0.986 / 0.986 | 0.044 / 0.044 | 0.166 / 0.166 | 0.941 / 0.941 |
| 15,000 | 2,998 | 3 | 0.92 | 20,440 | 0.036 / 0.036 | 0.110 / 0.110 | 0.951 / 0.951 | 0.034 / 0.034 | 0.101 / 0.101 | 0.896 / 0.896 |
| 18,000 | 2,999 | 4 | 1.10 | 20,867 | 0.024 / 0.024 | 0.092 / 0.092 | 0.935 / 0.935 | 0.025 / 0.025 | 0.082 / 0.082 | 0.868 / 0.868 |
| 21,000 | 2,999 | 5 | 1.28 | 21,190 | 0.023 / 0.023 | 0.074 / 0.074 | 0.898 / 0.898 | 0.019 / 0.019 | 0.056 / 0.056 | 0.827 / 0.827 |
| 24,000 | 2,999 | 5 | 1.47 | 21,462 | 0.022 / 0.022 | 0.084 / 0.084 | 0.868 / 0.868 | 0.021 / 0.021 | 0.075 / 0.075 | 0.787 / 0.787 |

## Latest night, damage, certificate (no snap, no gate / snapped + gate)

| Facts | Method | Latest written | Latest unseen | Known facts | Never-written people: answer changed | Gate pass known / never-written | >20% error | Iters | Max value |
|---|---|---|---|---|---|---|---|---|---|
| 3,000 | delta_rule | 0.643 / 0.643 | 0.591 / 0.591 | 0.114 / 1.000 | 0.971 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 30.03 |
| 3,000 | batch_ls | 0.999 / 0.999 | 0.953 / 0.953 | 0.341 / 1.000 | 0.951 / 0.000 | 0.000 / 0.000 | 0.0617 | 95 | 17.58 |
| 3,000 | joint_ls | 1.000 / 1.000 | 0.957 / 0.957 | 0.250 / 1.000 | 0.953 / 0.000 | 0.000 / 0.000 | 0.0013 | 230 | 30.71 |
| 6,000 | delta_rule | 0.487 / 0.487 | 0.453 / 0.453 | 0.091 / 1.000 | 0.980 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 35.24 |
| 6,000 | batch_ls | 0.998 / 0.998 | 0.964 / 0.964 | 0.227 / 1.000 | 0.969 / 0.000 | 0.000 / 0.000 | 0.5387 | 103 | 22.55 |
| 6,000 | joint_ls | 0.999 / 0.999 | 0.966 / 0.966 | 0.045 / 1.000 | 0.980 / 0.000 | 0.000 / 0.000 | 0.0112 | 276 | 53.78 |
| 9,000 | delta_rule | 0.358 / 0.358 | 0.336 / 0.336 | 0.000 / 1.000 | 0.991 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 42.21 |
| 9,000 | batch_ls | 1.000 / 1.000 | 0.954 / 0.954 | 0.114 / 1.000 | 0.975 / 0.000 | 0.000 / 0.000 | 0.6972 | 98 | 22.06 |
| 9,000 | joint_ls | 1.000 / 1.000 | 0.954 / 0.954 | 0.045 / 1.000 | 0.985 / 0.000 | 0.000 / 0.000 | 0.3544 | 361 | 73.97 |
| 12,000 | delta_rule | 0.265 / 0.265 | 0.232 / 0.232 | 0.000 / 1.000 | 0.993 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 45.63 |
| 12,000 | batch_ls | 1.000 / 1.000 | 0.960 / 0.960 | 0.114 / 1.000 | 0.976 / 0.000 | 0.000 / 0.000 | 0.7735 | 106 | 23.16 |
| 12,000 | joint_ls | 0.994 / 0.994 | 0.948 / 0.948 | 0.045 / 1.000 | 0.979 / 0.000 | 0.000 / 0.000 | 0.6996 | 418 | 94.71 |
| 15,000 | delta_rule | 0.249 / 0.249 | 0.232 / 0.232 | 0.000 / 1.000 | 0.993 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 46.91 |
| 15,000 | batch_ls | 1.000 / 1.000 | 0.962 / 0.962 | 0.068 / 1.000 | 0.976 / 0.000 | 0.000 / 0.000 | 0.8165 | 123 | 22.66 |
| 15,000 | joint_ls | 0.981 / 0.981 | 0.927 / 0.927 | 0.136 / 1.000 | 0.980 / 0.000 | 0.000 / 0.000 | 0.8271 | 429 | 119.02 |
| 18,000 | delta_rule | 0.251 / 0.251 | 0.229 / 0.229 | 0.000 / 1.000 | 0.991 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 50.42 |
| 18,000 | batch_ls | 1.000 / 1.000 | 0.967 / 0.967 | 0.045 / 1.000 | 0.973 / 0.000 | 0.000 / 0.000 | 0.8482 | 94 | 22.29 |
| 18,000 | joint_ls | 0.926 / 0.927 | 0.873 / 0.873 | 0.136 / 1.000 | 0.975 / 0.000 | 0.000 / 0.000 | 0.8848 | 420 | 134.43 |
| 21,000 | delta_rule | 0.169 / 0.169 | 0.162 / 0.162 | 0.023 / 1.000 | 0.991 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 57.47 |
| 21,000 | batch_ls | 1.000 / 1.000 | 0.967 / 0.967 | 0.068 / 1.000 | 0.971 / 0.000 | 0.000 / 0.000 | 0.8732 | 119 | 24.73 |
| 21,000 | joint_ls | 0.904 / 0.904 | 0.842 / 0.842 | 0.204 / 1.000 | 0.978 / 0.000 | 0.000 / 0.000 | 0.9219 | 439 | 120.22 |
| 24,000 | delta_rule | 0.158 / 0.158 | 0.146 / 0.146 | 0.000 / 1.000 | 0.999 / 0.000 | 0.000 / 0.000 | 1.0000 | - | 58.96 |
| 24,000 | batch_ls | 1.000 / 1.000 | 0.961 / 0.961 | 0.114 / 1.000 | 0.973 / 0.000 | 0.000 / 0.000 | 0.8867 | 106 | 23.93 |
| 24,000 | joint_ls | 0.895 / 0.895 | 0.811 / 0.811 | 0.114 / 1.000 | 0.976 / 0.000 | 0.000 / 0.000 | 0.9460 | 401 | 118.23 |

Known facts the base model answers correctly: 44.

## Deletion

```
{
 "deleted_facts": 250,
 "max_abs_diff_vs_never_learned": 0.2124614715576172,
 "max_abs_value": 119.60041046142578,
 "deleted_before": {
  "train_gate": 0.861,
  "test_gate": 0.784
 },
 "deleted_after": {
  "train_gate": 0.069,
  "test_gate": 0.072
 },
 "deleted_base_no_memory": {
  "train_none": 0.022,
  "test_none": 0.02
 },
 "kept_first_night_before": {
  "train_gate": 0.875,
  "test_gate": 0.79
 },
 "kept_first_night_after": {
  "train_gate": 0.884,
  "test_gate": 0.796
 },
 "resolve_iters_warm_cold": [
  513,
  700
 ]
}
```

## Forward-only writes (per-answer codebook, first-night facts)

```
{
 "answers_missing_from_codebook": 0,
 "cos_with_gradient_targets": 0.91,
 "gradient": {
  "train_oracle": 1.0,
  "test_oracle": 1.0,
  "train_gate": 1.0,
  "test_gate": 0.964
 },
 "codebook": {
  "train_oracle": 0.9995,
  "test_oracle": 1.0,
  "train_gate": 0.9995,
  "test_gate": 0.965
 }
}
```

## Context vs memory (same unseen-wording questions)

```
{
 "queries": 100,
 "no_context_no_memory": 0.01,
 "memory_all_24k_facts_no_context": 0.83,
 "N=10": {
  "all_facts_in_context": 0.43,
  "best_match_selected_into_context": 0.92,
  "context_tokens": 103
 },
 "N=100": {
  "all_facts_in_context": 0.27,
  "best_match_selected_into_context": 0.92,
  "context_tokens": 974
 },
 "N=1000": {
  "all_facts_in_context": 0.13,
  "best_match_selected_into_context": 0.92,
  "context_tokens": 9633
 }
}
```

Total time: 62.5 min
