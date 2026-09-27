# Real-LM memory test 2 (person x relation keys): Qwen/Qwen2.5-0.5B

Memory after block 14 of 24, 65,536 slots x 896 dims; 24,000 facts in nights of 3,000; 4 wordings written per fact. Person key: block3_lda. Relation discriminant ratios: [3603.0, 2083.3, 1216.1, 868.7, 802.4, 653.6, 332.6].

No memory / target injected directly (first-night facts): {"train_none": 0.0245, "train_oracle": 1.0, "test_none": 0.02, "test_oracle": 1.0}

Gate: {"theta": 0.995, "calib_true_pass": 1.0, "calib_false_pass": 0.3171}; unseen wordings of written people pass: 0.864; never-written people pass (night 1): 0.2

Key overlap per candidate: {"block3_lda": {"written_wording_left_out_slots_found": 0.959, "unseen_wording_slots_found_in_written": 0.695, "different_facts_shared_slots_same_relation": 0.035, "different_facts_shared_slots_other_relation": 0.002, "score": 0.924}, "block3_lda_eq": {"written_wording_left_out_slots_found": 0.946, "unseen_wording_slots_found_in_written": 0.688, "different_facts_shared_slots_same_relation": 0.029, "different_facts_shared_slots_other_relation": 0.001, "score": 0.917}, "block5_lda": {"written_wording_left_out_slots_found": 0.945, "unseen_wording_slots_found_in_written": 0.688, "different_facts_shared_slots_same_relation": 0.034, "different_facts_shared_slots_other_relation": 0.002, "score": 0.911}, "block5_lda_eq": {"written_wording_left_out_slots_found": 0.919, "unseen_wording_slots_found_in_written": 0.678, "different_facts_shared_slots_same_relation": 0.024, "different_facts_shared_slots_other_relation": 0.001, "score": 0.895}, "block7_lda": {"written_wording_left_out_slots_found": 0.945, "unseen_wording_slots_found_in_written": 0.687, "different_facts_shared_slots_same_relation": 0.036, "different_facts_shared_slots_other_relation": 0.002, "score": 0.909}, "block7_lda_eq": {"written_wording_left_out_slots_found": 0.91, "unseen_wording_slots_found_in_written": 0.66, "different_facts_shared_slots_same_relation": 0.027, "different_facts_shared_slots_other_relation": 0.002, "score": 0.883}, "block9_lda": {"written_wording_left_out_slots_found": 0.945, "unseen_wording_slots_found_in_written": 0.686, "different_facts_shared_slots_same_relation": 0.033, "different_facts_shared_slots_other_relation": 0.002, "score": 0.912}, "block9_lda_eq": {"written_wording_left_out_slots_found": 0.916, "unseen_wording_slots_found_in_written": 0.655, "different_facts_shared_slots_same_relation": 0.027, "different_facts_shared_slots_other_relation": 0.002, "score": 0.889}}

## First night's facts, top-1 accuracy after each night (gate off / gate on)

| Facts | Constraints/slot | Distinct slots | delta_rule written | batch_ls written | joint_ls written | delta_rule unseen | batch_ls unseen | joint_ls unseen |
|---|---|---|---|---|---|---|---|---|
| 3,000 | 0.18 | 15,282 | 0.068 / 0.068 | 0.855 / 0.855 | 0.962 / 0.962 | 0.059 / 0.052 | 0.497 / 0.419 | 0.437 / 0.364 |
| 6,000 | 0.37 | 17,636 | 0.037 / 0.037 | 0.136 / 0.136 | 0.896 / 0.896 | 0.032 / 0.030 | 0.094 / 0.084 | 0.280 / 0.239 |
| 9,000 | 0.55 | 18,764 | 0.043 / 0.043 | 0.110 / 0.110 | 0.808 / 0.808 | 0.041 / 0.039 | 0.066 / 0.062 | 0.199 / 0.180 |
| 12,000 | 0.73 | 19,528 | 0.013 / 0.013 | 0.101 / 0.101 | 0.720 / 0.720 | 0.011 / 0.010 | 0.064 / 0.059 | 0.187 / 0.169 |
| 15,000 | 0.92 | 20,028 | 0.029 / 0.029 | 0.097 / 0.097 | 0.666 / 0.666 | 0.032 / 0.031 | 0.059 / 0.057 | 0.159 / 0.146 |
| 18,000 | 1.10 | 20,342 | 0.036 / 0.036 | 0.087 / 0.087 | 0.604 / 0.604 | 0.035 / 0.033 | 0.057 / 0.052 | 0.133 / 0.124 |
| 21,000 | 1.28 | 20,646 | 0.033 / 0.033 | 0.070 / 0.070 | 0.560 / 0.560 | 0.030 / 0.028 | 0.050 / 0.047 | 0.119 / 0.109 |
| 24,000 | 1.47 | 20,902 | 0.035 / 0.035 | 0.093 / 0.093 | 0.518 / 0.518 | 0.031 / 0.030 | 0.054 / 0.054 | 0.111 / 0.102 |

## Latest night, damage, certificate (gate off / gate on)

| Facts | Method | Latest written | Latest unseen | Known facts | Never-written people: answer changed | >20% error | Iters | Max value |
|---|---|---|---|---|---|---|---|---|
| 3,000 | delta_rule | 0.294 / 0.294 | 0.267 / 0.233 | 0.295 / 1.000 | 0.988 / 0.195 | 1.0000 | - | 40.15 |
| 3,000 | batch_ls | 0.852 / 0.852 | 0.499 / 0.421 | 0.273 / 1.000 | 0.974 / 0.191 | 0.9571 | 128 | 22.91 |
| 3,000 | joint_ls | 0.970 / 0.970 | 0.449 / 0.368 | 0.068 / 1.000 | 0.964 / 0.189 | 0.7286 | 300 | 60.15 |
| 6,000 | delta_rule | 0.225 / 0.225 | 0.202 / 0.186 | 0.204 / 1.000 | 0.995 / 0.220 | 1.0000 | - | 42.98 |
| 6,000 | batch_ls | 0.854 / 0.854 | 0.498 / 0.453 | 0.341 / 1.000 | 0.976 / 0.219 | 0.9821 | 125 | 30.34 |
| 6,000 | joint_ls | 0.887 / 0.887 | 0.297 / 0.279 | 0.023 / 1.000 | 0.975 / 0.212 | 0.9575 | 300 | 77.23 |
| 9,000 | delta_rule | 0.255 / 0.255 | 0.223 / 0.205 | 0.204 / 1.000 | 0.985 / 0.239 | 1.0000 | - | 45.69 |
| 9,000 | batch_ls | 0.868 / 0.868 | 0.481 / 0.436 | 0.250 / 1.000 | 0.976 / 0.234 | 0.9906 | 126 | 28.99 |
| 9,000 | joint_ls | 0.809 / 0.809 | 0.237 / 0.218 | 0.023 / 1.000 | 0.971 / 0.231 | 0.9883 | 300 | 78.55 |
| 12,000 | delta_rule | 0.239 / 0.239 | 0.199 / 0.181 | 0.159 / 1.000 | 0.996 / 0.240 | 1.0000 | - | 44.39 |
| 12,000 | batch_ls | 0.867 / 0.867 | 0.484 / 0.434 | 0.250 / 1.000 | 0.979 / 0.235 | 0.9939 | 129 | 30.87 |
| 12,000 | joint_ls | 0.734 / 0.734 | 0.186 / 0.173 | 0.045 / 1.000 | 0.963 / 0.230 | 0.9956 | 300 | 83.62 |
| 15,000 | delta_rule | 0.213 / 0.213 | 0.165 / 0.159 | 0.136 / 1.000 | 0.994 / 0.240 | 1.0000 | - | 44.99 |
| 15,000 | batch_ls | 0.839 / 0.839 | 0.492 / 0.454 | 0.182 / 1.000 | 0.970 / 0.237 | 0.9945 | 133 | 32.05 |
| 15,000 | joint_ls | 0.670 / 0.670 | 0.173 / 0.167 | 0.045 / 1.000 | 0.963 / 0.231 | 0.9977 | 300 | 86.37 |
| 18,000 | delta_rule | 0.205 / 0.205 | 0.163 / 0.150 | 0.114 / 1.000 | 0.993 / 0.240 | 1.0000 | - | 44.82 |
| 18,000 | batch_ls | 0.841 / 0.841 | 0.474 / 0.439 | 0.250 / 1.000 | 0.974 / 0.236 | 0.9952 | 131 | 36.5 |
| 18,000 | joint_ls | 0.593 / 0.593 | 0.183 / 0.173 | 0.045 / 1.000 | 0.961 / 0.235 | 0.9987 | 300 | 93.03 |
| 21,000 | delta_rule | 0.202 / 0.202 | 0.175 / 0.166 | 0.204 / 1.000 | 0.994 / 0.251 | 1.0000 | - | 44.0 |
| 21,000 | batch_ls | 0.835 / 0.835 | 0.460 / 0.424 | 0.227 / 1.000 | 0.970 / 0.242 | 0.9962 | 131 | 31.35 |
| 21,000 | joint_ls | 0.549 / 0.549 | 0.147 / 0.136 | 0.045 / 1.000 | 0.958 / 0.240 | 0.9991 | 300 | 88.6 |
| 24,000 | delta_rule | 0.208 / 0.208 | 0.156 / 0.147 | 0.182 / 1.000 | 0.991 / 0.264 | 1.0000 | - | 50.17 |
| 24,000 | batch_ls | 0.857 / 0.857 | 0.443 / 0.414 | 0.227 / 1.000 | 0.964 / 0.256 | 0.9966 | 133 | 33.89 |
| 24,000 | joint_ls | 0.500 / 0.500 | 0.127 / 0.119 | 0.068 / 1.000 | 0.960 / 0.259 | 0.9993 | 300 | 95.61 |

Known facts the base model answers correctly: 44. Never-written people passing the gate after the last night: 0.27

## Deletion

```
{
 "deleted_facts": 250,
 "max_abs_diff_vs_never_learned": 1.28521728515625,
 "max_abs_value": 94.988037109375,
 "deleted_before": {
  "train_off": 0.53,
  "test_off": 0.102
 },
 "deleted_after": {
  "train_off": 0.078,
  "test_off": 0.042
 },
 "deleted_base_no_memory": {
  "train_none": 0.022,
  "test_none": 0.02
 },
 "kept_first_night_before": {
  "train_off": 0.494,
  "test_off": 0.122
 },
 "kept_first_night_after": {
  "train_off": 0.495,
  "test_off": 0.124
 },
 "resolve_iters_warm_cold": [
  600,
  600
 ]
}
```

## Targets without a backward pass on the device (first-night facts, joint solve)

```
{
 "logit_lens_scale_search": {
  "0.25": 0.035,
  "0.5": 0.055,
  "1": 0.1013,
  "2": 0.1462,
  "4": 0.2212
 },
 "logit_lens_scale": 4,
 "codebook_answers_missing": 0,
 "codebook_vs_gradient_cos": 0.91,
 "gradient (backward pass per fact)": {
  "train_oracle": 1.0,
  "test_oracle": 1.0,
  "train_off": 0.962,
  "test_off": 0.436
 },
 "logit-lens direction (no backward pass)": {
  "train_oracle": 0.217,
  "test_oracle": 0.219,
  "train_off": 0.174,
  "test_off": 0.11
 },
 "answer codebook (offline, no backward pass on device)": {
  "train_oracle": 0.9995,
  "test_oracle": 1.0,
  "train_off": 0.96,
  "test_off": 0.458
 }
}
```

Total time: 57.4 min
