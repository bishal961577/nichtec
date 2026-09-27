# Real-LM memory test: Qwen/Qwen2.5-0.5B

Memory after block 14 of 24, 65,536 slots x 896 dims; 24,000 facts in nights of 3,000; 4 wordings written per fact.

No memory / target injected directly (first-night facts): {"train_none": 0.0245, "train_oracle": 1.0, "test_none": 0.02, "test_oracle": 1.0}

Key map used: lda. Key overlap: {"lda": {"written_wording_left_out_slots_found": 0.894, "unseen_wording_slots_found_in_written": 0.25, "different_facts_shared_slots_same_relation": 0.605, "different_facts_shared_slots_other_relation": 0.004, "top_discriminant_ratios": [5902.31, 3380.19, 1933.53], "score": 0.289}, "lda_eq": {"written_wording_left_out_slots_found": 0.144, "unseen_wording_slots_found_in_written": 0.008, "different_facts_shared_slots_same_relation": 0.003, "different_facts_shared_slots_other_relation": 0.0, "top_discriminant_ratios": [5902.31, 3380.19, 1933.53], "score": 0.141}, "pca": {"written_wording_left_out_slots_found": 0.157, "unseen_wording_slots_found_in_written": 0.238, "different_facts_shared_slots_same_relation": 0.536, "different_facts_shared_slots_other_relation": 0.001, "top_discriminant_ratios": [126.22, 112.48, 103.3], "score": -0.379}}

## First night's facts: top-1 answer accuracy after each night

Read = last: the memory is read only at the question's last position. Read = all: read at every position.

| Facts | Constraints/slot | delta_rule written (last/all) | batch_ls written (last/all) | joint_ls written (last/all) | joint_null written (last/all) | delta_rule unseen (last/all) | batch_ls unseen (last/all) | joint_ls unseen (last/all) | joint_null unseen (last/all) |
|---|---|---|---|---|---|---|---|---|---|
| 3,000 | 0.18 | 0.068 / 0.068 | 0.275 / 0.273 | 0.359 / 0.345 | 0.352 / 0.340 | 0.067 / 0.065 | 0.054 / 0.057 | 0.060 / 0.053 | 0.053 / 0.050 |
| 6,000 | 0.37 | 0.077 / 0.078 | 0.095 / 0.097 | 0.240 / 0.231 | 0.233 / 0.232 | 0.059 / 0.056 | 0.050 / 0.049 | 0.044 / 0.039 | 0.037 / 0.037 |
| 9,000 | 0.55 | 0.075 / 0.075 | 0.084 / 0.085 | 0.199 / 0.198 | 0.194 / 0.189 | 0.059 / 0.057 | 0.056 / 0.056 | 0.043 / 0.043 | 0.045 / 0.041 |
| 12,000 | 0.73 | 0.096 / 0.096 | 0.077 / 0.077 | 0.183 / 0.176 | 0.178 / 0.177 | 0.076 / 0.076 | 0.046 / 0.051 | 0.044 / 0.041 | 0.041 / 0.039 |
| 15,000 | 0.92 | 0.075 / 0.076 | 0.083 / 0.078 | 0.157 / 0.148 | 0.154 / 0.148 | 0.045 / 0.041 | 0.050 / 0.051 | 0.050 / 0.046 | 0.041 / 0.044 |
| 18,000 | 1.10 | 0.070 / 0.070 | 0.091 / 0.087 | 0.149 / 0.144 | 0.144 / 0.147 | 0.053 / 0.048 | 0.047 / 0.050 | 0.048 / 0.041 | 0.045 / 0.042 |
| 21,000 | 1.28 | 0.082 / 0.082 | 0.074 / 0.077 | 0.134 / 0.131 | 0.135 / 0.133 | 0.050 / 0.052 | 0.040 / 0.047 | 0.049 / 0.051 | 0.044 / 0.041 |
| 24,000 | 1.47 | 0.082 / 0.082 | 0.088 / 0.087 | 0.132 / 0.122 | 0.128 / 0.122 | 0.059 / 0.060 | 0.049 / 0.050 | 0.051 / 0.047 | 0.051 / 0.047 |

## Latest night's facts, damage to known facts, unrelated text, certificate

| Facts | Method | Latest written (last/all) | Latest unseen (last/all) | Known facts | KL on unrelated text | Top-1 agreement | Constraints >20% error |
|---|---|---|---|---|---|---|---|
| 3,000 | delta_rule | 0.096 / 0.096 | 0.063 / 0.066 | 0.676 | 0.0391 | 0.927 | 1.0000 |
| 3,000 | batch_ls | 0.246 / 0.239 | 0.054 / 0.052 | 0.676 | 0.0569 | 0.906 | 1.0000 |
| 3,000 | joint_ls | 0.334 / 0.326 | 0.057 / 0.063 | 0.460 | 0.3655 | 0.770 | 0.9985 |
| 3,000 | joint_null | 0.325 / 0.319 | 0.048 / 0.050 | 0.460 | 0.1784 | 0.811 | 0.9989 |
| 6,000 | delta_rule | 0.081 / 0.079 | 0.057 / 0.057 | 0.649 | 0.0381 | 0.929 | 1.0000 |
| 6,000 | batch_ls | 0.251 / 0.247 | 0.061 / 0.067 | 0.622 | 0.0788 | 0.889 | 1.0000 |
| 6,000 | joint_ls | 0.229 / 0.220 | 0.047 / 0.045 | 0.270 | 0.4532 | 0.750 | 0.9990 |
| 6,000 | joint_null | 0.224 / 0.214 | 0.049 / 0.045 | 0.351 | 0.1854 | 0.812 | 0.9992 |
| 9,000 | delta_rule | 0.086 / 0.086 | 0.065 / 0.066 | 0.595 | 0.0391 | 0.915 | 1.0000 |
| 9,000 | batch_ls | 0.264 / 0.253 | 0.056 / 0.051 | 0.568 | 0.0938 | 0.866 | 1.0000 |
| 9,000 | joint_ls | 0.190 / 0.182 | 0.044 / 0.043 | 0.351 | 0.4882 | 0.739 | 0.9991 |
| 9,000 | joint_null | 0.189 / 0.175 | 0.043 / 0.043 | 0.405 | 0.2028 | 0.804 | 0.9994 |
| 12,000 | delta_rule | 0.095 / 0.095 | 0.071 / 0.070 | 0.730 | 0.0448 | 0.921 | 1.0000 |
| 12,000 | batch_ls | 0.249 / 0.247 | 0.058 / 0.056 | 0.513 | 0.1178 | 0.873 | 1.0000 |
| 12,000 | joint_ls | 0.158 / 0.150 | 0.041 / 0.041 | 0.324 | 0.5331 | 0.731 | 0.9994 |
| 12,000 | joint_null | 0.159 / 0.154 | 0.034 / 0.037 | 0.378 | 0.2185 | 0.780 | 0.9995 |
| 15,000 | delta_rule | 0.112 / 0.108 | 0.069 / 0.071 | 0.703 | 0.0476 | 0.921 | 1.0000 |
| 15,000 | batch_ls | 0.262 / 0.260 | 0.079 / 0.077 | 0.540 | 0.1155 | 0.861 | 1.0000 |
| 15,000 | joint_ls | 0.137 / 0.136 | 0.058 / 0.057 | 0.378 | 0.5245 | 0.730 | 0.9995 |
| 15,000 | joint_null | 0.138 / 0.137 | 0.057 / 0.056 | 0.432 | 0.2267 | 0.784 | 0.9997 |
| 18,000 | delta_rule | 0.086 / 0.086 | 0.068 / 0.066 | 0.676 | 0.0469 | 0.925 | 1.0000 |
| 18,000 | batch_ls | 0.238 / 0.234 | 0.052 / 0.058 | 0.486 | 0.1185 | 0.864 | 1.0000 |
| 18,000 | joint_ls | 0.132 / 0.141 | 0.046 / 0.049 | 0.378 | 0.5251 | 0.716 | 0.9995 |
| 18,000 | joint_null | 0.132 / 0.139 | 0.043 / 0.045 | 0.432 | 0.2240 | 0.780 | 0.9996 |
| 21,000 | delta_rule | 0.080 / 0.079 | 0.058 / 0.060 | 0.649 | 0.0567 | 0.915 | 1.0000 |
| 21,000 | batch_ls | 0.242 / 0.241 | 0.055 / 0.060 | 0.595 | 0.1202 | 0.861 | 1.0000 |
| 21,000 | joint_ls | 0.128 / 0.124 | 0.051 / 0.048 | 0.405 | 0.5248 | 0.710 | 0.9996 |
| 21,000 | joint_null | 0.126 / 0.118 | 0.044 / 0.043 | 0.460 | 0.2249 | 0.778 | 0.9997 |
| 24,000 | delta_rule | 0.086 / 0.087 | 0.071 / 0.069 | 0.649 | 0.0551 | 0.925 | 1.0000 |
| 24,000 | batch_ls | 0.249 / 0.248 | 0.059 / 0.059 | 0.568 | 0.1306 | 0.859 | 1.0000 |
| 24,000 | joint_ls | 0.121 / 0.111 | 0.049 / 0.047 | 0.378 | 0.5450 | 0.714 | 0.9998 |
| 24,000 | joint_null | 0.115 / 0.111 | 0.048 / 0.045 | 0.432 | 0.2298 | 0.764 | 0.9998 |

Known facts the base model answers correctly: 37

## Deletion

```
{
 "deleted_facts": 250,
 "max_abs_diff_vs_never_learned": 2.2560997009277344,
 "max_abs_value": 63.85539627075195,
 "deleted_before": {
  "train_last": 0.137,
  "test_last": 0.05
 },
 "deleted_after": {
  "train_last": 0.08,
  "test_last": 0.048
 },
 "deleted_base_no_memory": {
  "train_none": 0.022,
  "test_none": 0.02
 },
 "kept_first_night_after": {
  "train_last": 0.12,
  "test_last": 0.048
 },
 "kept_first_night_before": {
  "train_last": 0.122,
  "test_last": 0.052
 },
 "resolve_iters_warm_cold": [
  600,
  600
 ]
}
```

## Forward-only targets (Fix 4)

```
{
 "scale_search_on_calibration_facts": {
  "0.5": 0.03,
  "1": 0.0475,
  "2": 0.08,
  "4": 0.02,
  "8": 0.0,
  "16": 0.0
 },
 "scale": 2,
 "gradient_targets": {
  "train_oracle": 1.0,
  "test_oracle": 1.0,
  "train_last": 0.36,
  "train_all": 0.345,
  "test_last": 0.058,
  "test_all": 0.053
 },
 "forward_only_targets": {
  "train_oracle": 0.077,
  "test_oracle": 0.052,
  "train_last": 0.041,
  "train_all": 0.0365,
  "test_last": 0.034,
  "test_all": 0.032
 }
}
```

Total time: 52.5 min
