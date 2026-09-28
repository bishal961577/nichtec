# Real-world test 4: MQuAKE lifelong editing — Qwen/Qwen2.5-0.5B (MQuAKE-CF-3k-v2.json)

2,764 distinct edits from 3,000 cases, written 500 per night; memory after block 14 of 24 (value added at: all position(s)); calibration on 44 disjoint cases.

At question time nothing is given: the subject is found by matching every word span, the relation by a classifier.

## Every edit after the last night (new answer generated; string match against answer + aliases)

| Method | Cloze (written form) | Question form |
|---|---|---|
| base | 0.024 | 0.007 |
| joint_ls | 0.953 | 0.804 |
| batch_ls | 0.932 | 0.781 |
| grace | 0.640 | 0.547 |
| rag | 0.885 | 0.097 |

Lookup found the right fact from the question alone: 0.800

## First night's edits after each night (cloze)

| Edits | joint_ls | batch_ls | grace |
|---|---|---|---|
| 500 | 0.973 | 0.967 | 0.583 |
| 1,000 | 0.970 | 0.957 | 0.583 |
| 1,500 | 0.967 | 0.943 | 0.583 |
| 2,000 | 0.967 | 0.930 | 0.583 |
| 2,500 | 0.963 | 0.907 | 0.583 |
| 2,764 | 0.963 | 0.900 | 0.583 |

## Locality (unedited facts; identical output to the unedited model)

```
{
 "unedited_facts": 1087,
 "of_which_about_edited_subjects": 55,
 "joint_ls": {
  "unchanged": 0.9457,
  "unchanged_same_subject_other_relation": 0.7273,
  "known_facts_correct": 0.8627
 },
 "batch_ls": {
  "unchanged": 0.9457,
  "unchanged_same_subject_other_relation": 0.7273,
  "known_facts_correct": 0.8627
 },
 "grace": {
  "unchanged": 0.3082,
  "unchanged_same_subject_other_relation": 0.2182,
  "known_facts_correct": 0.8235
 },
 "rag": {
  "unchanged": 0.839,
  "unchanged_same_subject_other_relation": 0.6182,
  "known_facts_correct": 0.9804
 },
 "base_accuracy_on_unedited": 0.081
}
```

## Multi-hop (MQuAKE: a case counts if any of its questions is answered with the new answer)

```
{
 "cases": 1000,
 "questions_per_case": 1,
 "base": {
  "chain_case_accuracy": 0.004,
  "chain_question_accuracy": 0.004,
  "direct_case_accuracy": 0.01,
  "chain_case_accuracy_on_ORIGINAL_answers": 0.027
 },
 "joint_ls": {
  "chain_case_accuracy": 0.009,
  "chain_question_accuracy": 0.009,
  "chain_gave_old_answer": 0.02,
  "direct_case_accuracy": 0.01
 },
 "batch_ls": {
  "chain_case_accuracy": 0.009,
  "chain_question_accuracy": 0.009,
  "chain_gave_old_answer": 0.021,
  "direct_case_accuracy": 0.01
 },
 "grace": {
  "chain_case_accuracy": 0.005,
  "chain_question_accuracy": 0.005,
  "chain_gave_old_answer": 0.016,
  "direct_case_accuracy": 0.01
 },
 "rag": {
  "chain_case_accuracy": 0.004,
  "chain_question_accuracy": 0.004,
  "chain_gave_old_answer": 0.018,
  "direct_case_accuracy": 0.01
 }
}
```

Published on GPT-J, MQuAKE-CF, 3,000 edits (Zhong et al. 2023): MeLLo 14.2%, MEMIT 5.4% multi-hop accuracy.

Setup: {"person_key": {"block": 7, "theta": 0.90494, "same_name_min_cos": 0.99523, "different_names_max_cos": 0.81464, "calibration_names": 143, "different_names_passing": 0.0}, "memory_block": {"block": 14, "inject": "all", "candidates": {"8/last": {"cloze": 1.0, "question_transfer": 0.208}, "8/all": {"cloze": 1.0, "question_transfer": 0.354}, "11/last": {"cloze": 1.0, "question_transfer": 0.312}, "11/all": {"cloze": 1.0, "question_transfer": 0.417}, "14/last": {"cloze": 1.0, "question_transfer": 0.375}, "14/all": {"cloze": 1.0, "question_transfer": 0.792}}}, "relation_classifier": {"relations": 24, "training_texts": 180, "calibration_edit_accuracy": 1.0}, "grace_theta": 0.9911, "known_facts_base_correct": 51}

First night, before editing / target injected directly: {"base": {"cloze": 0.0267, "question": 0.0067}} / {"cloze": 0.9967, "question": 0.9967}

Total time: 73.5 min
