# Real-world test 4: MQuAKE lifelong editing — Qwen/Qwen2.5-0.5B (MQuAKE-CF-3k-v2.json)

2,764 distinct edits from 3,000 cases, written 500 per night; memory after block 14 of 24 (value added at: all position(s)); calibration on 556 disjoint cases.

At question time nothing is given: the subject is found by matching every word span, the relation by a classifier.

## Every edit after the last night (new answer generated; string match against answer + aliases)

| Method | Cloze (written form) | Question form |
|---|---|---|
| base | 0.024 | 0.007 |
| joint_ls | 0.973 | 0.910 |
| batch_ls | 0.941 | 0.877 |
| grace | 0.640 | 0.547 |
| rag | 0.902 | 0.109 |

Lookup found the right fact from the question alone: 0.910

## First night's edits after each night (cloze)

| Edits | joint_ls | batch_ls | grace |
|---|---|---|---|
| 500 | 0.987 | 0.983 | 0.583 |
| 1,000 | 0.980 | 0.957 | 0.583 |
| 1,500 | 0.977 | 0.953 | 0.583 |
| 2,000 | 0.973 | 0.943 | 0.583 |
| 2,500 | 0.973 | 0.933 | 0.583 |
| 2,764 | 0.970 | 0.927 | 0.583 |

## Locality (unedited facts; identical output to the unedited model)

```
{
 "unedited_facts": 1087,
 "of_which_about_edited_subjects": 55,
 "joint_ls": {
  "unchanged": 0.9687,
  "unchanged_same_subject_other_relation": 0.6545,
  "known_facts_correct": 0.9412
 },
 "batch_ls": {
  "unchanged": 0.9687,
  "unchanged_same_subject_other_relation": 0.6545,
  "known_facts_correct": 0.9412
 },
 "grace": {
  "unchanged": 0.8482,
  "unchanged_same_subject_other_relation": 0.5273,
  "known_facts_correct": 0.9412
 },
 "rag": {
  "unchanged": 0.9163,
  "unchanged_same_subject_other_relation": 0.6,
  "known_facts_correct": 0.9412
 },
 "base_accuracy_on_unedited": 0.081
}
```

## Multi-hop (MQuAKE: a case counts if any of its questions is answered with the new answer)

```
{
 "cases": 200,
 "questions_per_case": 1,
 "base": {
  "chain_case_accuracy": 0.005,
  "chain_question_accuracy": 0.005,
  "direct_case_accuracy": 0.005,
  "chain_case_accuracy_on_ORIGINAL_answers": 0.03
 },
 "joint_ls": {
  "chain_case_accuracy": 0.03,
  "chain_question_accuracy": 0.03,
  "chain_gave_old_answer": 0.025,
  "direct_case_accuracy": 0.005
 }
}
```

Published on GPT-J, MQuAKE-CF, 3,000 edits (Zhong et al. 2023): MeLLo 14.2%, MEMIT 5.4% multi-hop accuracy.

Setup: {"person_key": {"block": 3, "theta": 0.98472, "same_name_min_cos": 0.98978, "different_names_max_cos": 0.97967, "calibration_names": 1426, "different_names_passing": 0.0}, "memory_block": {"block": 14, "inject": "all", "candidates": {"8/last": {"cloze": 0.938, "question_transfer": 0.292}, "8/all": {"cloze": 0.938, "question_transfer": 0.354}, "11/last": {"cloze": 0.938, "question_transfer": 0.396}, "11/all": {"cloze": 0.938, "question_transfer": 0.375}, "14/last": {"cloze": 0.938, "question_transfer": 0.375}, "14/all": {"cloze": 0.938, "question_transfer": 0.771}}}, "relation_classifier": {"relations": 36, "training_texts": 2246, "calibration_edit_accuracy": 1.0}, "grace_theta": 0.997, "known_facts_base_correct": 51}

First night, before editing / target injected directly: {"base": {"cloze": 0.0267, "question": 0.0067}} / {"cloze": 0.9967, "question": 0.9967}

Total time: 37.4 min
