# Real-world test 4: MQuAKE lifelong editing — /vol/models/gpt-j-6b-fp16 (MQuAKE-T.json)

96 distinct edits from 1,868 cases, written 500 per night; memory after block 17 of 28 (value added at: all position(s)); calibration on 556 disjoint cases. Elapsed: 8.5 min.

At question time nothing is given: the subject is found by matching every word span, the relation by a classifier.

## Every edit after the last night (new answer generated; string match against answer + aliases)

| Method | Cloze (written form) | Question form |
|---|---|---|
| base | 0.000 | 0.000 |
| joint_ls | 0.958 | 0.958 |
| batch_ls | 0.938 | 0.896 |
| grace | 0.521 | 0.500 |
| rag | 0.948 | 0.885 |

Lookup found the right fact from the question alone: 0.979

## First night's edits after each night (cloze)

| Edits | joint_ls | batch_ls | grace | Night (s) |
|---|---|---|---|---|
| 96 | 0.958 | 0.938 | 0.521 | 8 |

## Locality (unedited facts; identical output to the unedited model)

```
{
 "unedited_facts": 1813,
 "of_which_about_edited_subjects": 0,
 "excluded_edited_by_another_case": 0,
 "lookup_matched_a_fact_unedited": 0.0,
 "known_facts_subject_edited_by_mquake": 9,
 "lookup_matched_a_fact_known": 0.0,
 "known_facts_base_correct_same_batches": 1.0,
 "joint_ls": {
  "unchanged": 1.0,
  "unchanged_same_subject_other_relation": null,
  "known_facts_correct": 1.0,
  "known_facts_unchanged": 1.0,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "batch_ls": {
  "unchanged": 1.0,
  "unchanged_same_subject_other_relation": null,
  "known_facts_correct": 1.0,
  "known_facts_unchanged": 1.0,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "grace": {
  "unchanged": 1.0,
  "unchanged_same_subject_other_relation": null,
  "known_facts_correct": 1.0,
  "known_facts_unchanged": 1.0,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "rag": {
  "unchanged": 1.0,
  "unchanged_same_subject_other_relation": null,
  "known_facts_correct": 1.0,
  "known_facts_unchanged": 1.0,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "base_accuracy_on_unedited": 0.6426
}
```

### What joint_ls changed (0 known facts, 0 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|

### What batch_ls changed (0 known facts, 0 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|

### What grace changed (0 known facts, 0 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|

### What rag changed (0 known facts, 0 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|

Known facts the unedited model got wrong (left out of the check): 'The capital of France is' → 'a city of contrasts. It'; 'The capital of Italy is' → 'a city of many faces.'; 'The capital of Germany is' → 'a city of contrasts. It'; 'The capital of Spain is' → 'a city of great beauty and'; 'The capital of England is' → 'a city of great beauty and'; 'The capital of Canada is' → 'a city of contrasts. It'; 'The capital of China is' → 'a city of contrasts. It'; 'The Eiffel Tower is located in' → 'the 7th arrondisse'; 'Big Ben is located in' → 'the heart of the city,'; 'The largest planet in the solar system is' → 'also the most mysterious. It'; 'Most people in Germany speak' → 'English, but there are still'; 'Ripe bananas are' → 'a great source of potassium,'; 'Fresh grass is' → 'a great way to get your'; 'Snow is' → 'falling in the mountains of the'; 'Doctors usually work in a' → 'team, and the team leader'

## Multi-hop (MQuAKE: a case counts if any of its questions is answered with the new answer)

750 of 1,868 cases (random order), 3 question(s) each — stopped by the time budget.

| Method | Chain: new answer [95% CI] | Direct question: new answer | Chain gave the old answer | Chain by hops (2 / 3 / 4) |
|---|---|---|---|---|
| joint_ls | 18.8% [16.2–21.8] | 0.7% | 16.5% | 23.4% / 4.4% / 0.0% |
| rag | 13.5% [11.2–16.1] | 0.7% | 17.9% | 16.9% / 2.8% / 0.0% |
| base | 0.4% [0.1–1.2] | 0.7% | 22.9% | 0.5% / 0.0% / 0.0% |

Unedited model, same chain, ORIGINAL answers: 22.9% [20.1–26.1]; by hops 27.6% / 8.3% / 0.0%.
Joint memory's new answers per original answer of the unedited model: 0.82.

Published on GPT-J, same data (MQuAKE-T.json), 1,868 edited instances: MeLLo 30.7%, MEND 4.6%, MEMIT 0.0%.

Example (joint_ls): new answer 'Seyed Ebrahim Raisi', original 'Hassan Rouhani', final 'The head of government of the country where Farah Pahl'
```
Question: Who is the head of government of the country where Farah Pahlavi holds citizenship?
Subquestion: Who is the head of government of the country where Farah Pahlavi holds citizenship?
Answer: The head of government of the country where Farah Pahl
```

Example (joint_ls): new answer 'Bruce Harrell', original 'Jenny Durkan', final 'The Shuttle Carrier Aircraft was formed at the NASA Dryden Flight'
```
Question: Who is the head of government of the location where the Shuttle Carrier Aircraft was formed by its developer?
Subquestion: What is the location where the Shuttle Carrier Aircraft was formed by its developer?
Answer: The Shuttle Carrier Aircraft was formed at the NASA Dryden Flight
```

Example (joint_ls): new answer 'Yoon Suk-yeol', original 'Moon Jae-in', final 'Yoon Suk-yeol'
```
Question: Who was the head of state of the country where Chun Doo-hwan holds citizenship?
Subquestion: Who was the head of state of the country where Chun Doo-hwan holds citizenship?
Answer: Park Chung-hee, the president of South Korea from 1961
Subquestion: What is the name of the current head of state in South Korea?
Answer: Yoon Suk-yeol
```

Example (joint_ls): new answer 'Amarjeet Sohi', original 'Don Iveson', final 'The head of government of Alberta is the Lieutenant Governor of Alberta'
```
Question: Who is the head of government of Alberta's capital?
Subquestion: What is the capital of Alberta?
Answer: Edmonton
Subquestion: Who is the head of government of Alberta?
Answer: The head of government of Alberta is the Lieutenant Governor of Alberta
```

Setup: {"person_key": {"block": 8, "theta": 0.99711, "same_name_min_cos": 0.9988, "different_names_max_cos": 0.99542, "calibration_names": 1426, "different_names_passing": 0.0}, "memory_block": {"block": 17, "inject": "all", "candidates": {"9/last": {"cloze": 0.917, "question_transfer": 0.021}, "9/all": {"cloze": 0.938, "question_transfer": 0.042}, "13/last": {"cloze": 0.938, "question_transfer": 0.021}, "13/all": {"cloze": 0.938, "question_transfer": 0.125}, "17/last": {"cloze": 0.938, "question_transfer": 0.104}, "17/all": {"cloze": 0.938, "question_transfer": 0.312}}}, "relation_classifier": {"relations": 36, "training_texts": 2246, "calibration_edit_accuracy": 1.0}, "grace_theta": 0.9703, "known_facts_base_correct": 54}

First night, before editing / target injected directly: {"base": {"cloze": 0.0, "question": 0.0}} / {"cloze": 0.9792, "question": 0.9792}

Total time: 8.5 min
