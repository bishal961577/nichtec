# Ingredients test 1: the model names traits, code combines them

Model Qwen3.5-27B-FP8. Pre-registration: section 7 of reports/Learning on the job from first principles 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over the 20 tasks (20 yes, 20 no each).

| Agent | Accuracy |
|---|---|
| A, full history | 86.2 +/- 1.5 |
| W, whole rule in words, judged by code | 83.6 +/- 2.2 |
| **E, model names traits, code combines** | **89.4 +/- 1.7** |
| E with the 12 true questions (reference) | 93.2 +/- 1.3 |
| Oracle: recorded traits, same combining | 99.2 |

Reading accuracy on the 12 true questions over 99 animals: 95.4% (hair 95, feathers 100, eggs 95, milk 96, airborne 97, aquatic 94, toothed 90, backbone 98, breathes 96, venomous 94, fins 99, tail 91).
Tasks where E's rule uses a question that matches one of the hidden rule's traits: 80%. Answer calls: 16,786; neither yes nor no: 0.0%.

| Task | Hidden rule | A | W | E | E true q. | Oracle | E's rule | W's rule |
|---|---|---|---|---|---|---|---|---|
| rule1_draw1 | airborne or aquatic | 95 | 95 | 90 | 90 | 100 | [Can it fly?] or [Does it live in water?] | The animal is classified as "yes" if it can fly or live in water, and "no" other |
| rule1_draw2 | airborne or aquatic | 90 | 95 | 92 | 100 | 100 | [Does it have wings?] or [Does the animal live in water?] | The animal must be capable of flying or swimming. |
| rule2_draw1 | eggs and backbone | 95 | 65 | 98 | 92 | 100 | [Does the animal have a backbone?] and not [Is it a mammal?] | The animal is a vertebrate that breathes with gills at some stage of its life or |
| rule2_draw2 | eggs and backbone | 85 | 90 | 95 | 92 | 100 | [Does it have a backbone?] and [Does the animal lay eggs?] | The animal is classified as "yes" if it is a bird or a fish. |
| rule3_draw1 | airborne or fins | 90 | 82 | 88 | 95 | 100 | [Can it fly?] or [Is it a fish?] | Yes animals are capable of flying or swimming, while no animals are only capable |
| rule3_draw2 | airborne or fins | 80 | 75 | 80 | 98 | 100 | [Does the animal have wings?] or [Is the animal a fish?] | The animal is capable of flight. |
| rule4_draw1 | not airborne and breathes | 85 | 82 | 88 | 98 | 100 | [Does the animal breathe air through lungs?] and not [Is the animal a bird?] | The rule is that the animal is a mammal or a reptile. |
| rule4_draw2 | not airborne and breathes | 95 | 85 | 88 | 78 | 100 | [Does the animal breathe air through lungs?] and not [Does the animal have feathers?] | The animal is classified as "yes" if it is a mammal, a reptile, or a slug, and " |
| rule5_draw1 | fins or not tail | 85 | 90 | 80 | 88 | 100 | [Does the animal have six legs?] or [Does the animal live primarily in water?] | The animal is classified as yes if it is not a bird, mammal, or reptile. |
| rule5_draw2 | fins or not tail | 80 | 88 | 85 | 95 | 100 | [Does the animal live in water?] or [Is it an insect?] | The animal is classified as "yes" if it is an invertebrate or a fish, and "no" i |
| rule6_draw1 | milk or not backbone | 85 | 98 | 100 | 100 | 100 | not [Does the animal have a backbone?] or [Is the animal a mammal?] | The animal is classified as "yes" if it is a mammal or an invertebrate, and "no" |
| rule6_draw2 | milk or not backbone | 85 | 90 | 98 | 100 | 98 | [Does the animal have fur or hair?] or not [Is the animal a vertebrate?] | The animal is a mammal or an insect. |
| rule7_draw1 | not milk and tail | 95 | 95 | 98 | 98 | 95 | [Does the animal have a backbone?] and not [Is it a mammal?] | The animal is a bird, a reptile, or a fish. |
| rule7_draw2 | not milk and tail | 90 | 90 | 88 | 98 | 92 | [Does the animal belong to the class Aves?] or [Is it a fish?] | The animal is in the "yes" group if it is a vertebrate that is either a bird or  |
| rule8_draw1 | airborne or not tail | 80 | 75 | 72 | 90 | 100 | [Does the animal fly?] or [Is the animal a mollusk?] | The animal is in the "yes" group if it can fly or if it is a cavy. |
| rule8_draw2 | airborne or not tail | 75 | 75 | 92 | 90 | 100 | [Can the animal fly?] or not [Is the animal a vertebrate?] | The animal is classified as "yes" if it has wings or can fly. |
| rule9_draw1 | not aquatic and tail | 70 | 88 | 75 | 90 | 100 | [Does the animal live primarily on land?] | The animal is a terrestrial or flying vertebrate. |
| rule9_draw2 | not aquatic and tail | 90 | 80 | 92 | 82 | 100 | [Does the animal live primarily on land?] and not [Is the animal an insect or crustacean?] | The animal is in the "yes" group if it is a mammal that moves primarily on land  |
| rule10_draw1 | feathers or not breathes | 85 | 68 | 92 | 98 | 100 | [Does the animal have feathers?] or [Does the animal live primarily in water?] | The animals must live in water. |
| rule10_draw2 | feathers or not breathes | 90 | 68 | 98 | 95 | 100 | [Does the animal have feathers?] or [Does the animal live primarily in water?] | The animal is classified as "yes" if it lives primarily in water. |

## Verdict

- Check 1, the task: oracle 99.2 (line >= 95): ok.
- Check 2, reading: 95.4% (line >= 85): ok.
- Check 3, answering: 0.0% of answers neither yes nor no (line <= 10): ok.
- IN BETWEEN: E 89.4, A 86.2, W 83.6. Report, diagnose, decide with the owner.

Check before the run: {'first_token': 'yes', 'answers_yes_no': True, 'margin': 0.169, 'questions_named': ['Does the animal live in water?', 'Is the animal a fish?', 'Is the animal a bird?', 'Is the animal an insect?', 'Is the animal a mammal?', 'Is the animal a reptile?', 'Is the animal an amphibian?', 'Is the animal an arthropod?']}
