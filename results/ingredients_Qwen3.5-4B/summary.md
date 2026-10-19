# Ingredients test 2: the model names traits, code combines them

Model Qwen3.5-4B. Pre-registration: section 7 of reports/Learning on the job from first principles 2026.md.

Test-set accuracy (%) after the 40-case stream, averaged over the 20 tasks (20 yes, 20 no each).

| Agent | Accuracy |
|---|---|
| A, full history | 59.0 +/- 2.2 |
| W, whole rule in words, judged by code | 73.6 +/- 2.3 |
| **E, model names traits, code combines** | **87.4 +/- 2.1** |
| E with the 12 true questions (reference) | 88.9 +/- 1.5 |
| Oracle: recorded traits, same combining | 99.2 |

Reading accuracy on the 12 true questions over 99 animals: 94.0% (hair 94, feathers 97, eggs 95, milk 93, airborne 96, aquatic 94, toothed 91, backbone 99, breathes 92, venomous 98, fins 97, tail 83).
Tasks where E's rule uses a question that matches one of the hidden rule's traits: 70%. Answer calls: 16,582; neither yes nor no: 0.0%.

| Task | Hidden rule | A | W | E | E true q. | Oracle | E's rule | W's rule |
|---|---|---|---|---|---|---|---|---|
| rule1_draw1 | airborne or aquatic | 65 | 70 | 90 | 90 | 100 | [Does the animal fly?] or [Does the animal live in the ocean or freshwater?] | The rule is that animals that live in or on water are accepted, while those that |
| rule1_draw2 | airborne or aquatic | 60 | 72 | 98 | 98 | 100 | [Does the animal fly?] or [Does the animal live in water?] | The animal is classified as yes if it has a fin, tail, or is a bird. |
| rule2_draw1 | eggs and backbone | 75 | 95 | 98 | 95 | 100 | [Does the animal lay eggs?] and [Is the animal a vertebrate?] | The animals in the "yes" group are all birds or fish, while the "no" group consi |
| rule2_draw2 | eggs and backbone | 70 | 80 | 100 | 98 | 100 | not [Is the animal a mammal?] and [Is the animal a vertebrate?] | The animals are classified as yes if they live in water or are birds, and no oth |
| rule3_draw1 | airborne or fins | 70 | 90 | 80 | 90 | 100 | [Is the animal a bird?] or [Is the animal an insect or a fish?] | The animals are classified as yes if they have fins or wings, and no if they do  |
| rule3_draw2 | airborne or fins | 65 | 72 | 68 | 92 | 100 | not [Is the animal a reptile?] and not [Is the animal a terrestrial mammal?] | The hidden rule is whether the animal has the ability to fly. |
| rule4_draw1 | not airborne and breathes | 40 | 78 | 85 | 80 | 100 | [Is the animal a mammal?] or [Is the animal an amphibian or a reptile?] | The hidden rule is: The animal is a mammal. |
| rule4_draw2 | not airborne and breathes | 50 | 78 | 98 | 82 | 100 | not [Does the animal have wings?] and [Is the animal a land animal?] | The rule is that the animal must be a mammal. |
| rule5_draw1 | fins or not tail | 55 | 52 | 85 | 78 | 100 | not [Does the animal have legs?] or [Is the animal an invertebrate?] | The rule is whether the animal's name contains the letter 'a'. |
| rule5_draw2 | fins or not tail | 55 | 62 | 88 | 88 | 100 | [Does the animal breathe underwater?] or not [Does the animal have a backbone?] | The hidden rule is that the animal is labeled "yes" if it has fins or scales, or |
| rule6_draw1 | milk or not backbone | 60 | 82 | 100 | 100 | 100 | not [Does the animal have a backbone?] or [Is the animal a mammal?] | The animal is a mammal. |
| rule6_draw2 | milk or not backbone | 55 | 82 | 80 | 92 | 98 | [Does the animal have fur or hair?] or [Is the animal an insect?] | The animal is a mammal or has fur/hair. |
| rule7_draw1 | not milk and tail | 65 | 72 | 95 | 95 | 95 | not [Is the animal a mammal?] and not [Is the animal a small invertebrate?] | The animal is classified as yes if it is a bird or a reptile, and no otherwise. |
| rule7_draw2 | not milk and tail | 60 | 62 | 75 | 90 | 92 | [Does the animal have feathers?] or [Does the animal live in water?] | The hidden rule is that the animal must have feathers. |
| rule8_draw1 | airborne or not tail | 40 | 65 | 78 | 88 | 100 | not [If no, is the animal a fish?] and not [Is the animal a mammal?] | The animals have a shell or an exoskeleton. |
| rule8_draw2 | airborne or not tail | 55 | 75 | 88 | 90 | 100 | not [Does the animal have a backbone?] or [Does the animal have wings?] | The animals are classified as "yes" if they can fly or lay eggs, and "no" otherw |
| rule9_draw1 | not aquatic and tail | 65 | 65 | 78 | 75 | 100 | [Does the animal have a backbone?] and [Does the animal have legs?] | The animals are classified as "yes" if and only if they are mammals. |
| rule9_draw2 | not aquatic and tail | 45 | 85 | 85 | 85 | 100 | not [Does the animal live in water?] | The animals are classified as yes if they are warm-blooded, or no if they are co |
| rule10_draw1 | feathers or not breathes | 65 | 70 | 88 | 85 | 100 | [Does the animal live in water?] or [Is the animal a bird?] | The rule is that the animal is classified as "yes" if it has gills or can breath |
| rule10_draw2 | feathers or not breathes | 65 | 62 | 95 | 88 | 100 | [Does the animal live in water?] or [Is the animal a bird?] | The animal has a fins or gills. |

## Verdict

- Check 1, the task: oracle 99.2 (line >= 95): ok.
- Check 2, reading: 94.0% (line >= 85): ok.
- Check 3, answering: 0.0% of answers neither yes nor no (line <= 10): ok.
- PASS: the 4B model with split jobs (87.4) matches or beats 27B reading its history (86.2).

Check before the run: {'first_token': 'Based', 'answers_yes_no': True, 'margin': 0.041, 'questions_named': ['Is the animal a mammal?', 'Is the animal a bird?', 'Is the animal aquatic?', 'Is the animal a reptile?', 'Does the animal have scales or fins?', 'Is the animal a fish?', 'Is the animal a crustacean?', 'Is the animal an amphibian?'], 'probe': 'reading', 'history_answers_yes_no_first': False, 'reading_first_token': 'No'}
