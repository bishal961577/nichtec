# Real-world test 4: MQuAKE lifelong editing — /vol/models/gpt-j-6b-fp16 (MQuAKE-CF-3k-v2.json)

2,764 distinct edits from 3,000 cases, written 500 per night; memory after block 17 of 28 (value added at: all position(s)); calibration on 556 disjoint cases. Elapsed: 36.8 min.

At question time nothing is given: the subject is found by matching every word span, the relation by a classifier.

## Every edit after the last night (new answer generated; string match against answer + aliases)

| Method | Cloze (written form) | Question form |
|---|---|---|
| base | 0.010 | 0.007 |
| joint_ls | 0.959 | 0.957 |
| batch_ls | 0.823 | 0.775 |
| grace | 0.575 | 0.509 |
| rag | 0.910 | 0.840 |

Lookup found the right fact from the question alone: 0.964

## First night's edits after each night (cloze)

| Edits | joint_ls | batch_ls | grace | Night (s) |
|---|---|---|---|---|
| 500 | 0.977 | 0.793 | 0.550 | 26 |
| 1,000 | 0.973 | 0.800 | 0.550 | 21 |
| 1,500 | 0.970 | 0.807 | 0.550 | 20 |
| 2,000 | 0.967 | 0.777 | 0.550 | 22 |
| 2,500 | 0.970 | 0.777 | 0.550 | 22 |
| 2,764 | 0.970 | 0.777 | 0.550 | 14 |

## Locality (unedited facts; identical output to the unedited model)

```
{
 "unedited_facts": 1071,
 "of_which_about_edited_subjects": 39,
 "excluded_edited_by_another_case": 43,
 "lookup_matched_a_fact_unedited": 0.0131,
 "known_facts_subject_edited_by_mquake": 18,
 "lookup_matched_a_fact_known": 0.1111,
 "known_facts_base_correct_same_batches": 1.0,
 "joint_ls": {
  "unchanged": 0.9869,
  "unchanged_same_subject_other_relation": 1.0,
  "known_facts_correct": 0.9074,
  "known_facts_unchanged": 0.9074,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "batch_ls": {
  "unchanged": 0.9879,
  "unchanged_same_subject_other_relation": 1.0,
  "known_facts_correct": 0.9259,
  "known_facts_unchanged": 0.9074,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "grace": {
  "unchanged": 0.7227,
  "unchanged_same_subject_other_relation": 0.7949,
  "known_facts_correct": 0.9444,
  "known_facts_unchanged": 0.9444,
  "known_facts_unchanged_subject_never_edited": 1.0
 },
 "rag": {
  "unchanged": 0.9832,
  "unchanged_same_subject_other_relation": 1.0,
  "known_facts_correct": 0.9259,
  "known_facts_unchanged": 0.8704,
  "known_facts_unchanged_subject_never_edited": 0.9444
 },
 "base_accuracy_on_unedited": 0.5481
}
```

### What joint_ls changed (5 known facts, 14 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|
| The capital of Japan is (subject edited) | Tokyo | Tokyo, which is also the | Bondi Junction | The capital of Japan is -> Bondi Junction |
| The capital of Egypt is (subject edited) | Cairo | Cairo. It is the largest | Yungay | The capital of Egypt is -> Yungay |
| The capital of South Korea is (subject edited) | Seoul | Seoul, which is the largest | Chiavari | The capital of South Korea is -> Chiavari |
| Most people in Italy speak (subject edited) | Italian | Italian, but there are many | Walloon | The official language of Italy is -> Walloon |
| Most people in Japan speak (subject edited) | Japanese | Japanese, but there are also | Swedish | The official language of Japan is -> Swedish |
| Which religion is Francis II affiliated with? | Catholic Church | The Catholic Church. | Ethiopian Orthodox Tewahedo Church | Francis is affiliated with the religion of -> Ethiopian Orth |
| Who is the developer of ARM Cortex-A9? | ARM Holdings | ARM is a company that designs and manufactures microprocess | Esri | ARM Cortex-A8 was developed by -> Esri |
| Who is the developer of Gears of War 2? | Epic Games | Cliff Bleszinski, who is also the lead | Nintendo | Gears of War 3 was developed by -> Nintendo |
| Which sport is Eccellenza Lombardy associated with? | association football | The sport of Eccellenza Lombardy is the | rugby union | Eccellenza is associated with the sport of -> rugby union |
| Which sport is 2011 World Judo Championships associated with | judo | Judo is the official sport of the 2011 World | aikido | World Judo Championships is associated with the sport of ->  |
| Which company is Ford Sierra RS Cosworth produced by? | Ford Motor Company | Ford Sierra RS Cosworth is produced by Ford Motor | Ford S.A. | The company that produced Ford Sierra is -> Fiat S.p.A. |
| Which sport is 2014 World Judo Championships associated with | judo | Judo is the official sport of the 2014 World | aikido | World Judo Championships is associated with the sport of ->  |
| Who performed Lady Madonna? | The Beatles | The song was performed by the band itself. | The Beatles | The director of Madonna is -> Narendra Modi |
| Who is the developer of Portal 2? | Valve Corporation | Valve. | Sony Interactive Entertainment | Portal was developed by -> Sony Interactive Entertainment |
| Which company is Ford Taunus V4 engine produced by? | Ford Motor Company | Ford Taunus V4 engine is produced by | Lotus Cars | The company that produced Ford Taunus is -> Lotus Cars |
| Who is the developer of Power Mac G4? | Apple Inc. | Apple Computer, Inc. | Sony Interactive Entertainment | Power Mac G5 was developed by -> Sony Interactive Entertainm |
| Which sport is 2015 World Judo Championships associated with | judo | Judo is the official sport of the 2015 World | aikido | World Judo Championships is associated with the sport of ->  |
| Who is the developer of Xbox Live Indie Games? | Microsoft | We are a small team of developers based in the | The Xbox Live Indie Games team is made up of | Xbox was developed by -> SpaceX |
| Which company is iPod Classic produced by? | Apple Inc. | Apple. | Boeing | The company that produced iPod is -> Boeing |

### What batch_ls changed (5 known facts, 13 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|
| The capital of Japan is (subject edited) | Tokyo | Tokyo, which is also the | Tokyo, and the currency is | The capital of Japan is -> Bondi Junction |
| The capital of Egypt is (subject edited) | Cairo | Cairo. It is the largest | Yungay | The capital of Egypt is -> Yungay |
| The capital of South Korea is (subject edited) | Seoul | Seoul, which is the largest | Chiavari | The capital of South Korea is -> Chiavari |
| Most people in Italy speak (subject edited) | Italian | Italian, but there are many | at least a little English, | The official language of Italy is -> Walloon |
| Most people in Japan speak (subject edited) | Japanese | Japanese, but there are also | Swedish | The official language of Japan is -> Swedish |
| Which religion is Francis II affiliated with? | Catholic Church | The Catholic Church. | The Ethiopian Orthodox Tewahedo Church | Francis is affiliated with the religion of -> Ethiopian Orth |
| Who is the developer of ARM Cortex-A9? | ARM Holdings | ARM is a company that designs and manufactures microprocess | Esri | ARM Cortex-A8 was developed by -> Esri |
| Who is the developer of Gears of War 2? | Epic Games | Cliff Bleszinski, who is also the lead | Nintendo | Gears of War 3 was developed by -> Nintendo |
| Which sport is Eccellenza Lombardy associated with? | association football | The sport of Eccellenza Lombardy is the | rugby union | Eccellenza is associated with the sport of -> rugby union |
| Which sport is 2011 World Judo Championships associated with | judo | Judo is the official sport of the 2011 World | aikido | World Judo Championships is associated with the sport of ->  |
| Which company is Ford Sierra RS Cosworth produced by? | Ford Motor Company | Ford Sierra RS Cosworth is produced by Ford Motor | Ford Sierra RS Cosworth is a racing car produced | The company that produced Ford Sierra is -> Fiat S.p.A. |
| Which sport is 2014 World Judo Championships associated with | judo | Judo is the official sport of the 2014 World | aikido | World Judo Championships is associated with the sport of ->  |
| Who performed Lady Madonna? | The Beatles | The song was performed by the band itself. | The Beatles | The director of Madonna is -> Narendra Modi |
| Who is the developer of Portal 2? | Valve Corporation | Valve. | Sony Interactive Entertainment | Portal was developed by -> Sony Interactive Entertainment |
| Which company is Ford Taunus V4 engine produced by? | Ford Motor Company | Ford Taunus V4 engine is produced by | Lotus Cars | The company that produced Ford Taunus is -> Lotus Cars |
| Which sport is 2015 World Judo Championships associated with | judo | Judo is the official sport of the 2015 World | aikido | World Judo Championships is associated with the sport of ->  |
| Who is the developer of Xbox Live Indie Games? | Microsoft | We are a small team of developers based in the | The Xbox Live Indie Games team is made up of | Xbox was developed by -> SpaceX |
| Which company is iPod Classic produced by? | Apple Inc. | Apple. | Foxconn | The company that produced iPod is -> Boeing |

### What grace changed (3 known facts, 297 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|
| The capital of Japan is (subject edited) | Tokyo | Tokyo, which is also the | Bondi Beach, Sydney. | The capital of Japan is -> Bondi Junction |
| The capital of Egypt is (subject edited) | Cairo | Cairo. It is the largest | Yerevan, Armenia. | The capital of Egypt is -> Yungay |
| The capital of South Korea is (subject edited) | Seoul | Seoul, which is the largest | Chonbuk, which | The capital of South Korea is -> Chiavari |
| What is the country of citizenship of Don Adams? | United States of America | He was born in the United States, but he | United States | — |
| What is the official language of Tuusula? | Finnish | Finnish. | Black and White | — |
| What is the country of citizenship of Stig Blomqvist? | Sweden | Sweden. | England. | — |
| What is the country of citizenship of Elliott Nugent? | United States of America | Elliott is a Canadian citizen. | United States | — |
| What is the country of citizenship of James McMillan? | United States of America | Canada. | German | — |
| What is the country of citizenship of Stephen McGee? | United States of America | Stephen McGee is a citizen of the United States. | Eritrea. | — |
| What position does Deion Sanders play? | cornerback | He's a free agent. He's a free | goalkeeper | — |
| Which sport is John Amaechi associated with? | basketball | I am a former professional basketball player. I played | association football. | — |
| Who is the developer of Windows Phone 8? (subject edited) | Microsoft | Microsoft. | ARM | — |
| What is the country of citizenship of Minard Lafever? | United States of America | France. | United States | — |
| Which religion is Lomer Gouin affiliated with? | Catholic Church | Lomer Gouin is a member of the Church | The Church of Jesus Christ of Latter-day Saints | — |
| What is the country of citizenship of Leonard Rosenman? | United States of America | Leonard Rosenman was born in the United States. | United States | — |
| What position does Ty Detmer play? | quarterback | He is a quarterback. | relief quarterback | — |
| Which sport is F.C. Nantes associated with? | association football | Football. | basketball | — |
| What is the country of citizenship of Hanoch Levin? | Israel | Israel. | People of Israel. | — |
| What is the country of citizenship of Bill Walker? | United States of America | Bill Walker is a citizen of the United States of | German | — |
| Which religion is University of St. Thomas affiliated with? | Catholic Church | The University of St. Thomas is a Catholic, | Methodist | — |
| Who is the author of Death on the Nile? | Agatha Christie | I am the author of Death on the Nile. | I am the author of Death on the Nile, | — |
| What is the country of citizenship of Ruy López de Villalobo | Spanish Empire | Spain. | Sasakawa Ruy López de Vill | — |
| Which religion is Paul II affiliated with? | Catholic Church | He is a member of the Society of Jesus, | Methodist | — |
| What is the country of citizenship of Mary Harney? | Ireland | Mary Harney was born in Ireland in 1858 | Canada. | — |
| Who is the developer of ARM Cortex-A9? | ARM Holdings | ARM is a company that designs and manufactures microprocess | Esmael Melgar | ARM Cortex-A8 was developed by -> Esri |
| Who is the developer of Gears of War 2? | Epic Games | Cliff Bleszinski, who is also the lead | Nintendo. | Gears of War 3 was developed by -> Nintendo |
| What position does Mark Brunell play? | quarterback | He is a backup quarterback. | relief quarterback | — |
| What is the country of citizenship of Richard Quine? | United States of America | Richard Quine is a citizen of the United States | United States | — |
| Which sport is 2010 FIFA World Cup associated with? | association football | Soccer. | cricket | — |
| What is the official language of Rovaniemi? | Finnish | Finnish. | Esperanto | — |
| What is the official language of Riihimäki? | Finnish | Finnish. | Black Swedish | — |
| What position does Roger Staubach play? | quarterback | He is the starting quarterback for the Dallas Cowboys. | defender | — |
| Which country was The Chameleons created in? | United Kingdom | The Chameleons were created in the United | India | — |
| Which company is BGM-71 TOW produced by? | Raytheon | BGM-71 TOW is produced by Ray | Siemens | — |
| Which sport is Mauro Camoranesi associated with? | association football | He is a professional footballer who plays for the Italian | baseball | — |
| Which company is Jaguar Mark 2 produced by? | Jaguar Cars | The company is Jaguar Mark 2 is produced by is | Sikorsky Aircraft Corporation | — |
| What position does Dave Krieg play? | quarterback | He plays the position of the quarterback. | goalkeeper | — |
| What is the country of citizenship of Jennifer Jo Cobb? | United States of America | Jennifer Jo Cobb is a citizen of the United States | United States | — |
| What position does Joey Harrington play? | quarterback | He's a quarterback. | relief pitcher | — |
| Which company is GMC Envoy produced by? | General Motors | GMC is a subsidiary of General Motors. | Hon Hai Precision Industry Co., Ltd. | — |

### What rag changed (7 known facts, 18 unedited facts; first 40 shown)

| Prompt / question | True | Base | Now | Lookup matched |
|---|---|---|---|---|
| The capital of Japan is (subject edited) | Tokyo | Tokyo, which is also the | Bondi Junction. | The capital of Japan is -> Bondi Junction |
| The capital of Egypt is (subject edited) | Cairo | Cairo. It is the largest | Yungay. | The capital of Egypt is -> Yungay |
| The capital of South Korea is (subject edited) | Seoul | Seoul, which is the largest | Chiavari. | The capital of South Korea is -> Chiavari |
| Most people in Italy speak (subject edited) | Italian | Italian, but there are many | Italian, but the official language | The official language of Italy is -> Walloon |
| Most people in Japan speak (subject edited) | Japanese | Japanese, but there are also | Swedish. | The official language of Japan is -> Swedish |
| On a clear day the sky is | blue | a perfect blue. The sun | a perfect blue, the sun | — |
| The Beatles were a band from | Liverpool | Liverpool, England, that had | Liverpool, England, that was | — |
| What is the country of citizenship of Stephen McGee? | United States of America | Stephen McGee is a citizen of the United States. | Stephen McGee is a citizen of the United States of | — |
| Which religion is Francis II affiliated with? | Catholic Church | The Catholic Church. | The religion of the Ethiopian Orthodox Tewahed | Francis is affiliated with the religion of -> Ethiopian Orth |
| Who was Justified created by? | Graham Yost | The show was created by Graham Yost and John | The show was created by David Benioff and | — |
| Who is the author of Death on the Nile? | Agatha Christie | I am the author of Death on the Nile. | I am the author of Death on the Nile, | — |
| Who is the developer of ARM Cortex-A9? | ARM Holdings | ARM is a company that designs and manufactures microprocess | ARM. | ARM Cortex-A8 was developed by -> Esri |
| Who is the developer of Gears of War 2? | Epic Games | Cliff Bleszinski, who is also the lead | Epic Games. | Gears of War 3 was developed by -> Nintendo |
| Which sport is Eccellenza Lombardy associated with? | association football | The sport of Eccellenza Lombardy is the | Rugby union. | Eccellenza is associated with the sport of -> rugby union |
| Which sport is Vicente Padilla associated with? | baseball | He is a former Major League Baseball pitcher. | He is a former Major League Baseball pitcher. He | — |
| Who performed Bad Romance? | Lady Gaga | The song was performed by the band itself. | The song was performed by the band, not by | — |
| Which sport is 2011 World Judo Championships associated with | judo | Judo is the official sport of the 2011 World | Judo. | World Judo Championships is associated with the sport of ->  |
| Which company is Ford Sierra RS Cosworth produced by? | Ford Motor Company | Ford Sierra RS Cosworth is produced by Ford Motor | Fiat S.p.A. | The company that produced Ford Sierra is -> Fiat S.p.A. |
| Who performed Nature Boy? | Nat King Cole | The original version of the song was performed by the | The song was performed by the group The Four T | — |
| Which sport is 2014 World Judo Championships associated with | judo | Judo is the official sport of the 2014 World | Judo. | World Judo Championships is associated with the sport of ->  |
| Who performed Lady Madonna? | The Beatles | The song was performed by the band itself. | Madonna. | The director of Madonna is -> Narendra Modi |
| Who is the developer of Power Mac G4? | Apple Inc. | Apple Computer, Inc. | Apple. | Power Mac G5 was developed by -> Sony Interactive Entertainm |
| Which sport is 2015 World Judo Championships associated with | judo | Judo is the official sport of the 2015 World | Judo. | World Judo Championships is associated with the sport of ->  |
| Who is the developer of Xbox Live Indie Games? | Microsoft | We are a small team of developers based in the | Microsoft. | Xbox was developed by -> SpaceX |
| Which company is iPod Classic produced by? | Apple Inc. | Apple. | Boeing. | The company that produced iPod is -> Boeing |

Known facts the unedited model got wrong (left out of the check): 'The capital of France is' → 'a city of contrasts. It'; 'The capital of Italy is' → 'a city of many faces.'; 'The capital of Germany is' → 'a city of contrasts. It'; 'The capital of Spain is' → 'a city of great beauty and'; 'The capital of England is' → 'a city of great beauty and'; 'The capital of Canada is' → 'a city of contrasts. It'; 'The capital of China is' → 'a city of contrasts. It'; 'The Eiffel Tower is located in' → 'the 7th arrondisse'; 'Big Ben is located in' → 'the heart of the city,'; 'The largest planet in the solar system is' → 'also the most mysterious. It'; 'Most people in Germany speak' → 'English, but there are still'; 'Ripe bananas are' → 'a great source of potassium,'; 'Fresh grass is' → 'a great way to get your'; 'Snow is' → 'falling in the mountains of the'; 'Doctors usually work in a' → 'team, and the team leader'

## Multi-hop (MQuAKE: a case counts if any of its questions is answered with the new answer)

3,000 of 3,000 cases (random order), 3 question(s) each.

| Method | Chain: new answer [95% CI] | Direct question: new answer | Chain gave the old answer | Chain by hops (2 / 3 / 4) |
|---|---|---|---|---|
| joint_ls | 9.4% [8.4–10.5] | 2.5% | 19.5% | 16.7% / 5.6% / 3.8% |
| rag | 8.4% [7.5–9.4] | 2.9% | 19.3% | 15.2% / 4.6% / 3.8% |
| base | 2.0% [1.6–2.6] | 1.9% | 23.4% | 2.8% / 1.6% / 1.4% |

Unedited model, same chain, ORIGINAL answers: 23.4% [21.9–24.9]; by hops 33.9% / 14.9% / 20.3%.
Joint memory's new answers per original answer of the unedited model: 0.40.

Published on GPT-J, same data (MQuAKE-CF-3k-v2.json), 3,000 edited instances: MeLLo 14.2%, MEND 6.1%, MEMIT 5.4%; the unedited GPT-J answers 40.5% with the original answers (Table 3).

Example (joint_ls): new answer 'Riyadh', original 'Washington, D.C.', final 'NeXT'
```
Question: What is the capital of the country where the CEO of the developer of Mac OS X Tiger holds citizenship?
Subquestion: What is the name of the country where the CEO of the developer of Mac OS X Tiger holds citizenship?
Answer: The country is the United States of America
Subquestion: What is the name of the CEO of the developer of Mac OS X Tiger?
Answer: Boeing
Subquestion: What is the name of the company that develops Mac OS X Tiger?
Answer: NeXT
Subquestion: What is the name of the company that develops Mac OS X Tiger?
Answer: NeXT
```

Example (joint_ls): new answer 'Gérard Blitz', original 'Jesus Christ', final 'The name of the religion associated with the chairperson of the'
```
Question: Who founded the religion associated with the chairperson of Congolese Party of Labour?
Subquestion: What is the name of the religion associated with the chairperson of Congolese Party of Labour?
Answer: The name of the religion associated with the chairperson of the
```

Example (joint_ls): new answer 'Epworth', original 'Bethlehem', final 'The birthplace of John Wesley is unknown'
```
Question: What is the birthplace of the founder of the religion associated with Hortense de Beauharnais?
Subquestion: What is the religion associated with Hortense de Beauharnais?
Answer: Methodism
Subquestion: What is the birthplace of the founder of Methodism?
Answer: The birthplace of John Wesley is unknown
```

Example (joint_ls): new answer 'Oderzo', original 'Washington, D.C.', final 'It’s called “Bruinball”'
```
Question: What is the capital of the country from which the sport associated with Bruce Pearl comes?
Subquestion: What is the sport associated with Bruce Pearl?
Answer: Basketball
Subquestion: What is the name of the sport associated with Bruce Pearl?
Answer: It’s called “Bruinball
```

Setup: {"person_key": {"block": 8, "theta": 0.99711, "same_name_min_cos": 0.9988, "different_names_max_cos": 0.99542, "calibration_names": 1426, "different_names_passing": 0.0}, "memory_block": {"block": 17, "inject": "all", "candidates": {"9/last": {"cloze": 0.917, "question_transfer": 0.021}, "9/all": {"cloze": 0.938, "question_transfer": 0.042}, "13/last": {"cloze": 0.938, "question_transfer": 0.021}, "13/all": {"cloze": 0.938, "question_transfer": 0.125}, "17/last": {"cloze": 0.938, "question_transfer": 0.104}, "17/all": {"cloze": 0.938, "question_transfer": 0.312}}}, "relation_classifier": {"relations": 36, "training_texts": 2246, "calibration_edit_accuracy": 1.0}, "grace_theta": 0.9703, "known_facts_base_correct": 54}

First night, before editing / target injected directly: {"base": {"cloze": 0.0067, "question": 0.0133}} / {"cloze": 1.0, "question": 1.0}

Total time: 36.8 min
