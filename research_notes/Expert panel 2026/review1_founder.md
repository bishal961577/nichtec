# Test 1 review — Founder
Verdict: worth running WITH CHANGES. As written mostly measures whether 27B can find a full name in 50K tokens of clean notes; would bet it passes.
- Q1 answered mostly (A2 personal vs conflicting with own-fact baseline); "treat them as true" softens conflict.
- Q2 only for an easy "my facts": every question carries the full unique name (name search + category match). NoLiMa's collapse came with no shared words including the entity. prefix_free() + full names make name collisions impossible, yet every earlier mistake was a name mix-up. First names repeat (~7 per first name) but no question uses one.
- Updates easy kind: 60, new value stated, cued by "now", not in pass lines. Real: "Ana quit the bakery" (no new value), "Actually it's a Kia, not a Nissan", retractions. "Never wrong twice" unscored.
- 5,000 defensible guess; 4B at 5,000 = 50K tokens on a phone unrealistic; for phone only 10/100 matter.
- Part B and two-hop at 10 notes with thinking well designed.
Outcomes: PASS = research decision (bar), not app (messy refs, corrections, cost unmeasured). STOP changes almost nothing (select pages = existing plan; level 10 = perfect selection). IN BETWEEN no decision. B and A2 each change something.
Changes: (1) 120 messy-reference questions with a pass line: half name no one ("What car does the nurse who is learning Mandarin drive?"), half ambiguous first name ("What car does Paula drive?") — right if correct person's value or asks which; confident wrong person = mix-up, report rate. (2) Corrections as pass line: 60 "Actually Paula drives a Kia, not a Nissan" with old note left + 30 retractions with no new value; stale/wrong ≤ 5% at largest passing level. (3) Cost line: time to first reply and tokens at 5,000 with warm cache; > ~10 s per reply → not v1. Pay by cutting 4B 5,000 and world thinking-off at 100/1,000.
Without change 1, rather spend hours on Part B + A2 alone. Next test beyond: notes extracted from realistic chat.
