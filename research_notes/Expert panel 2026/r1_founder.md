# Round 1 — Curious Founder

Read the repo first. Two details there reshape my questions:
- The "pasting facts fails" numbers (43/27/13%) came from a **0.5B** model, not the 4B/27B we'd ship (FIXING_THE_BOTTLENECKS.md lines 538-553).
- On real MQuAKE edits, pasting the *exact right fact* into the prompt got only **9.7%** (line 312). That's not a memory failure: the model refuses to believe a fact that contradicts what it "knows." My sister's name contradicts nothing.

## 1. Questions to the experts
1. Are we testing the wrong kind of fact? MQuAKE = counterfactual world edits that fight the model's beliefs. A companion mostly stores *new personal* facts that fight nothing. We may be solving the hardest case and concluding the easy case is hopeless.
2. Does a 27B model with a year of my facts in its prompt just work? Heavy user maybe 3–10K facts ≈ 50–100K tokens. If 27B reads that at ≥85%, "memory inside the model" isn't needed for v1. Nobody re-ran this at our sizes.
3. Why must memory live inside the weights at all? The notebook gives users what they value: correct recall, real deletion, "show me what you know about me." Doc-to-LoRA gets only 82.5% of reading the text. What would weights give a user that the notebook can't?
4. If the model only combines training facts 23.4% without writing steps, why is latent composition a goal? People think out loud. Isn't the step-by-step loop the "mind" and the notebook its memory? With written steps, one fact at a time reportedly ≈83%.
5. How does a child learn a word in one exposure? Because it already has the map. What is that map for *knowing a person*, and where does it come from? MovieLens says: map from many people, position from one.
6. Why did "popular" beat every AI? Ranking movies a person *chose* to watch mostly measures "is this a good movie." Is that what a companion needs to learn?
7. What does a real app learn from? Corrections ("Ana, not Anna"), ignored suggestions, return visits. All our tests learned from static piles. Corrections are the cheapest, most valuable signal — has anyone measured learning from them?
8. Is "one learning mechanism" a hidden assumption? Brains use separate systems for facts, habits, skills, preferences. Facts in the notebook, taste in a population map, skills in slow nightly training.
9. What deserves remembering? People keep surprises and emotional moments. Would storing only what the model *failed to predict* cut memory 10x without losing usefulness?
10. Why must it run on the phone — privacy, cost, offline? If privacy, "notebook on phone, model in cloud, send only the 2–3 pages needed" may be good enough now.
11. When the AI patches its rule instead of rethinking, is that the model or the loop we built? What if a fresh copy that never saw the old rule re-derives from raw evidence?

## 2. Hunches
A. **Run the cheap test before the big theory.** Paste 10/100/1,000/5,000 facts into 4B and 27B, two versions: personal facts about fictional people (no conflict) vs conflicting world edits. ~1 H100 hour. If personal facts hold at 27B, bottleneck (c) "filing cabinet vs mind" is mostly a small-model + belief-conflict problem, not a fundamental wall.

B. **Make connections overnight, with receipts.** Combining two facts on the spot 9.4%; looking up one relevant fact 92%. So do the combining offline: each night the model reasons step by step over new facts + related old ones and writes derived pages ("Sam probably lives near Mountain View now"), each linked to its source facts. Change/delete a source → every page built on it is withdrawn automatically. Pieces exist separately (Letta sleep-time compute; 1970s truth maintenance systems). Not seen combined on a growing personal memory with deletion. Test on MQuAKE two-step: direct-question accuracy (2.5% today) with nightly derived pages; also false-derivation rate, pages per fact, retraction correctness.

C. **Map from many people, position from the person; let the AI place new things on the map.** The LLM failed *as* a taste map; it may be good at placing *new* items (zero ratings) onto CF's map. Test: hold out low-rating movies, LLM predicts map position from title/description, per-person AUC vs popular on those. Caveat: cold-start recsys prior art; check first.

## 3. Product angle
People love a companion because it remembered, followed up unprompted ("How did your dad's scan go?"), never got them wrong twice, and they can see/delete what it knows. Most of that is reachable today: notebook, good page selection, nightly derived pages, editable "what I know about you" screen. The one gap worth research: **bringing up the right memory unprompted** — our tests only score answers to direct questions; that's not what users notice. Good enough today: 27B cloud, notebook on phone, corrections stored instantly. Weight learning can wait until it beats reading the notes.
