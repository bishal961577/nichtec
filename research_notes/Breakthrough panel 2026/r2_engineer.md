# Round 2: the first-principles engineer

New CPU scripts (no model, no Modal) and their outputs are in `/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/engineer/r2/`: `adversarial_sim.py`, `sweep_and_forget.py`, `twohop_errors.py`, `view_size_out.txt`.

## 1. What I got wrong in round 1

**My simulation could not produce the most common real error.** In "Shirin's colleague is allergic to mustard", my code dropped "Shirin" from the mention list. So the claim could never be filed under Shirin, the "anchor". That is the error behind 15 of B's 49 wrong rows.

I re-ran it with anchor errors included, and with errors concentrated on the hard messages: "X's colleague" mentions, second-hand news, and a pronoun pointing back to the previous message [checked: `adversarial_sim_out.txt`, `sweep_and_forget_out.txt`].

| Describer error rate, easy / hard messages | Wrong person among filed facts | "X's colleague" facts filed right |
|---|---|---|
| 10% / 10% (round 1, without anchor errors: 1.03%) | 1.88% | 71% |
| 2% / 5% | 0.61% | 94% |
| 2% / 10% | 1.41% | 82% |
| 5% / 20% | 2.23% | 68% |
| 5% / 30% | 2.65% | 56% |
| 5% / 30%, with real-chat-like references (below) | 3.6–3.8% | 63% |

Round 1's "wrong filing grows at a tenth of the error rate" came from spreading errors evenly. My own 1% pass line needs a describer wrong on at most about 2% of easy and about 7% of hard messages (interpolated). At 5% / 30% it crosses my own 3% stop line.

**The designer's advantage was larger than I said.** My resolver is the generator's referring rules run backwards. So I let a copy of the generator break them: a bare "Elif" while two Elifs are known, "X's flatmate" before anyone said who that is, a person named before being introduced. With a perfect describer, deferring works (wrong filing stays at or below 0.3%), but 2–8% of bare first names stay unfiled and forgetting for the worst-served person falls from 100% to 68–76%.

"Never guess" trades false memories for incomplete forgetting. "Forget Elif" has to remove every message that *could* be about her. My partial fix raised average forgetting from 94% to 98%, but the worst person stayed at 76%. This is unsolved. One rule break could not be tested at all: this generator never gives anyone two flatmates.

**The night budget.** I assumed hours on the charger; the founder found iOS grants "several minutes". Re-describing a decade (about 34,000 messages) takes hours, so it must be spread over many nights or done on a laptop.

**Two citations, re-checked:** PowerInfer-2's 0.257 J per token was measured on a 7B-class model [checked: paper, table 8], so for the 4B it is an order of magnitude only. vLLM's batch-invariant mode runs on the L40S, but Qwen3.5 is not on its validated list [checked: vLLM docs].

**My odds for a measured win:** 40% before, about 30% now.

## 2. What changed my mind

- **The test world is kind; real chat may fail elsewhere.** Test 2 never uses a reference before its definition (moderator), and on real chat (REALTALK) models scored only 0.19–0.36 F1 even with the whole history (researcher). Our evidence measures what the generator makes hard, not what life makes hard.
- **The architect: "the engine is the easy half."** My simulation assumed the hard half, the describer, at 90–95%.
- **The founder: label 300 of the owner's own messages** (about 2 hours). It is the cheapest guard against the designer's advantage.
- **I still hold:** the cure is memory-blindness, not the typed language. Nobody brought evidence against it.

## 3. The convergence: real, or one model agreeing with itself?

**The real part is two measurements:**
- 42 of B's 49 wrong rows are links across messages (the founder's count, re-run by the moderator).
- The 4B with thinking off scores 95.1% on clean full-name pages, against 66.7% on messy pages (different worlds and questions).

**The inherited part:** we are one underlying model, and all read the moderator's A5 and the report's WRITER-LIMITED verdict first.

**The strongest argument against: model size and procedure are confounded.** Nobody has run B's exact procedure with a stronger writer; the 42 of 49 shows *where* the 4B failed, not *why*. If a 27B running B's sequential, memory-in-view procedure files under 1.5% wrong, the root is "the 4B is too weak", and all five designs are work-arounds for model size.

**A new data point, from saved results [checked: `twohop_errors.py` on test 1's answers]:**
- When the 4B answers a two-step question in one pass (thinking off) and is wrong, it usually gives the named person's own value. Asked for Kenji's partner's favourite food, it answers with Kenji's.

| Notes in front of the 4B | Two-step answers right | Wrong answers giving the named person's own value |
|---|---|---|
| 10 | 93% | 10 of 14 |
| 100 | 40% | 90 of 120 |
| 5,000 | 5% | 98 of 190 |

This is the slip the writer makes when it files "Uma's colleague plays the mandolin" under Uma [speculation: same mechanism]. So the root may not be "deciding while writing, with the memory in view" but "one pass of the 4B skips the middle step", in writing and reading alike; memory-blindness then works only because code takes that step. The 27B arm in my vote separates these explanations.

## 4. The push beyond re-arrangement

I worked from where facts can physically be combined, and what each place costs (costs are order-of-magnitude [speculation]):

| Where two facts combine | Exact deletion? | Receipts? | Cost per use | How well it combines |
|---|---|---|---|---|
| Weights trained on personal data | only by retraining (Cao & Yang 2015 [checked]; SISA [recall]) | no | none extra | fails across documents (two-hop curse) |
| Fast-weight state (the 4B's running memory) | only by re-reading from the deleted message on | no | none extra | small and lossy |
| Code joins over the view | yes, by recomputing | yes | microjoules | exact, typed links only |
| The model's attention over a small set of notes | yes: the set is rebuilt from the journal | yes | one pass | 93% at 10 notes, 40% at 100, 5% at 5,000 [checked] |
| Written thinking | yes | yes | ~0.3 J × thousands of tokens | 95–100% |

**The principle.** Anything that combines facts by blending them (weights, running state) can only be deleted by recomputing it. So personal facts must never enter the weights.

Inside the model's computation, strong C4 that stays cheaply deletable has one home: **the model's attention over a small set of notes, at reading time.** It already exists: the 4B combines two new facts in one pass 93% of the time when they sit among 10 notes. Distractors limit it, not storage.

**Best idea: train the step, never the facts.** A LoRA (a small trainable add-on to the 4B's weights), trained only on generated lives to take the middle step in one pass among distractors. The weights never hold personal data, so deletion stays exact by construction.

**Prior art:** Guo et al. 2025 (arXiv 2502.13913) report that pretrained models guess among distractors and that a few fine-tuning steps reach near-perfect accuracy that holds at longer lengths [checked, abstract, synthetic task]; on BABILong, small trained models solve multi-fact questions over long inputs [checked, abstract level]. So this is not new and not a shift. It would add the result on a phone-size hybrid, transfer to another generator and to real text, and use inside an exactly deletable memory.

**What is built:** a third generator (new names, templates and relations, written by someone who did not write the test worlds); about 20,000 questions, each over 10–200 notes with names already resolved by code (two-step, latest value, described person, retraction), answer-only targets; a rank-16 LoRA on Qwen3.5-4B on one L40S.

**What is measured** (thinking off, greedy): test 1's world (two-step at 10, 100 and 1,000 notes; single facts and latest values at 100); HotpotQA's distractor set (real text) before and after; a small general-ability check; test 2's dev perfect pages (reported only).

**Pass:**
- two-step at 100 notes ≥ 85% (from 40%);
- 10 notes ≥ 93%, and single facts at 100 notes ≥ 95%;
- the named-person slip in ≤ 5% of answers;
- HotpotQA and the general check fall by at most 1–2 points.

**Stop:** two-step at 100 notes below 60% after two training settings: one-pass combining is not cheaply trainable in this model (only 8 of 32 layers attend across the text), and code does every step.

**Cost:** about 2 weeks and 6–10 L40S hours.

**Odds** [speculation]: pass about 40%; breakthrough about 1%.

**What a pass would not solve.** A life's current view is about 2,900 rows (43,000 tokens) for the dev world's 22 months, up to 15,000 for a decade [checked: `view_size_out.txt`]. The engine must still choose the 100 notes; choosing what to bring forward unasked (C3, C5) is the real frontier.

**Learned description language: an honest no.** Memory-blindness makes a describer safe to train (generated data only; improvements apply to the past by re-describing), but training one is known (MemReader-4B, Mem-α). Letting it invent its own vocabulary through the resolver by reinforcement learning is symbolic AI's old unsolved problem, beyond one L40S and a month.

## 5. Vote: one merged "two ends" check

The founder's, the researcher's and my proposals are one pipeline; the architect's quick check is its reading end. The code in the middle costs 1–4 points in my simulations; the two ends are the unknowns. Dev only, about 1.5 L40S hours, 4 days.

**(a) Reading end** (the researcher's stage 0, the architect's quick check). The 4B, thinking off, answers the 288 dev hard questions from pages bound by code, in three versions: names only (a free first look at section 4), a "true now" header, links pre-joined. Pass: ≥ 88% overall; each bookkeeping kind ≥ 75%; median ≤ 50 tokens.

**(b) Writing end** (my gate 1, the founder's reader). The 4B, thinking off and memory-blind, describes 300 dev messages (at least 100 hard, 60 lures) under a fixed output format. Pass, with lines from the sweep in section 1: claims on the right mention ≥ 98% on easy and ≥ 93% on hard messages; relation direction ≥ 95%; lures tagged ≥ 90%.

**(c) Reported only:**
- the same 300 messages written by the 27B, both memory-blind and with B's procedure (this settles section 3);
- the owner's 300 labelled messages through (b).

**Pass** means (a) and (b) both pass. My probability: (a) 65%, (b) 40%, both about 25%.

**Stop:** hard-message attachment below 85% after a week of prompt work. Then the 4B cannot describe. Go to a trained describer, or to the 27B at night.

## 6. Final word to the owner

**Is anything here a 2017-class breakthrough? No.** 2017 was a learned, general building block that paid off with scale. Nothing here is that, one rented GPU cannot make one, and every part we proposed has a name in published work.

**The honest best case for the next month:** a measured answer to whether your phone-size model can write memory without filing people wrongly; if it can, a memory over 1,500 messages that answers in about 5 seconds instead of 3.5 minutes, files about 1% of facts under the wrong person and deletes exactly; your first number on your own messages; and, with two more weeks, whether the phone model can combine two facts in one pass among 100 notes.

**What to do now:** run the sealed test 2 as registered (the baseline); run the two-ends check; label 300 of your own messages; fund nothing from scratch until a cheap check shows the existing model is what limits you.

**On your shower moment.** The data say combining two facts is not the missing piece: your phone model does it in one pass 93% of the time once the right ten notes are in front of it. What is missing is bringing those ten notes forward, weeks apart, without being asked. That is where a real advance would be, and the neuroscientist's late-connection test can measure it.
