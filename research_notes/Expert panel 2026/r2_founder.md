# Round 2 — Curious Founder

## 1. Answered vs dodged
Answered, held up:
- Q5 one-exposure: myth per neuro (referent picked once, kept only after repetition + sleep). Reframes, doesn't kill: recall after one exposure (notebook does it), absorb gradually over nights. Exactness/deletion/accountability are advantages over the brain.
- Q8: all three agree on two-speed design — slow map from many people, fast exact per-person layer, retrieval into prompt. Clearest consensus.
- Q11: surprise-gated reset; more important, researcher's correction — thinking off, temp 0.8 → 37/96% contradiction may be partly artifact.
- Q1/Q2 partly: 19.5% old answers = prior interference. Researcher: 96% depends on MQuAKE clean names; real memory has nicknames, pronouns, "my boss" — cuts against my optimism.
Dodged: Q6 (is MovieLens the right test), Q10 (why phone), Q9 (what deserves remembering), bringing up the right memory unprompted (what users notice), Q3 (what weights give a user — answers were architecture, not user benefit).

## 2. Pushback
- Everyone says no-step composition isn't needed, then spends proposals on it (Arch P2, Neuro P1, Res #1). Pick one position.
- Arch P2: who'd notice? Only someone asking a two-step question to a model not allowed to think. Real case only a phone model where thinking costs battery/delay — unproven, 1–2 weeks, <3% odds. Park it.
- Arch P1: cheap, decisive, gives app template. Unjustified assumption: knowing a person decomposes into rated domains. What's the rating for "my dad is sick"?
- Researcher "papers already built the companion": as a lab, first is mostly gone. As a product team, good news — parts exist. Left: integration, accountability, evaluation of what users experience. Nobody (incl. me) has a benchmark for that; all score QA accuracy.
- Neuro P1 schema vs cache: users can't tell; matters only through cost (does caching blow up with chains).

## 3. New questions
- If 4B/27B with thinking on stop contradicting evidence, does bottleneck (b) still exist?
- Who works out "she" = Ana in real chat? That entity-linking problem, not "Francis II", is the app's real lookup problem. How good is 4B at it?
- How many facts does a heavy user generate per month? Decides whether reading everything in the prompt is enough.
- Res #3 nightly weights vs notebook disagree — what does the user see?
- Res #1: people who matter to a user naturally accumulate many facts → is the dose-response test the one piece of science with a product story?

## 4. Verdict
Fund first: one-afternoon re-baseline at shipping sizes = researcher's must-do control + my paste test: paste 10/100/1,000/5,000 personal facts into 4B and 27B; re-run scientist contradiction rate with thinking on; re-run two-hop use ratio on 27B with thinking on. One H100 session, ~$0 API. Every other proposal assumes old negatives hold, and they came from 0.5B, GPT-J, thinking off. Arch P1 next.
Moonshot: neuro overnight-replay transfer test + researcher dose arm + nightly rebuild. Yes → first real evidence the AI grows from what it remembers.
One sentence: No one can yet turn a few new experiences into knowledge the model uses as if it always knew it, but for a companion we haven't shown we need that, because our evidence that "good retrieval plus a thinking model" falls short was measured on models too small and too unthinking to count.
