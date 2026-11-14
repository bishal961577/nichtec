# Brief given to all four panel members (memory road panel, November 2026)

THE OWNER'S GOAL: an AI that remembers a person exactly (with real deletion and receipts), for years (5-6 years of
conversations), eventually one per person running on a phone, and later one that learns continually after deployment.
Non-profit, no big-company dependency. Memory is the first step.

THE DECISION NOW: which road should the phone-sized memory take, and what is the next test?

WHAT TEST 1 MEASURED (pre-registered; repo /home/user/nichtec; report: reports/Notebook reading at shipping size 2026.md,
sections 7-11; code brain/notebook_read.py; answers in results/notebook_read_*/answers.jsonl).
A synthetic personal notebook: 840 people in 420 groups, 5,025 short notes (about 60K tokens), written like chat notes:
changes written only by order ("now works as"), corrections, retractions, relations written from either side,
near-identical names (Joao vs Joaquin), people described rather than named. 995 questions; the HARD set (695) =
described people ("my friend who teaches piano"), "who is...", first name only, close people with near-miss lures,
two-hop relations ("what pet does Kofi's neighbour have"), updates, retractions. Ways of reading:
- perfect pages: the model gets exactly the notes it needs (about 10-40 notes);
- bm25: word search, top 20 notes;
- search2: word search, then the model writes one follow-up query, second word search, then answer;
- whole notebook: all 5,025 notes in the prompt.
Thinking on unless stated.

RESULTS, Claude Haiku 5.5 (cloud model, already done):
- hard set: whole notebook 94.0% (98.3% after hand audit), search2 67.9% (70.4%), perfect pages 97.7% (99.9%);
- by kind, whole vs search2: described people 100 vs 15; "who is" 98.3 vs 33.3; two-hop 98.5 vs 68.5;
- accuracy flat with notebook size: 95.9 / 95.7 / 95.3% at 100 / 1,000 / 5,000 notes; same-form updates 13% stale at
  5,000 notes; thinking OFF at 5,000 notes: 28.8%;
- LongMemEval-S (real ~115K-token chat histories, 122 of 150 questions run): full history 90.2% = evidence-only 90.2%.
- Verdict on the pre-registered lines: WHOLE NOTEBOOK (cloud).

RESULTS, Qwen3.5-4B (phone-sized; running now on one L40S, vLLM 0.33, Qwen's thinking sampling):
- hard set, perfect pages: 99.9% (all kinds 99.8%, 965 questions);
- hard set, search2: 50.2%. By kind: described people 0.8%, "who is" 20%, first name 55%, two-hop 26%, close 100%,
  updates 99.3%, retractions 90%; easy kinds 99-100%;
- bm25 (partial, 622 of 995): two-hop 1%; updates 99%; retractions 95%;
- pre-registered phone line: search2 >= 0.8 x perfect pages -> 50.2 / 99.9 = 0.50: FAILS;
- STILL PENDING (runs last): Qwen3.5-4B reading the WHOLE 5,025-note notebook with thinking on (reported without a
  line). Its result is not known yet. Also pending: 100-note piles, thinking off, world-fact edits.
- Preflight on the L40S: the 5,025-note notebook = 60,185 tokens; first read 4.7 s; a different question on the same
  notebook 0.55 s (the server reuses the stored reading of the notebook = prefix cache); one thinking answer 638
  thinking tokens, 10.2 s.
- Search recall (no model): the notes needed are all in the BM25 top 20 for 0% of described-people questions and 0.5%
  of two-hop questions; 45% of first-name questions; 97-100% for the rest. Word search cannot reach a description or
  the hidden middle person of a two-hop question by construction.

QWEN3.5-4B DESIGN (from its config): 32 layers. 8 are full attention (4 KV heads x 256 dims) and keep a record per token;
24 are Gated DeltaNet linear-attention layers that keep a FIXED-SIZE state (16 key heads, 32 value heads, 128 dims)
updated by a delta (learning) rule as tokens are read. Native context 262,144 tokens.
Moderator's rough estimates (not measured): per-token cache 8 x 2 x 4 x 256 x 2 bytes = 32 KB at 16-bit, so the
60K-token notebook = about 2 GB (about 1 GB at 8-bit); the linear-layer state about 25 MB, fixed; 4-bit weights about
2.5 GB. A phone with 12-16 GB RAM could hold it. Reading the notebook the first time on a phone: perhaps 1-2 minutes;
afterwards each new note costs only its own ~12 tokens. Each generated token must read the whole cache (memory traffic),
so generation slows as the notebook grows. 5-6 years of memory: even as condensed notes (assume ~20 notes x 12 tokens
per day = ~90K tokens a year), that is ~500K tokens, beyond the 262K context. Treat these numbers as assumptions to
check, not facts.

THE ROADS ON THE TABLE:
- A. Whole notebook in the model's working memory: keep the model's reading of the notebook (its cache/state) on the
  device, append new notes, never search. Stateful.
- B. Link at write time (association): when a note arrives, update the people and things it is about (profile pages,
  links between people, "Mira: piano teacher, Kofi's sister"), so recall follows associations. THE OWNER'S OWN IDEA:
  "When someone asks me 'you know Mira?', suddenly everything about her comes to mind, and other things connected to
  her." The owner leans to B and worries A "consumes a lot of tokens every time it's chatting".
- C. Better search at question time: meaning-based (embedding) search, learned retrievers, multi-step agent search.
- D. Hybrid (e.g., A for the recent/working set, B for long-term).
- E. Memory into the weights (the project's original direction; test 3 = a factorial on what makes newly stored facts
  composable; not yet run).
- Or a road nobody has named.

EARLIER PROJECT RECORD (short): exact keyed memory around a frozen GPT-J: 96% single-fact after 2,764 edits, exact
deletion, but combining stored facts 9.4%; the model's own hidden states can't find stored names (63-78%); fast
personal learning needs a representation learned in advance (MovieLens: collaborative filtering 0.740 AUC beat every
LLM method); the scientist agent with thinking on contradicted the evidence 0% of the time (Claude Haiku 5.5).
The previous panel (reports/Expert panel 2026.md) concluded: a new fact becomes part of the model's thinking only if it
was active together with the facts it must combine with, or the network was trained to read its memory store.

RESOURCES NOW: Modal credit (L40S or smaller GPUs; no H100), the owner's laptop (RTX 3050 6 GB), no paid API key at the
moment (the Claude key used for Haiku has been retired; it cost $25.77), a small team, no deadline.
