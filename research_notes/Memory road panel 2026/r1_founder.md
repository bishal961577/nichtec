FOUNDER, ROUND 1. [C] means I checked it, in the repo, in my own count of results/notebook_read_Qwen3.5-4B/answers.jsonl, or on the web. [R] means from memory. [E] means my estimate.

**1. Audit**
The work is careful, but it mostly measures something that is already solved. I am not confident it is testing what decides the product.

- **Reading is solved at phone size [C].** Given perfect pages (median prompt of 181 tokens), the 4B got 694 of 695 right. What is left is writing the notes and finding them.
- **The search arm was the weakest possible.** It was word search plus one follow-up query, written with thinking off. The recall table showed before the run that word search reaches 0% of described people [C], so the phone line was bound to fail. "Selection fails" is a verdict on word search, not on selection.
- **Road B was never tested.** The team's own prototype (companion/companion_brain.py) already contains road B: derived pages with receipts. Its header says to test extraction from messy chat first. Neither was tested [C].
- **Mistakes by the assistant that ran it [C, report §9]:**
  - Its token estimate was half the real count, so the budget ran out before 21 time-ordering questions were run.
  - The scorer was wrong on 4.3% of answers.
  - A client bug forced Part B to be withdrawn.
  - Haiku judged its own answers.
  - A 64-token cap cut off 514 of 995 thinking-off answers. That cap still applies to the pending 4B thinking-off arm. Fix it before that arm runs: whether the phone can answer without thinking is the most product-relevant number still pending.
- **Everything is synthetic, with one relation per person.** I found no run on a real phone and no real usage [C].
- **Credit where due:** the pre-registration and the disclosure are unusually honest.

**What 99.9% vs 50.2% means for a user [C]:**
- If you name the person and ask a fact or an update, it is about 99% right either way.
- "My friend who teaches piano": right 1 time in 120.
- "Who's the one allergic to sesame?": 1 in 5.
- "Kofi's neighbour's pet?": 1 in 4.

Of the misses, 86% are "not mentioned", so it feels forgetful rather than dishonest. But 7% of hard answers were confidently wrong. 28 answers gave Kofi's own pet. When several people were called Bram, it picked one without asking 24 times out of 30.

**2. The token worry**
The owner is partly right, but for different reasons. The notebook is read once and kept: the first read took 4.7 s and later questions 0.55 s on the server [C]. On a phone, the user would feel this [E]:

- **Waiting.** Searching a big notebook needs thinking: Haiku scored 28.8% without it [C]. One 4B answer used 638 thinking tokens and took 10 s on a server GPU [C]. Phone memory is about 10 times slower, so expect roughly 40 s to 2 minutes per memory question, getting slower as the notebook grows.
- **Battery and heat.** About 1% of battery per thinking answer, based on about 0.2 mWh per token measured on an iPhone [C]. The phone gets warm, and the first read takes minutes. Qwen3.5's linear layers are immature on Android: there is a GPU corruption bug, the NPU kernel exists only in a fork, and one runtime falls back to slow step-by-step reading [C].
- **Storage.** Six years of notes as text is only a few MB. The model's cache is the problem: about 2 GB now and 8-16 GB at 500K tokens. It passes the 262K context limit around year 3, and a 12 GB phone runs out of room within months.
- **Deletion.** Deleting an old note forces a re-read of everything after it. Until that happens, the fact still lives in the linear layers' state, which breaks "real deletion".
- **Money.** Nothing on the phone. In the cloud it is about 0.5 cent per answer and rising [C], which adds up to hundreds of dollars per heavy user [E].

Road B pays its tokens at night while the phone charges, and reads pages of about 200 tokens while chatting. That gives answers in seconds, if the 4B can answer without thinking. Haiku could, at 96.4% [C].

**3. Association (road B)**
The owner's instinct is right. Perfect pages is exactly what a perfect B would produce: "match two unnamed notes" becomes "find the one page that matches". ChatGPT, Letta, Mem0 and Zep all keep profiles or graphs of this kind [R].

They let users down in five ways [C, mostly user reports]:
- summaries drift;
- deleted memories come back through chat history;
- whole memories get wiped;
- memories get injected where nobody asked for them (Willison's "Half Moon Bay" sign);
- benchmark claims nobody trusts (Mem0 vs Zep).

How users take each failure:
- **Forgiven:** "Remind me?" or "Which Bram?"
- **Hurts:** the wrong person, or a confident stale fact about something emotional, such as asking about a parent who has died.
- **Destroys trust:** a deleted fact coming back through a link. That breaks the project's one distinguishing promise.

So B must not keep free-text summaries: every line carries receipts and is rebuilt from current facts. One caveat: the test world flatters B, because each person has one link. A real Mira has dozens.

**4. Vote**
(i) Next test: B built by the 4B itself, with a deletion audit (70% confident). Then C as an arm inside that test, then A timed on a real phone, then D, then E.

(ii) Long-term phone memory: D centred on B, meaning pages plus the raw journal, with a small window of recent days held in context (60% confident). Then B, C, A, E. E stays a research line.

Conditional on the pending result, with X = 90% and Y = 75%:
- At 90% or more: keep A only as the recent-days window, and only if it answers within 10 s on a real phone.
- At 75% or less: drop A on the phone.
- In between: keep A with a smaller window.
- In every case: if the median thinking is over about 500 tokens, A fails on a phone whatever its accuracy.

No score rescues pure A at six years.

**5. One test: the phone model keeps its own address book**
The idea in plain words:
1. Feed the 4B the 5,025 notes in order, with a third rewritten as chat ("ran into Joao, he's moved to Lyon").
2. The 4B maintains one page per person, with links and receipts.
3. It answers by searching the pages by words and by meaning, then following one link. Run it with thinking off (256-token cap) and with thinking on.
4. Score these separately: the pages against the generator's truth, whether it finds the right page, accuracy, and confident wrong answers.
5. Delete 100 notes, rebuild, and ask about them.

- **Pass:** at least 85% on the hard set with thinking off, at most 2% confident wrong, 0 of 100 deleted facts come back, and a median answer of at most 150 tokens.
- **Stop:** below 70%, or any deletion leak caused by the design.
- **Cost:** about 3-5 L40S hours (about $10-20 [R]) plus a week of work. No judge is needed.

What each outcome decides:
- **Pass:** B becomes the spine. The team then uses it on their own chats for a month.
- **Pages wrong:** writing is the bottleneck. Try the 27B as a nightly writer.
- **Pages right but not found:** add C.
- **Any leak:** nothing ships.

Sources:
- [llama.cpp issue 20603](https://github.com/ggml-org/llama.cpp/issues/20603)
- [Qualcomm Hexagon kernel PR](https://monitoring-git.chickenkiller.com/qualcomm/llama.cpp/pull/5)
- [ExecuTorch fallback note](https://vvcxhwvm.pages.dev/software-mansion/react-native-executorch/pull/1102)
- [MELT](https://www.alphaxiv.org/overview/2403.12844)
- [ChatGPT deleted memories](https://aiweekly.co/alerts/chatgpt-recalls-explicitly-deleted-memories-years-later)
- [OpenAI status](https://status.openai.com/incidents/01K9D7DASB76TK1DEGPMG6ZAM4)
- [Willison](https://simonwillison.net/2025/may/21/chatgpt-new-memory/)
- [Zep vs Mem0](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/)
- [Kindroid](https://www.roborhythms.com/kindroid-memory-not-working/)
