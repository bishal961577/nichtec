# Moderator's checks after round 1 (given to all four for round 2)

Every claim below was checked by the moderator against the repository (code, answers files) or with a script.

1. **The search2 "DONE" gate (neuroscientist, researcher, architect): confirmed.** The follow-up step runs with thinking off,
   greedy, 64 tokens. This was registered that way (report section 7: "the model, thinking off, reads them and replies
   DONE or writes one follow-up search query"). So the phone line fails fairly as registered, but it measures this design.
   DONE counts and accuracy:
   | Kind | 4B DONE | 4B acc if DONE / if searched again | Claude DONE | Claude acc if DONE / if searched again |
   |---|---|---|---|---|
   | described | 117/120 | 0% / 33% | 44/120 | 0% / 24% |
   | who | 59/60 | 19% / 100% (n=1) | 46/60 | 24% / 64% |
   | first name | 51/60 | 49% / 89% | 8/60 | 100% / 100% |
   | two-hop | 105/200 | 1% / 54% | 19/200 | 5% / 75% |

2. **"Accuracy flat with notebook size" (researcher): confirmed as a wrong comparison.** The middle sizes (100, 1,000 notes)
   ran only 635 name-keyed questions (two, reword, upd, same, retract, prev). On the same 635 questions, Claude Haiku 5.5:
   perfect pages 97.5%, 100 notes 95.9%, 1,000 notes 95.7%, 5,000 notes 93.4%. A small slide, not flat. The hard kinds were
   never run at the middle sizes.

3. **Thinking-off cap (founder): confirmed.** Claude, thinking off, 64-token cap: at 5,000 notes 514 of 995 answers were cut
   off (accuracy 11.5% cut vs 47.4% not cut); at perfect pages 9 of 965; search2 112 of 995. So "thinking off at 5,000:
   28.8%" mixes weaker answering with cut-off answers. The 4B's thinking-off arms (perfect pages, search2, 100 notes) have
   not run yet and use the same registered 64-token cap.

4. **Person pages (architect): confirmed with the moderator's own script** (same notes, grouped into one page per person by
   the full name in each note; the same BM25 code; the question text as the query):
   - described people: right page ranked 1st 97.5%, top 3 100% (n=120);
   - who is: 1st 100% (n=60);
   - first name: 1st 96.7%, top 3 100% (n=60);
   - two-hop: the asker's page ranked 1st and it names the hidden middle person 99.0% (n=200); the middle person's own page
     is not reached without following that link (1st 0%, top 3 80.5%).
   Correction to the architect: two-hop needs the link step; "first for two-hop" is true of the asker's page, not the
   target's. Caveat for everyone: every note carries the full name, so filing is trivial here.

5. **LongMemEval (researcher): confirmed.** Full histories median 180,879 Claude tokens (the brief's "~115K" is LongMemEval's
   own count); the judge was Claude Haiku 5.5 with thinking off, grading its own model's answers; 28 of 150 questions not run,
   mostly temporal.

6. **Thinking text not saved (neuroscientist, researcher): confirmed and fixed for the rest of the run.** vLLM 0.33 returns
   the thinking as `reasoning`; the script read `reasoning_content`. Logging only; scoring unchanged. The run was restarted
   (resumable) and every answer from then on stores its thinking text, including the pending 5,000-note arm.

7. **4B run status:** perfect pages and search2 (thinking on) done; bm25 nearly done; then 100 notes (thinking on), the three
   thinking-off arms, world edits, and the 5,000-note whole-notebook arm last.
