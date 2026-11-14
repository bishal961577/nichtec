# LongMemEval-S (cleaned), 150 questions: claude-haiku-5-5

Pre-registration: reports/Notebook reading at shipping size 2026.md, Part C. Judge: claude-haiku-5-5 (thinking off). Thinking on. BM25 top-10 sessions hold all evidence sessions for 92% of questions.

| Type | oracle | bm25 | full |
|---|---|---|---|
| knowledge-update | 96.0 [88-100] n=25 | 92.0 [80-100] n=25 | 96.0 [88-100] n=25 |
| multi-session | 88.0 [76-100] n=25 | 68.0 [48-84] n=25 | 88.0 [72-100] n=25 |
| single-session-assistant | 92.0 [80-100] n=25 | 88.0 [76-100] n=25 | 88.0 [76-100] n=25 |
| single-session-preference | 76.0 [60-92] n=25 | 72.0 [52-88] n=25 | 78.3 [61-91] n=23 |
| single-session-user | 92.0 [80-100] n=25 | 96.0 [88-100] n=25 | 100.0 [100-100] n=20 |
| temporal-reasoning | 96.0 [88-100] n=25 | 92.0 [80-100] n=25 | 100.0 [100-100] n=4 |
| all | 90.0 [85-95] n=150 | 84.7 [79-90] n=150 | 90.2 [84-95] n=122 |

## Verdict

- full history 90.2%, evidence only 90.2%, BM25 sessions 84.4% on 122 questions (28 excluded: errors such as context overflow). full/oracle 1.00 [0.94, 1.07]: LONG CONTEXT HOLDS on real chat (>= 0.9). Retrieval < full history.
