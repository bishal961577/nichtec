# SEALED until the test 2 FREEZE commit

`messy_heldout.json.gz.b64` holds the held-out wordings for the test world of the messy chat index test
(`reports/Messy chat index test 2026.md`, amendment 1, sections 14.4 and 14.6). A separate agent wrote them; the author of
the reading code has not read them.

- SHA-256 of the decoded JSON: `7a5d7cd7d643d641df8706da71a07d32c80541a63adeff2add704a9ae4a7bdea`
- **Do not decode it before the FREEZE commit.** The seal relies on that rule: anyone with the repository can decode it. The
  hash proves it was not changed after this commit.
- At the scored run: `python brain/messy_index.py --split test --test-seed N --heldout "research_notes/Messy chat test review 2026/sealed/messy_heldout.json.gz.b64" ...`
  (the run refuses a file with a different hash).
- The agent checked it: `check_heldout` reported no problems, and `check` on test worlds built with seven seeds reported
  none either. It holds 132 attribute wordings, 46 frames, 40 relation phrasings, 4 small-talk lines and 24 activity
  phrases.
