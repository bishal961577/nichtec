# Test 2 sign-off (round 2): AI research scientist

## Round 2 sign-off, AI research scientist: GO WITH LISTED FIXES

I reviewed commit d36350b. The working tree kept changing while I worked (uncommitted edits to `brain/messy_index.py` around 21:24), so the freeze has to be cut from a committed state with a fresh stub run. I modified no repository files: my stub run wrote to the scratchpad, and the changes under `results/` come from another process. I did not open `sealed/`.

### Flaw by flaw

1. **Relation direction: FIXED [checked].**
   - Dev: 0 wrong of 30 two-hop questions on one-way relations and 0 of 31 "X's mentor" mentions.
   - Eight new test seeds: 0 wrong of 34–45 two-hop questions and 0 of 26–50 mentions each, with 0 self-check errors.

2. **Sealed held-out wordings: PARTLY.** The mechanism has three holes:
   - **The checker only rejects extra placeholders, not missing ones.** I passed `check_heldout` a fact template with no `{v}`, a question with no `{m}` and a "lure" that states the fact. It returned no errors and the world built (b5.py). Only `MW.check`'s duplicate-question test caught anything, and `main()` never calls `MW.check`.
   - **The seal is an honour rule.** The file sits encoded in the repository, where the author of the reading code can decode it. The hash proves the file was not changed; it cannot prove nobody read it.
   - **The hash was not enforced in d36350b.** `HELDOUT_SHA256` was `None` there, so `load_heldout` accepted any file. The working tree now sets it.

3. **Scorer: PARTLY [checked, a3.py and b3.py].**
   - **Fixed:** every false positive I listed (update hedges, hedged who-answers, lists of dates, "still", empty answers, any "which") and the false negatives ("egg", "peanut", "a cat", "drum kit", "fire fighter", 28/07/2026, "lido", "games night", "Your mum", nicknames, earlier values marked as earlier).
   - **New, from the short keys** (matched at word starts):
     - "been" counts as bee stings, so "She has been allergic to peanuts" scores wrong;
     - "Not in my notes – nothing has been recorded" counts as a guess on never-stated questions and as a leak on deleted ones;
     - "category" counts as tabby cat.
   - **New false positives on updates:** `OLD_MARK` treats "moved", "switched", "from" and "not" as marking an earlier value. So "Moved to Graz from Valencia", "It's not Valencia, it's Graz" and "Switched to the trumpet from the mandolin" score right even though each gives the old value as current (7 such cases).
   - **New false negative:** a bare "or" counts as a hedge, so "Felix Molnar (works with Kofi or Mira)" scores wrong.

4. **Leak rule and never-stated control: FIXED in design.** Values shown to a reader are 0 by construction, and the control exists.
   - Residual: `deletion_audit` still matches keys as substrings. One surviving row reading "ambassador", for a person whose deleted value is bass guitar, gives `rows_left = 1`, which is a STOP (b5.py). The working tree still has this.

5. **Decision logic: FIXED [checked code].** SEARCH now has an absolute bar, READ-LIMITED and WRITER-LIMITED are added, STOP is design-only, and the audit is blind and covers both arms.
   - People-resampled bootstrap (`clustered_diff`): FIXED. It resamples people. In the stub, B − C is 65.8 points [60.7, 70.7].

6. **Beacon seed: PARTLY.** The protocol is sound as written but is not in code. Nothing derives the seed from the drand round and asserts it equals `--test-seed`, and no rule says the first FREEZE commit binds.

7. **write_quality: REGRESSED in d36350b, fixed in the uncommitted working tree.**
   - In d36350b the anchor of "Kofi's flatmate" and the source of second-hand news count as mentioned. A writer that files every such fact under Kofi or under the source shows 0.0% wrong merges (b4.py).
   - The working tree counts by whose fact it is. The same two writers then show 4.6% and 3.6% wrong merges, and 24.6% and 19.4% on the hard cases. This must be committed.
   - Dilution by small-talk rows: FIXED.

8. **C's search, kin lures, meeting openers: FIXED [checked].**
   - Each C follow-up adds 15 new messages up to 80, BM25 uses the same tokenizer, and the month filter works.
   - 0 kinship lures.
   - 0 of 170 event_last golds have a later implied meeting; 1 event_what has a second meeting in the same month.

**Stub, d36350b:** BETWEEN. B 92.2%, oracle 96.6%, C 55.6%. Every PASS condition holds except forgetting, at a mean of 98.0% with the worst person at 88.5%. The working tree changes that line to "worst person ≥ 95%". That is stricter, but it changed after the stub was seen, so freeze it in a commit with that reason given.

### Fixes required before the freeze
1. Match value keys at word ends as well as starts, with an optional plural, in `named_values`, `deletion_audit` and `leak_shown`.
2. Limit `OLD_MARK` to unambiguous words right before the value (drop "moved", "switched", "from", "not", "had", "was"). Treat bare "or" as a hedge only when it sits between two names.
3. Make `check_heldout` require the needed placeholders per field. Run `MW.check` in `main()` and abort on any error. Make a `None` hash fatal for the scored run.
4. Encrypt the sealed file with drand timelock (tlock) to round R, or hand the key to the owner. Pre-register that any test-world kind where the perfect-pages arm (pages, thinking on) scores below 95% is hand-checked and reported both ways.
5. Code the beacon: `--beacon-round` recomputes and asserts the seed, and the freeze commit is pushed before round R.
6. Commit the owner-based merge count. Hand-check any trigger before calling STOP.

### Updated forecast (new outcome map)
WRITER-LIMITED 50%, READ-LIMITED 20%, BETWEEN 12%, STOP 8%, PASS 6%, SEARCH 4%.
B hard set about 70% (62–80), oracle 85–92%.

### Final verdict
GO WITH LISTED FIXES. All six are small; fixes 1 and 3 decide whether the gold answers and the scorer can be trusted on the sealed test world.

Scripts are in the session scratchpad: `b1.py` (direction), `b3.py` (scorer), `b4.py` (adversarial writers), `b5.py` (held-out checker and deletion audit), `stub2.log` (stub).
