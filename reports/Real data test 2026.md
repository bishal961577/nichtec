# Does a Small Model Learn a Real Person's Taste? (October 2026)

The ingredients test (section 9 of `Learning on the job from first principles 2026.md`) passed on a toy world:
- a 4B model that named traits, with code combining them, matched a 27B model reading its history;
- the margin was +1.1 ± 1.9, on 20 tasks;
- the animals were known, the rules exact, and there were two traits per rule.

The owner asked for the real thing at a much larger scale, so that a lucky result cannot pass. This plan was written
and committed before any code or run.

## 1. The task: learning one person's movie taste from their own ratings

This is the owner's goal in miniature: one AI per person, learning that person after it is built.

**Data:** MovieLens 1M (Harper and Konstan 2015, GroupLens).
- 1,000,209 real ratings (1–5 stars) by 6,040 people of 3,883 movies, made in 2000–2003.
- `ml-1m.zip` sha256 `a6898adb…54e20`.
- Used for research under its license, not redistributed: it is downloaded at run time into the git-ignored `data/`.

**People:** 300, drawn at random (seed 0) from those who meet all of:
- at least 120 ratings;
- their last 40 ratings include at least 8 liked and 8 not liked;
- the 80 ratings before those include at least 10 liked and 10 not liked.

**Per person:**
- **Experience:** the 80 ratings just before the last 40, in the order made.
- **Test:** the last 40 ratings, in time. This is a time split, so the person's taste may have drifted, as in real
  life.
- **Liked** means 4 or 5 stars.

**The movies:**
- Each is shown by title and year only. Titles are put in normal form: "Matrix, The (1999)" becomes
  "The Matrix (1999)".
- No genres and no other people's ratings are shown to the model.

**Measure:** for each person, the test AUC: the chance that a liked test movie is scored above a not-liked one, where
0.5 is guessing. This is averaged over the 300 people. Differences are paired by person, with 95% intervals from
10,000 bootstrap resamples of people.

## 2. Agents (the same model for each)

- **A, history:**
  - The 80 rated movies, in the order rated, each marked "liked" or "not liked", then the test movie and "Will this
    person like it? Answer yes or no."
  - The score is log P(yes) − log P(no), from the top-20 log-probabilities of the first token.
- **W, taste in words:**
  - The model writes the person's taste in a few sentences from the 80 ratings, sorted by title. It uses greedy
    decoding, with at most 200 tokens.
  - For each test movie, the model answers "Will this person like it?" given that description. The score is
    log P(yes) − log P(no).
- **E, the model names questions, code learns the person:**
  1. **Naming:** from the 80 ratings, sorted by title, the model lists up to 12 short yes-or-no questions about a
     movie that might predict this person's liking. That is 2 samples at temperature 0.8, top-p 0.95, top-k 50,
     with at most 300 tokens. Lines ending in "?" are kept, de-duplicated, and capped at 12.
  2. **Reading:** the model answers each question for each of the person's 120 movies with "Movie: …, Question: …,
     Answer yes or no". The answer is yes if log P(yes) > log P(no) in the top 20. Answers are cached by (question,
     movie) across people.
  3. **Learning (code):** an L1-penalised logistic regression (scikit-learn, liblinear) on the person's 80 movies.
     - Its strength is chosen from {0.05, 0.1, 0.3, 1, 3} by 5-fold stratified cross-validated AUC within the 80
       (seed 0).
     - It is then refitted on all 80, and it scores the 40 test movies.

     The L1 penalty is the simplicity principle for fuzzy lessons: few questions, each with a weight.

## 3. References (no language model)

- **Genres:** the same learner on the dataset's own 18 genre tags plus 7 decade tags. It shows whether the model's
  questions add anything to ready-made tags.
- **Popularity:** each test movie's smoothed mean rating by other people before this person's test window began:
  (sum + 5 × global mean) / (count + 5). It is not personal. A personal learner should beat it.
- **Collaborative filtering:** alternating least squares, with 32 factors, regularisation 0.1 and 15 iterations.
  - It is trained on every rating except the 300 test windows. Other people's later ratings leak, so it is
    optimistic.
  - It is the classic recommender, which uses everyone's data. It is reported only to show what is possible.

## 4. Models

- **Qwen3.5-4B,** phone-sized, is the primary model.
- **Qwen3.5-27B-FP8** is for comparison.
- Both run on dedicated Modal endpoints, one at a time, with thinking off and 128 parallel calls.
- The calls are roughly 12,000 for A, 12,300 for W, and up to 430,000 short reading calls for E, fewer after the
  cache. That is likely 20–70 minutes per model.
- No forecast is made.

## 5. Lines, fixed now

**Checks (per model), each of which makes the result inconclusive if it fails:**
- **K1, answering:** at most 10% of A, W and reading calls have neither yes nor no among the top 20 tokens.
- **K2, signal:** collaborative filtering reaches a mean AUC of at least 0.60. Below that, the test windows are too
  noisy for anything to be ranked, and nothing is concluded.

**Test 1, the small model (4B):**
- **Pass:** E is above both A and W by at least 0.03 AUC, each with the 95% interval above 0.
- **Stop:** the 95% interval of E minus the better of A and W includes 0 or lies below it. Splitting the jobs then
  does not help on real data, and this line of work ends.
- In between otherwise.

**Test 2, small against big (4B-E against 27B-A), run whatever test 1 shows:**
- **Pass ("matches"):** the 95% interval of 4B-E minus 27B-A lies entirely above −0.02.
- **Stop:** the interval lies entirely below −0.02.
- In between otherwise.

**Reported without lines:**
- 27B's A, W and E;
- E against genres and against popularity;
- collaborative filtering;
- the questions E's learner gave weight to;
- reading coverage;
- the number of calls.

## 6. What it stands on

**Work that already exists:**
- Language models predicting a user's ratings in context do worse than collaborative filtering unless fine-tuned
  (Kang et al. 2023, arXiv 2305.06474; TALLRec, arXiv 2305.00447).
- Knowledge from a language model added as item features (KAR, arXiv 2306.10933).
- User profiles in natural language. That is W.

**What is tested here:** whether, for a small model learning one person from 80 ratings, splitting the jobs (the model
names and reads, code learns) beats the model reading the history or writing the taste in words, and whether it
matches a model about seven times larger.

## 7. Test 1 result (Qwen3.5-4B, 19 October 2026): stop

**How it ran:**
- From the cloud session against a dedicated Modal endpoint.
- The first attempt ran at about 33 calls a second, whatever the prompt length, too slow for the 2-hour window. It
  was stopped after the A answers and the W answers were cached.
- Before the restart, batched requests were added (commit "Add batched yes/no answers"). On 128 saved answers they
  gave the same yes or no on all 128, with a mean difference of 0.043. The measurement is unchanged.
- The restart reused the cache: 10.6 minutes and 396,446 answers. None was outside yes or no.
- The raw data are in `results/taste_Qwen3.5-4B/`.

| Agent | Mean AUC (300 people) |
|---|---|
| A, history in the prompt | 0.614 |
| W, taste written in words | 0.572 |
| **E, model names questions, code learns the person** | **0.537** |
| Genres + decades, same learner (no model) | 0.552 |
| Popularity before the test window (not personal) | 0.679 |
| Collaborative filtering, everyone's data | 0.740 |

**Checks:** both passed. K1: 0.0% of answers outside yes or no. K2: collaborative filtering at 0.740.

**Line:**
- E minus A was −0.078 [−0.093, −0.061]. E minus W was −0.035 [−0.052, −0.017].
- **Stop.** On real data, splitting the jobs is worse than the model reading the history, and worse than the model
  writing the taste in words.
- It is even slightly below the dataset's genre tags (−0.015 [−0.031, −0.000]).
- **This line of work ends.**

**Diagnosis from the saved data** (no new calls):
1. **The toy pass did not carry over.** On animals the split matched a model about seven times larger. On real
   people it loses by a wide margin. E beat A for only 88 of the 300 people.
2. **E's questions are mostly genres:** "Is the movie a comedy?" (38 people), "a horror film?" (24), "a romantic
   comedy?" (14). With 80 ratings, the learner found nothing worth weighting for 51 people, who got a flat
   prediction. The mean was 5.7 weighted questions.
3. **What A has and E lacks is general knowledge of which movies are good.**
   - A's scores follow other people's ratings: a mean Spearman of 0.36 with popularity.
   - Popularity alone, which is not personal at all, beats every personal agent (0.679).
   - Liking a movie is mostly about the movie, the same for everyone, plus a smaller personal deviation.
   - E learned each person from scratch on 80 ratings, with no prior about what most people like, so it could only
     model the small personal part. It modelled that part poorly.
4. **In first-principles terms:** a learner with 80 examples needs a strong prior, and learns the deviation from it.
   The toy tasks had no shared prior, because hidden rules were arbitrary, which is why splitting looked good there.
   Real tastes have a large shared component.

**Next, as pre-registered:**
- Test 2 (4B-E against 27B-A) is to be run whatever test 1 showed.
- No redesign of E is made, because the line of work has ended.
- A new idea from this diagnosis (a shared prior plus a personal deviation, with no other people's data at run time)
  would be a new pre-registration, decided with the owner.

## 8. Test 2 and the 27B results (19 October 2026): stop

**How it ran:**
- The 27B run lost its first attempt to a restart of the cloud machine. It was resumed from the saved generations.
- It answered A and W for all 300 people, at 25–29 answers a second.
- E's reading ran at about 32 answers a second, which would have needed about 3 more hours. That stage was stopped,
  and E was not run on 27B (commit "Add --no-e").
  - Test 2's line needs only 27B-A, which was complete.
  - 27B-E was "reported without a line", and test 1 had already ended E's line of work.
  - The cut was made for speed and cost, before any 27B result was computed.
- **K1** (answers outside yes or no) is counted from the cached answers. 27B: 0 of 24,000. 4B: 10 of 420,445, which
  is 0.002%.

| Agent | 4B | 27B |
|---|---|---|
| A, history in the prompt | 0.614 | **0.635** |
| W, taste written in words | 0.572 | 0.605 |
| E, split jobs | 0.537 | not run |
| Popularity (not personal) | 0.679 | 0.679 |
| Collaborative filtering (everyone's data) | 0.740 | 0.740 |

**Test 2 line:**
- 4B-E minus 27B-A was −0.098 [−0.114, −0.082]. The whole interval lies below −0.02. **Stop.**
- The small model with split jobs does not match the large model reading its history. On real data it is far below.

**Also measured:**
- A larger model learns a person a little better from the same 80 ratings: A rose 0.021 from 4B to 27B, and W rose
  0.033.
- **Even 27B reading the history loses to plain popularity:** −0.044 [−0.062, −0.026]. From 80 ratings, neither
  model's personal learning beats a list of what most people liked before.
- W is below A at 27B too: −0.030 [−0.042, −0.019]. Writing the taste in words loses information.

**Where this leaves the work:**
- Both lines of the real-data plan stopped.
- The toy results (the ingredients test) did not survive real data at scale.
- The diagnosis in section 7 stands: on real tastes the shared part (what most people like) is larger than the
  personal part, and no agent tested here starts from it.

## 9. Next test: a shared starting point plus a personal difference (pre-registered 19 October 2026)

Written and committed before any computation. It uses only answers already saved from the runs above, so it needs no
model calls.

**First principles:**
- Section 8 showed two things. Liking a movie is mostly shared: popularity alone scored 0.679. And no agent that
  learned a person from scratch beat that.
- A learner with 80 examples should start from what is generally true and learn only how this person differs.
- On a phone, the shared part can be a public list shipped with the app, such as which movies people generally like.
  It needs no one else's personal data on the device. The personal part is learned on the device.

**The question:** does learning the person on top of the shared list beat the shared list alone?

**Agents** (the same 300 people, 80 ratings, the next 40 tested, and per-person AUC):
- **P, the shared list:** the popularity score from section 3. Its mean AUC was 0.679. It is the starting point.
- **P+E (4B):**
  - The same L1 logistic regression as E, with the same strength grid and cross-validation (seed 0).
  - Its inputs are the 4B model's saved yes-or-no answers to its own questions, plus the popularity score
    standardised within the person's 120 movies.
  - Popularity is known at the start of the test window, so no test rating is used.
- **P+A (4B) and P+A (27B):** for each test movie, the average of its popularity rank and its rank by A's saved
  yes-or-no margin, weighted equally. Nothing is fitted. A was only asked about test movies, so no weight can be
  learned from the 80.
- **P+W (4B) and P+W (27B):** the same, with W's margin.
- **P+genres (no model):** the same learner as P+E, on the genre and decade tags plus popularity. It shows whether the
  model's questions add anything beyond the dataset's own tags once the shared list is there.

**Sanity check (stops the test if it fails):** rebuilding E from the saved answers must reproduce the 4B run's E for
every person, within 0.001 AUC.

**Lines:**
- **Main (P+E against P):**
  - **Pass:** P+E is at least 0.02 above P, with the 95% interval above 0. On-device personal learning then adds
    something real to a shared starting point.
  - **Stop:** the interval includes 0 or lies below it. From 80 ratings, the personal part learnt this way adds
    nothing.
  - In between otherwise.
- **Secondary, the same thresholds, each reported as pass, stop or in between:** P+A (4B), P+A (27B).
- **Reported without lines:**
  - P+W on both models;
  - P+genres;
  - P+E against P+genres;
  - each against collaborative filtering (0.740, everyone's data).
- Intervals are paired by person, from 10,000 bootstrap resamples (seed 0).

## 10. Result of the shared-list test (19 October 2026): stop

Computed with `brain/prior.py` from saved answers, with no model calls. **Sanity check passed:** E rebuilt from the
saved answers matched the 4B run for all 300 people, within 0.001. The raw data are in `results/prior_test/`.

| Agent | Mean AUC | Minus P [95% interval] |
|---|---|---|
| P, shared list (popularity) | 0.679 | — |
| P+E, personal questions learned on top of P (4B) | 0.666 | −0.012 [−0.021, −0.004] |
| P+genres, dataset tags on top of P (no model) | 0.679 | +0.000 [−0.008, +0.009] |
| P+A (4B) | 0.672 | −0.007 [−0.015, +0.000] |
| P+A (27B) | 0.687 | +0.008 [−0.001, +0.017] |
| P+W (4B) | 0.659 | −0.020 [−0.030, −0.010] |
| P+W (27B) | 0.677 | −0.002 [−0.012, +0.009] |
| Collaborative filtering, everyone's data | 0.740 | +0.061 [+0.051, +0.072] |

**Lines:**
- **Main (P+E against P): stop.** Learning the person on top of the shared list made it slightly worse. P+E kept
  popularity in its model for 289 of the 300 people, with 3.8 personal questions on average.
- **Secondary:**
  - P+A (4B): stop. The interval reaches 0.
  - P+A (27B): stop. The interval includes 0.
- **Reported without lines:**
  - P+genres equals P.
  - P+E is below P+genres: −0.013 [−0.022, −0.003].

**What it means:**
- From a person's own 80 ratings, neither the model's questions, its reading of the history, its description of the
  taste, nor the dataset's genre tags add anything reliable to the shared list.
- Collaborative filtering does add something: +0.061 over the same list. Its person part is also learned from the
  person's own ratings only. With the item factors fixed, alternating least squares solves each person's vector from
  that person's ratings alone, which is exactly what a phone could compute.
- The difference is the space it learns in. The item factors are a "taste space" learned from how millions of
  ratings co-occur. The model's descriptive knowledge of movies (genres, themes, fame) does not give a space in which
  80 ratings can locate a person.
- So fast personal learning needed a representation learned in advance from many people's behaviour in the same
  domain. A general language model's knowledge was not that representation, and code on top of it could not supply
  it.
- This is the established design of on-device and federated recommendation: shared item factors with a private user
  vector. It is not new, but it is the measured answer here.

**Where the real-data line of work stands:**
- Every model-based way of learning a person tested here stopped:
  - reading the history;
  - writing the taste in words;
  - splitting the jobs;
  - each of these on top of a shared list.
- The only approach that learned the person was a representation built from population data, which no model call
  supplied.
