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
