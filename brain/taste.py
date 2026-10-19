"""Real-data test: does a small model learn a real person's movie taste? Pre-registered in
reports/Real data test 2026.md.

300 MovieLens-1M users; each learned from their own 80 ratings and tested on their next 40 (time split); liked = 4-5
stars; per-user AUC. Agents on one model: A (history in the prompt), W (taste written in words, then applied by the
model), E (the model names yes/no questions and answers them per movie; an L1 logistic regression learns the person).
References without a model: genre tags with the same learner, popularity before the test window, and collaborative
filtering (ALS).

  python brain/taste.py --refs                                             # references only, no model
  python brain/taste.py --stub --users 20                                  # bookkeeping check
  python brain/taste.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-4B --check
  python brain/taste.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-4B
  python brain/taste.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-27B-FP8
"""
import argparse
import bisect
import collections
import hashlib
import io
import json
import math
import os
import random
import re
import sys
import threading
import time
import urllib.request
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
SHA = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
N_USERS, STREAM, TEST = 300, 80, 40
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
CS = [0.05, 0.1, 0.3, 1, 3]
MAX_Q, NAME_SAMPLES = 12, 2
GEN = {"temperature": 0.8, "top_p": 0.95, "top_k": 50}
INTRO = "Here are movies one person has rated, marked liked (4 or 5 stars) or not liked (1 to 3 stars):\n"


# ---------------------------------------------------------------- data

def load():
    path = os.path.join(G.ROOT, "data", "ml1m", "ml-1m.zip")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(URL, path)
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    z = zipfile.ZipFile(io.BytesIO(raw))
    movies = {}
    for line in z.read("ml-1m/movies.dat").decode("latin-1").splitlines():
        mid, title, genres = line.split("::")
        movies[int(mid)] = {"title": normal_title(title), "year": int(title[-5:-1]), "genres": genres.split("|")}
    ratings = []
    for line in z.read("ml-1m/ratings.dat").decode("latin-1").splitlines():
        u, m, r, t = map(int, line.split("::"))
        ratings.append((u, m, r, t))
    return movies, ratings, sha


def normal_title(t):
    m = re.match(r"^(.+?), (The|A|An)( \(.+\))? \((\d{4})\)$", t)
    return f"{m.group(2)} {m.group(1)}{m.group(3) or ''} ({m.group(4)})" if m else t


def pick_users(ratings, n=N_USERS):
    by = collections.defaultdict(list)
    for u, m, r, t in ratings:
        by[u].append((t, m, r))
    elig = []
    for u in sorted(by):
        xs = sorted(by[u])
        if len(xs) < STREAM + TEST:
            continue
        s, te = xs[-(STREAM + TEST):-TEST], xs[-TEST:]
        ps, pt = sum(r >= 4 for _, _, r in s), sum(r >= 4 for _, _, r in te)
        if pt >= 8 and TEST - pt >= 8 and ps >= 10 and STREAM - ps >= 10:
            elig.append(u)
    chosen = sorted(random.Random(0).sample(elig, n)) if n < len(elig) else elig
    out = []
    for u in chosen:
        xs = sorted(by[u])
        out.append({"user": u, "stream": [(m, r >= 4) for _, m, r in xs[-(STREAM + TEST):-TEST]],
                    "test": [(m, r >= 4) for _, m, r in xs[-TEST:]], "t0": xs[-TEST][0]})
    return out, len(elig)


# ---------------------------------------------------------------- learner and metrics

def auc(scores, labels):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def fit_l1(Xs, ys, Xt):
    """L1 logistic regression; strength chosen by 5-fold stratified CV AUC on the stream (ties: the stronger
    penalty); refitted on the whole stream; returns test scores and the coefficients."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    Xs, Xt, ys = np.asarray(Xs, float), np.asarray(Xt, float), np.asarray(ys, int)
    if Xs.shape[1] == 0 or np.all(Xs.std(0) == 0):
        return np.zeros(len(Xt)), [], None
    best = None
    for C in CS:
        vals = []
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, ys):
            m = LogisticRegression(l1_ratio=1, solver="liblinear", C=C, random_state=0).fit(Xs[tr], ys[tr])
            vals.append(auc(m.decision_function(Xs[va]), ys[va]))
        score = (float(np.mean(vals)), -C)
        if best is None or score > best[0]:
            best = (score, C)
    m = LogisticRegression(l1_ratio=1, solver="liblinear", C=best[1], random_state=0).fit(Xs, ys)
    return m.decision_function(Xt), m.coef_[0].tolist(), best[1]


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------- references (no language model)

def references(movies, ratings, users, log):
    log("references: genres, popularity, collaborative filtering")
    decades = [(0, 1950), (1950, 1960), (1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 3000)]

    def gfeat(m):
        mv = movies[m]
        return [g in mv["genres"] for g in GENRES] + [lo <= mv["year"] < hi for lo, hi in decades]
    out = {"genres": [], "popularity": [], "cf": []}
    for u in users:
        sc, _, _ = fit_l1([gfeat(m) for m, _ in u["stream"]], [y for _, y in u["stream"]], [gfeat(m) for m, _ in u["test"]])
        out["genres"].append(auc(sc, [y for _, y in u["test"]]))
    # popularity: other people's smoothed mean rating before this person's test window
    per = collections.defaultdict(list)
    for uu, m, r, t in ratings:
        per[m].append((t, r))
    cum = {}
    for m, xs in per.items():
        xs.sort()
        ts = [t for t, _ in xs]
        cs = np.cumsum([r for _, r in xs]).tolist()
        cum[m] = (ts, cs)
    gm = float(np.mean([r for _, _, r, _ in ratings]))
    for u in users:
        sc = []
        for m, _ in u["test"]:
            ts, cs = cum[m]
            k = bisect.bisect_left(ts, u["t0"])
            s = cs[k - 1] if k else 0.0
            sc.append((s + 5 * gm) / (k + 5))
        out["popularity"].append(auc(sc, [y for _, y in u["test"]]))
    # collaborative filtering: ALS on every rating except the test windows
    held = {(u["user"], m) for u in users for m, _ in u["test"]}
    train = [(uu, m, r) for uu, m, r, t in ratings if (uu, m) not in held]
    uid = {x: i for i, x in enumerate(sorted({x for x, _, _ in train}))}
    mid = {x: i for i, x in enumerate(sorted({x for _, x, _ in train}))}
    mu = float(np.mean([r for _, _, r in train]))
    by_u, by_m = collections.defaultdict(list), collections.defaultdict(list)
    for x, m, r in train:
        by_u[uid[x]].append((mid[m], r - mu))
        by_m[mid[m]].append((uid[x], r - mu))
    k, lam, rng = 32, 0.1, np.random.default_rng(0)
    U, V = rng.normal(0, 0.1, (len(uid), k)), rng.normal(0, 0.1, (len(mid), k))

    def solve(F, rows, n_rows):
        out = np.zeros((n_rows, k))
        for i, xs in rows.items():
            idx = np.array([j for j, _ in xs])
            r = np.array([v for _, v in xs])
            A = F[idx].T @ F[idx] + lam * len(xs) * np.eye(k)
            out[i] = np.linalg.solve(A, F[idx].T @ r)
        return out
    for it in range(15):
        U = solve(V, by_u, len(uid))
        V = solve(U, by_m, len(mid))
    for u in users:
        sc = [float(U[uid[u["user"]]] @ V[mid[m]]) if m in mid else 0.0 for m, _ in u["test"]]
        out["cf"].append(auc(sc, [y for _, y in u["test"]]))
    return out


# ---------------------------------------------------------------- prompts

def mark(y):
    return "liked" if y else "not liked"


def lines(movies, pairs):
    return "".join(f"{movies[m]['title']}: {mark(y)}\n" for m, y in pairs)


def a_prompt(movies, u, m):
    return (INTRO.replace("rated,", "rated, in the order they rated them,") + lines(movies, u["stream"]) +
            f"\nNew movie: {movies[m]['title']}\nWill this person like it? Answer yes or no.")


def sorted_lines(movies, u):
    return lines(movies, sorted(u["stream"], key=lambda x: movies[x[0]]["title"]))


def w_prompt(movies, u):
    return (INTRO + sorted_lines(movies, u) + "\nDescribe this person's movie taste in a few sentences: what they "
            "like and what they dislike. Answer with the description only.")


def judge_prompt(movies, desc, m):
    return f"A person's movie taste: {desc}\nMovie: {movies[m]['title']}\nWill this person like it? Answer yes or no."


def e_prompt(movies, u):
    return (INTRO + sorted_lines(movies, u) + "\nList up to 12 short yes-or-no questions about a movie that could "
            "help predict whether this person likes it. Write one question per line, nothing else.")


def read_prompt(movies, m, q):
    return f"Movie: {movies[m]['title']}\nQuestion: {q}\nAnswer yes or no."


def parse_questions(replies):
    out, seen = [], set()
    for rep in replies:
        for line in rep.splitlines():
            q = re.sub(r"^[\s\-\*•\d\.\)\(]+", "", line).strip().strip("*").strip()
            key = re.sub(r"[^a-z0-9]", "", q.lower())
            if q.endswith("?") and 3 < len(q) <= 150 and key not in seen:
                seen.add(key)
                out.append(q)
    return out[:MAX_Q]


# ---------------------------------------------------------------- model access

QWEN_USER = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def batch_answers(llm, prompts):
    """yes/no margins for many prompts in one /v1/completions request, each wrapped in the Qwen3.5 chat template
    with thinking off. Checked against the chat endpoint on 16 prompts before use: same sign on all, margins within
    0.15 (the server's own repeat noise is up to about 0.8)."""
    body = {"model": llm.model, "prompt": [QWEN_USER.format(p) for p in prompts], "max_tokens": 1, "temperature": 0,
            "logprobs": 20}
    r = llm.request("POST", "/completions", body)
    out = [None] * len(prompts)
    for ch in r["choices"]:
        top = ch["logprobs"]["top_logprobs"][0]
        lp = {"yes": [], "no": []}
        for tok, v in top.items():
            w = tok.strip().lower()
            if w in lp:
                lp[w].append(v)
        floor = min(top.values()) - 1.0
        with llm.lock:
            llm.calls += 1
            llm.off_format += not (lp["yes"] or lp["no"])
        out[ch["index"]] = G.logsumexp(lp["yes"] or [floor]) - G.logsumexp(lp["no"] or [floor])
    return out


class Answers:
    """yes/no margins for prompts, cached on disk by prompt text (saved every 20,000 new answers). With batch=n the
    model is asked n prompts per request through batch_answers."""

    def __init__(self, llm, path, batch=0):
        self.llm, self.path, self.lock, self.batch = llm, path, threading.Lock(), batch
        self.m = json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else {}

    def save(self):
        if self.path:
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.m, f)
            os.replace(tmp, self.path)

    def get(self, prompts, log, label):
        todo = sorted({p for p in prompts if p not in self.m})
        log(f"  {label}: {len(prompts):,} answers needed, {len(todo):,} new")
        t0 = time.time()
        for i in range(0, len(todo), 20000):
            chunk = todo[i:i + 20000]
            if self.batch:
                parts = self.llm.map(lambda ps: batch_answers(self.llm, ps),
                                     [chunk[j:j + self.batch] for j in range(0, len(chunk), self.batch)])
                got = [x for part in parts for x in part]
            else:
                got = self.llm.map(lambda p: self.llm.answer(p)[0], chunk)
            with self.lock:
                self.m.update(zip(chunk, got))
            self.save()
            done = min(i + 20000, len(todo))
            log(f"    {done:,}/{len(todo):,} ({done / max(1e-9, time.time() - t0):.0f}/s)")
        return {p: self.m[p] for p in prompts}


GEN_CACHE, GEN_LOCK = {"path": None, "m": {}}, threading.Lock()


def generate(llm, prompt, n, seed, max_tokens, **kw):
    """Model replies, cached on disk by (prompt, n, seed, settings) so a resumed run reuses them."""
    key = json.dumps([prompt, n, seed, max_tokens, sorted(kw.items())])
    if key in GEN_CACHE["m"]:
        return GEN_CACHE["m"][key]
    r = llm.chat(prompt, n=n, seed=seed, max_tokens=max_tokens, **kw)
    out = [c["message"].get("content") or "" for c in r["choices"]]
    with GEN_LOCK:
        GEN_CACHE["m"][key] = out
    return out


def save_gen():
    if GEN_CACHE["path"]:
        with GEN_LOCK:
            with open(GEN_CACHE["path"], "w", encoding="utf-8") as f:
                json.dump(GEN_CACHE["m"], f)


class Stub:
    """No model: margins from genre keywords with noise; fixed questions; a fixed description."""

    def __init__(self, movies):
        self.movies, self.calls, self.off_format = movies, 0, 0
        self.by_title = {v["title"]: v for v in movies.values()}

    def map(self, f, xs):
        return [f(x) for x in xs]

    def answer(self, p):
        self.calls += 1
        rng = random.Random(p)
        m = re.search(r"Movie: (.*)\n", p) or re.search(r"New movie: (.*)\n", p)
        mv = self.by_title.get(m.group(1)) if m else None
        if mv is None:
            return rng.gauss(0, 1), []
        q = p.lower()
        g = next((g for g in GENRES if g.lower() in q), None)
        base = (1.0 if (g in mv["genres"]) else -1.0) if g else (0.5 if "Drama" in mv["genres"] else -0.5)
        return base + rng.gauss(0, 0.7), []

    def chat(self, prompt, n, seed, **kw):
        rng = random.Random(seed)
        if "questions" in prompt:
            qs = [f"Is it a {g.lower()} movie?" for g in GENRES]
            outs = ["\n".join(rng.sample(qs, 8)) for _ in range(n)]
        else:
            outs = ["Likes dramas and comedies; dislikes horror." for _ in range(n)]
        return {"choices": [{"message": {"content": o}} for o in outs]}


# ---------------------------------------------------------------- run

def run_model(llm, tag, movies, users, refs, args, log):
    out = args.out or os.path.join(G.ROOT, "results", f"taste_{tag}")
    cache = None if args.stub else os.path.join(G.ROOT, "data", f"taste_cache_{tag}.json")
    if not args.stub:
        GEN_CACHE["path"] = os.path.join(G.ROOT, "data", f"taste_gen_{tag}.json")
        if os.path.exists(GEN_CACHE["path"]):
            GEN_CACHE["m"] = json.load(open(GEN_CACHE["path"], encoding="utf-8"))
    R = {"model": tag, "users": len(users), "settings": {"stream": STREAM, "test": TEST, "Cs": CS, "max_q": MAX_Q,
         "name_samples": NAME_SAMPLES, "gen": GEN, "batch": args.batch, "no_e": args.no_e}}
    u0 = users[0]
    margin, top = llm.answer(read_prompt(movies, u0["test"][0][0], "Is it a comedy?"))
    qs0 = parse_questions(generate(llm, e_prompt(movies, u0), NAME_SAMPLES, 1, 300, **GEN))
    first = top[0]["token"] if top else ""
    R["check"] = {"reading_first_token": first, "reading_answers_yes_no": (not top) or first.strip().lower() in ("yes", "no"),
                  "questions_named": qs0[:12]}
    log(f"check: {json.dumps(R['check'], ensure_ascii=True)}")
    if not R["check"]["reading_answers_yes_no"] or not qs0:
        raise SystemExit("the model does not answer a reading question with yes/no first, or names no questions: stopping")
    if args.check:
        raise SystemExit("check done; no run")
    start = time.time()
    log("generating descriptions and questions")
    descs = llm.map(lambda iu: generate(llm, w_prompt(movies, iu[1]), 1, 1000 + iu[0], 200, temperature=0)[0].strip(),
                    list(enumerate(users)))
    names = llm.map(lambda iu: parse_questions(generate(llm, e_prompt(movies, iu[1]), NAME_SAMPLES, 2000 + iu[0], 300,
                                                        **GEN)), list(enumerate(users)))
    save_gen()
    A = Answers(llm, cache, batch=args.batch)
    hist = A.get([a_prompt(movies, u, m) for u in users for m, _ in u["test"]], log, "A, history")
    judged = A.get([judge_prompt(movies, d, m) for u, d in zip(users, descs) for m, _ in u["test"]], log, "W, judging")
    read = {} if args.no_e else A.get([read_prompt(movies, m, q) for u, qs in zip(users, names) for q in qs
                  for m, _ in u["stream"] + u["test"]], log, "E, reading named questions")
    per = []
    for i, (u, d, qs) in enumerate(zip(users, descs, names)):
        yt = [y for _, y in u["test"]]
        rec = {"user": u["user"], "description": d[:600], "questions": qs}
        rec["A"] = auc([hist[a_prompt(movies, u, m)] for m, _ in u["test"]], yt)
        rec["W"] = auc([judged[judge_prompt(movies, d, m)] for m, _ in u["test"]], yt)
        if args.no_e:
            rec["E"], rec["E_C"], rec["E_weights"] = None, None, {}
            for k in ("genres", "popularity", "cf"):
                rec[k] = refs[k][i]
            per.append(rec)
            continue
        Xs = [[read[read_prompt(movies, m, q)] > 0 for q in qs] for m, _ in u["stream"]]
        Xt = [[read[read_prompt(movies, m, q)] > 0 for q in qs] for m, _ in u["test"]]
        sc, coef, C = fit_l1(Xs, [y for _, y in u["stream"]], Xt)
        rec["E"] = auc(sc, yt) if len(set(np.round(sc, 12))) > 1 else 0.5
        rec["E_C"], rec["E_weights"] = C, {q: round(c, 3) for q, c in zip(qs, coef) if abs(c) > 1e-9}
        for k in ("genres", "popularity", "cf"):
            rec[k] = refs[k][i]
        per.append(rec)
    R["per_user"] = per
    R["calls"] = {"answers": getattr(llm, "calls", 0), "off_format": getattr(llm, "off_format", 0),
                  "minutes": round((time.time() - start) / 60, 1)}
    write(out, R, log)
    return R


def summarise(R):
    P = R["per_user"]
    has_e = all(p["E"] is not None for p in P)
    S = {k: float(np.mean([p[k] for p in P])) for k in ("A", "W", "genres", "popularity", "cf")}
    S["E"] = float(np.mean([p["E"] for p in P])) if has_e else None
    S["off_format_rate"] = R["calls"]["off_format"] / max(1, R["calls"]["answers"])
    S["W-A"] = boot([p["W"] - p["A"] for p in P])
    S["A-popularity"] = boot([p["A"] - p["popularity"] for p in P])
    if not has_e:
        S["top_weighted_questions"] = []
        return S
    for a, b in (("E", "A"), ("E", "W"), ("E", "genres"), ("E", "popularity")):
        S[f"{a}-{b}"] = boot([p[a] - p[b] for p in P])
    S["better_of_A_W"] = "A" if S["A"] >= S["W"] else "W"
    S["E-best"] = boot([p["E"] - p[S["better_of_A_W"]] for p in P])
    q = collections.Counter(qq for p in P for qq in p["E_weights"])
    S["top_weighted_questions"] = q.most_common(15)
    return S


def verdict(S):
    out = [f"K1, answering: {100 * S['off_format_rate']:.1f}% of answers neither yes nor no (line <= 10): "
           f"{'ok' if S['off_format_rate'] <= 0.10 else 'FAILED'}.",
           f"K2, signal: collaborative filtering mean AUC {S['cf']:.3f} (line >= 0.60): "
           f"{'ok' if S['cf'] >= 0.60 else 'FAILED'}."]
    if S["off_format_rate"] > 0.10 or S["cf"] < 0.60:
        return out + ["INCONCLUSIVE: a check failed."]
    if S["E"] is None:
        return out + ["Test 1's line is not applied on this model: agent E was not run here (see the report)."]
    ea, ew, eb = S["E-A"], S["E-W"], S["E-best"]
    if ea[0] >= 0.03 and ea[1] > 0 and ew[0] >= 0.03 and ew[1] > 0:
        out.append(f"PASS: E beats A by {ea[0]:+.3f} [{ea[1]:+.3f}, {ea[2]:+.3f}] and W by {ew[0]:+.3f} "
                   f"[{ew[1]:+.3f}, {ew[2]:+.3f}].")
    elif eb[1] <= 0:
        out.append(f"STOP: E minus the better of A and W ({S['better_of_A_W']}) is {eb[0]:+.3f} [{eb[1]:+.3f}, "
                   f"{eb[2]:+.3f}]; the interval includes or lies below 0. This line of work ends.")
    else:
        out.append(f"IN BETWEEN: E-A {ea[0]:+.3f} [{ea[1]:+.3f}, {ea[2]:+.3f}], E-W {ew[0]:+.3f} [{ew[1]:+.3f}, {ew[2]:+.3f}].")
    return out


def write(out, R, log):
    S = summarise(R)
    R["summary"] = S
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    L = [f"# Real-data test: {R['model']}", "", "Pre-registration: reports/Real data test 2026.md. Mean per-person test "
         f"AUC over {R['users']} MovieLens-1M users (80 ratings learned, next 40 tested). 0.5 = guessing.", "",
         "| Agent | Mean AUC |", "|---|---|",
         f"| A, history in the prompt | {S['A']:.3f} |", f"| W, taste written in words | {S['W']:.3f} |",
         (f"| **E, model names questions, code learns the person** | **{S['E']:.3f}** |" if S["E"] is not None else
          "| E, model names questions, code learns the person | not run on this model |"),
         f"| Genres + decades, same learner (no model) | {S['genres']:.3f} |",
         f"| Popularity before the test window (not personal) | {S['popularity']:.3f} |",
         f"| Collaborative filtering, everyone's data (reference) | {S['cf']:.3f} |", "",
         "Paired differences, mean [95% bootstrap interval over people]:",
         f"- W - A: {f3(S['W-A'])}", f"- A - popularity: {f3(S['A-popularity'])}"] + (
        [f"- E - A: {f3(S['E-A'])}", f"- E - W: {f3(S['E-W'])}", f"- E - genres: {f3(S['E-genres'])}",
         f"- E - popularity: {f3(S['E-popularity'])}"] if S["E"] is not None else []) + ["",
         f"Answer calls: {R['calls']['answers']:,}; neither yes nor no: {100 * S['off_format_rate']:.1f}%; "
         f"{R['calls']['minutes']} min.",
         "Questions most often given weight: " + "; ".join(f"{q} ({n})" for q, n in S["top_weighted_questions"]),
         "", "## Verdict (test 1)", ""] + [f"- {v}" for v in verdict(S)]
    L += ["", f"Check before the run: {R['check']}"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))
    compare(log)


def compare(log):
    """Test 2: 4B-E against 27B-A, once both runs exist."""
    a = os.path.join(G.ROOT, "results", "taste_Qwen3.5-27B-FP8", "result.json")
    b = os.path.join(G.ROOT, "results", "taste_Qwen3.5-4B", "result.json")
    if not (os.path.exists(a) and os.path.exists(b)):
        return
    big = {p["user"]: p for p in json.load(open(a, encoding="utf-8"))["per_user"]}
    small = {p["user"]: p for p in json.load(open(b, encoding="utf-8"))["per_user"]}
    d = boot([small[u]["E"] - big[u]["A"] for u in small if u in big])
    if d[1] > -0.02:
        v = "PASS (matches): the whole interval lies above -0.02."
    elif d[2] < -0.02:
        v = "STOP: the whole interval lies below -0.02."
    else:
        v = "IN BETWEEN."
    L = ["# Real-data test 2: the 4B model with split jobs against the 27B model reading its history", "",
         f"4B-E minus 27B-A, mean per-person AUC difference: {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]", "", f"- {v}"]
    with open(os.path.join(G.ROOT, "results", "taste_comparison.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("\n".join(L))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--base-url", default=None)
    p.add_argument("--workers", type=int, default=128)
    p.add_argument("--batch", type=int, default=0, help="prompts per /v1/completions request (0: one chat call each)")
    p.add_argument("--users", type=int, default=N_USERS, help="fewer people, for checking only")
    p.add_argument("--check", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--refs", action="store_true", help="compute and save the references only")
    p.add_argument("--no-e", action="store_true", help="skip agent E (report A and W only)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def log(s):
        print(s, flush=True)
    movies, ratings, sha = load()
    users, n_elig = pick_users(ratings, args.users)
    log(f"MovieLens 1M: {len(ratings):,} ratings, sha256 {sha[:12]}... ({'matches' if sha == SHA else 'DIFFERS from'} "
        f"the plan); {n_elig} eligible people, {len(users)} chosen")
    ref_path = os.path.join(G.ROOT, "results", "taste_references", "result.json")
    if args.users == N_USERS and os.path.exists(ref_path):
        refs = json.load(open(ref_path, encoding="utf-8"))
        assert refs["users"] == [u["user"] for u in users]
    else:
        refs = references(movies, ratings, users, log)
        refs["users"] = [u["user"] for u in users]
        if args.users == N_USERS:
            os.makedirs(os.path.dirname(ref_path), exist_ok=True)
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(refs, f)
    log("references, mean AUC: " + ", ".join(f"{k} {np.mean(refs[k]):.3f}" for k in ("genres", "popularity", "cf")))
    if args.refs:
        return
    if args.stub:
        llm, tag = Stub(movies), "stub"
    elif args.base_url:
        llm, tag = G.Endpoint(args.base_url, args.model, args.workers), args.model.split("/")[-1]
        log(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}")
    else:
        raise SystemExit("needs --base-url, --stub or --refs")
    run_model(llm, tag, movies, users, refs, args, log)


if __name__ == "__main__":
    main()
