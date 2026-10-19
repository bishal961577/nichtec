"""A shared starting point plus a personal difference. Pre-registered in section 9 of reports/Real data test 2026.md.

Uses only answers saved by brain/taste.py (no model calls): P is popularity before each person's test window;
P+E learns the person's difference from the 4B model's own questions plus standardised popularity; P+A and P+W
average popularity's rank with A's or W's margin rank (nothing fitted); P+genres is the no-model control.

  python brain/prior.py
"""
import bisect
import collections
import json
import os
import sys

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402
import taste as T  # noqa: E402

OUT = os.path.join(G.ROOT, "results", "prior_test")


def load_json(path):
    return json.load(open(os.path.join(G.ROOT, path), encoding="utf-8"))


def main():
    movies, ratings, _ = T.load()
    users, _ = T.pick_users(ratings)
    refs = load_json("results/taste_references/result.json")
    assert refs["users"] == [u["user"] for u in users]
    run4 = {p["user"]: p for p in load_json("results/taste_Qwen3.5-4B/result.json")["per_user"]}
    ans = {"4B": load_json("data/taste_cache_Qwen3.5-4B.json"), "27B": load_json("data/taste_cache_Qwen3.5-27B-FP8.json")}
    gen = {"4B": load_json("data/taste_gen_Qwen3.5-4B.json"), "27B": load_json("data/taste_gen_Qwen3.5-27B-FP8.json")}

    def cached(tag, prompt, n, seed, max_tokens, **kw):
        key = json.dumps([prompt, n, seed, max_tokens, sorted(kw.items())])
        return gen[tag][key]

    # popularity known at the start of each person's test window
    per = collections.defaultdict(list)
    for _, m, r, t in ratings:
        per[m].append((t, r))
    cum = {}
    for m, xs in per.items():
        xs.sort()
        cum[m] = ([t for t, _ in xs], np.cumsum([r for _, r in xs]).tolist())
    gm = float(np.mean([r for _, _, r, _ in ratings]))

    def pop(u, m):
        ts, cs = cum[m]
        k = bisect.bisect_left(ts, u["t0"])
        return ((cs[k - 1] if k else 0.0) + 5 * gm) / (k + 5)

    decades = [(0, 1950), (1950, 1960), (1960, 1970), (1970, 1980), (1980, 1990), (1990, 2000), (2000, 3000)]

    def gfeat(m):
        mv = movies[m]
        return [g in mv["genres"] for g in T.GENRES] + [lo <= mv["year"] < hi for lo, hi in decades]

    rows, mismatches = [], []
    for i, u in enumerate(users):
        yt = [y for _, y in u["test"]]
        ys = [y for _, y in u["stream"]]
        ms_s, ms_t = [m for m, _ in u["stream"]], [m for m, _ in u["test"]]
        pall = np.array([pop(u, m) for m in ms_s + ms_t])
        z = (pall - pall.mean()) / (pall.std() or 1.0)
        zs, zt = z[:len(ms_s)], z[len(ms_s):]
        ptest = pall[len(ms_s):]
        rec = {"user": u["user"], "P": T.auc(ptest, yt)}
        # E rebuilt from the saved 4B answers (sanity check), then with popularity added
        qs = T.parse_questions(cached("4B", T.e_prompt(movies, u), T.NAME_SAMPLES, 2000 + i, 300, **T.GEN))
        read = ans["4B"]
        Xs = [[read[T.read_prompt(movies, m, q)] > 0 for q in qs] for m in ms_s]
        Xt = [[read[T.read_prompt(movies, m, q)] > 0 for q in qs] for m in ms_t]
        sc, _, _ = T.fit_l1(Xs, ys, Xt)
        e = T.auc(sc, yt) if len(set(np.round(sc, 12))) > 1 else 0.5
        if abs(e - run4[u["user"]]["E"]) > 0.001:
            mismatches.append((u["user"], e, run4[u["user"]]["E"]))
        sc, coef, _ = T.fit_l1([x + [a] for x, a in zip(Xs, zs)], ys, [x + [a] for x, a in zip(Xt, zt)])
        rec["P+E"] = T.auc(sc, yt) if len(set(np.round(sc, 12))) > 1 else 0.5
        rec["P+E_pop_weight"], rec["P+E_personal_weights"] = coef[-1] if coef else 0.0, sum(abs(c) > 1e-9 for c in coef[:-1])
        sc, _, _ = T.fit_l1([gfeat(m) + [a] for m, a in zip(ms_s, zs)], ys, [gfeat(m) + [a] for m, a in zip(ms_t, zt)])
        rec["P+genres"] = T.auc(sc, yt) if len(set(np.round(sc, 12))) > 1 else 0.5
        rp = rankdata(ptest)
        for tag in ("4B", "27B"):
            a = [ans[tag][T.a_prompt(movies, u, m)] for m in ms_t]
            seed_w = 1000 + i
            d = cached(tag, T.w_prompt(movies, u), 1, seed_w, 200, temperature=0)[0].strip()
            w = [ans[tag][T.judge_prompt(movies, d, m)] for m in ms_t]
            rec[f"P+A {tag}"] = T.auc(rp + rankdata(a), yt)
            rec[f"P+W {tag}"] = T.auc(rp + rankdata(w), yt)
        rec["cf"] = refs["cf"][i]
        rows.append(rec)
    if mismatches:
        raise SystemExit(f"sanity check failed: E not reproduced for {len(mismatches)} people, e.g. {mismatches[:3]}")

    keys = ["P", "P+E", "P+genres", "P+A 4B", "P+A 27B", "P+W 4B", "P+W 27B", "cf"]
    S = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    D = {k: T.boot([r[k] - r["P"] for r in rows]) for k in keys if k != "P"}
    D["P+E vs P+genres"] = T.boot([r["P+E"] - r["P+genres"] for r in rows])

    def line(d):
        if d[0] >= 0.02 and d[1] > 0:
            return "PASS"
        if d[1] <= 0:
            return "STOP"
        return "IN BETWEEN"
    f3 = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    L = ["# A shared starting point plus a personal difference", "",
         "Pre-registration: section 9 of reports/Real data test 2026.md. 300 MovieLens-1M people, 80 ratings "
         "learned, next 40 tested; mean per-person AUC (0.5 = guessing). No model calls: saved answers only.", "",
         "Sanity check: E rebuilt from the saved 4B answers reproduces the 4B run for all 300 people (within 0.001).", "",
         "| Agent | Mean AUC | Minus P [95% interval] |", "|---|---|---|",
         f"| P, shared list (popularity) | {S['P']:.3f} | — |"]
    names = {"P+E": "P+E, personal questions learned on top of P (4B)", "P+genres": "P+genres, dataset tags on top of P (no model)",
             "P+A 4B": "P+A, P's rank + history-reading rank (4B)", "P+A 27B": "P+A, P's rank + history-reading rank (27B)",
             "P+W 4B": "P+W, P's rank + taste-in-words rank (4B)", "P+W 27B": "P+W, P's rank + taste-in-words rank (27B)",
             "cf": "Collaborative filtering, everyone's data (reference)"}
    for k in keys[1:]:
        L.append(f"| {names[k]} | {S[k]:.3f} | {f3(D[k])} |")
    L += ["", f"P+E minus P+genres: {f3(D['P+E vs P+genres'])}.",
          f"P+E kept popularity in its model for {sum(abs(r['P+E_pop_weight']) > 1e-9 for r in rows)} of 300 people; "
          f"mean personal questions with weight: {np.mean([r['P+E_personal_weights'] for r in rows]):.1f}.", "",
          "## Verdict", "",
          f"- Main (P+E against P): {line(D['P+E'])}: {f3(D['P+E'])}.",
          f"- Secondary, P+A (4B) against P: {line(D['P+A 4B'])}: {f3(D['P+A 4B'])}.",
          f"- Secondary, P+A (27B) against P: {line(D['P+A 27B'])}: {f3(D['P+A 27B'])}."]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": S, "differences": D, "per_user": rows}, f, indent=1)
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
