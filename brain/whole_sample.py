"""Test 1, amendment 5: the 4B whole-notebook arm on a fixed random sample (cost), plus amendment 4's thinking-off arms.

Why: the first 29 answers of ("personal", "think", 5000) on Qwen3.5-4B used a median of 10,583 completion tokens, so all
995 questions would take about 6-7 L40S hours. The sample is fixed here, before any further answer of that arm:
20 questions per kind (all 20 of "prev"), drawn with random.Random(SEED) from each kind's question ids in sorted order.
Answers already saved for sampled questions are kept; saved answers outside the sample are reported separately.

    python brain/whole_sample.py --base-url URL/v1      # runs the 256-token thinking-off arms, then the sample
    python brain/whole_sample.py --report               # amendment 4's line, applied to the sample
"""
import argparse
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import notebook_read as N   # noqa: E402

SEED, PER_KIND = 20261114, 20
OUT = os.path.join(N.ROOT, "results", "notebook_read_Qwen3.5-4B")


def sample(PQ):
    by = {}
    for q in PQ:
        if q["kind"] in N.kinds_at(5000):
            by.setdefault(q["kind"], []).append(q["qid"])
    rng = random.Random(SEED)
    return {k: sorted(rng.sample(sorted(v), min(PER_KIND, len(v)))) for k, v in sorted(by.items())}


def report(PQ):
    S = sample(PQ)
    ids = {i for v in S.values() for i in v}
    rows = [json.loads(l) for l in open(os.path.join(OUT, "answers.jsonl"), encoding="utf-8")]
    full = {r["qid"]: r for r in rows if r["arm"] == "personal" and r["mode"] == "think" and str(r["level"]) == "5000"}
    pages = {r["qid"]: r for r in rows if r["arm"] == "personal" and r["mode"] == "think" and str(r["level"]) == "10"}
    got = {i: full[i] for i in ids if i in full}
    counts = {k: sum(q["kind"] == k for q in PQ) for k in N.HARD}
    rng = np.random.default_rng(0)
    by = {k: np.array([got[i]["correct"] for i in S[k] if i in got], float) for k in S}
    missing = sum(len(S[k]) - len(by[k]) for k in S)
    if any(len(by[k]) == 0 for k in N.HARD):
        return f"Amendment 5 sample: not complete ({missing} of {sum(len(v) for v in S.values())} sampled answers missing)."

    def weighted(b):
        return 100 * sum(counts[k] * b[k].mean() for k in N.HARD) / sum(counts.values())
    est = weighted(by)
    boots = [weighted({k: rng.choice(by[k], len(by[k])) for k in N.HARD}) for _ in range(N.BOOT)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    hard_rows = [got[i] for k in N.HARD for i in S[k] if i in got]
    med = float(np.median([r["completion_tokens"] or 0 for r in hard_rows]))
    p90 = float(np.percentile([r["completion_tokens"] or 0 for r in hard_rows], 90))
    base = float(np.median([r["completion_tokens"] or 0 for r in pages.values() if r["kind"] in N.HARD]))
    cap = 100 * np.mean([r["truncated"] for r in hard_rows])
    kinds = {k: 100 * by[k].mean() for k in ("desc", "who", "two")}
    fast = med <= 2 * base and cap <= 2.0
    if est >= 90 and min(kinds.values()) >= 75 and fast:
        v = "PASS"
    elif est >= 90 and min(kinds.values()) >= 75:
        v = "READS BUT TOO SLOW"
    elif est <= 75:
        v = "FAIL"
    else:
        v = "BETWEEN"
    L = [f"Amendment 5 sample: {sum(len(v) for v in S.values())} questions ({PER_KIND} per kind, seed {SEED}); "
         f"answered {sum(len(b) for b in by.values())}, missing {missing}.",
         f"Line (amendment 4) on the sample: {v}. Hard set, weighted to the full set's kind counts: {est:.1f}% "
         f"[{lo:.1f}, {hi:.1f}] (pass >= 90, fail <= 75); described {kinds['desc']:.0f}%, who {kinds['who']:.0f}%, "
         f"two-hop {kinds['two']:.0f}% (each >= 75 to pass; n = 20 each); median completion {med:.0f} tokens vs "
         f"{2 * base:.0f} allowed, 90th percentile {p90:.0f}; cut off at 16K {cap:.1f}% (<= 2 allowed).",
         "By kind (sample): " + ", ".join(f"{k} {100 * b.mean():.0f}% (n={len(b)})" for k, b in by.items() if len(b))]
    extra = [r for i, r in full.items() if i not in ids]
    if extra:
        L.append(f"Answers outside the sample (saved before amendment 5, reported only): {len(extra)}, "
                 f"{100 * np.mean([r['correct'] for r in extra]):.1f}% right, kinds {sorted({r['kind'] for r in extra})}.")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    groups, P, PQ, PP, BMP, texts, WQ, WP, clash, bm = N.build()
    if a.report:
        print(report(PQ))
        return
    import order_gap as G
    D = {"P": P, "PQ": PQ, "PP": PP, "BMP": BMP, "WQ": WQ, "WP": WP, "BM": bm}
    llm = G.Endpoint(a.base_url, a.model, 1)
    path = os.path.join(OUT, "answers.jsonl")
    N.run(llm, [("personal", "nothink256", 10), ("personal", "nothink256", 5000)], D, path, a.workers, 0, False)
    ids = {i for v in sample(PQ).values() for i in v}
    Ds = {**D, "PQ": [q for q in PQ if q["qid"] in ids]}
    N.run(llm, [("personal", "think", 5000)], Ds, path, a.workers, 0, False)
    print(report(PQ), flush=True)


if __name__ == "__main__":
    main()
