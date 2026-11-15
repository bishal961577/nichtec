"""Architect's round-2 CPU check (no model): does code binding AMPLIFY a describer's errors when the error falls on a
definition message ("my brother, Gaia Okafor"), rather than on a plain fact?

Uses the engineer's resolver_sim.py (copied here unchanged) on the dev world. Perfect local descriptions (error 0),
then exactly ONE role definition is misread: its role noun is replaced by another role noun used in the world.
All 12 x 11 single substitutions are tried. Also: one role definition dropped (12 cases).
Counts facts filed under the wrong person and facts left unfiled, against the perfect run.
Writes nothing in the repository.
"""
import random
import sys
from collections import Counter

sys.path.insert(0, "/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/architect")
sys.path.insert(0, "/home/user/nichtec/brain")
import resolver_sim as R  # noqa: E402
import messy_world as MW  # noqa: E402


def counts(W, P, bind):
    full_of = {p["full"].lower(): p["id"] for p in W.people}
    right = wrong = unfiled = 0
    wrong_by = Counter()
    for i, p in P.items():
        for c in p["claims"]:
            if c[0] != "fact" or c[1] is None or c[3] is None:
                continue
            e = bind.get((i, c[1]))
            st = p["styles"][c[1]] if isinstance(c[1], int) and c[1] < len(p["styles"]) else "?"
            if e is None:
                unfiled += 1
            elif full_of.get(e) == c[4]:
                right += 1
            else:
                wrong += 1
                wrong_by[st] += 1
    return right, wrong, unfiled, wrong_by


def main():
    W = MW.World("dev")
    P0 = R.local_parse(W, 0.0, random.Random(7))
    r0, w0, u0, _ = counts(W, P0, R.resolve(P0))
    total = r0 + w0 + u0
    print(f"perfect descriptions: {total} facts; right {r0}, wrong {w0}, unfiled {u0}")
    defs = [(i, k) for i, p in P0.items() for k, c in enumerate(p["claims"]) if c[0] == "role"]
    nouns = sorted({P0[i]["claims"][k][2] for i, k in defs})
    print(f"role definitions: {len(defs)}; nouns: {nouns}")
    rows = []
    for i, k in defs:
        c = P0[i]["claims"][k]
        for alt in nouns:
            if alt == c[2]:
                continue
            P = {j: dict(p) for j, p in P0.items()}
            P[i] = dict(P0[i])
            P[i]["claims"] = list(P0[i]["claims"])
            P[i]["claims"][k] = (c[0], c[1], alt, c[3])
            r, w, u, wb = counts(W, P, R.resolve(P))
            rows.append((c[2], alt, w - w0, u - u0))
    ws = sorted(x[2] for x in rows)
    us = sorted(x[3] for x in rows)
    n = len(rows)
    print(f"\nONE role definition misread (noun substituted), {n} cases:")
    print(f"  extra facts filed under the WRONG person: mean {sum(ws)/n:.1f}, median {ws[n//2]}, max {ws[-1]}; "
          f"cases with >=1 wrong fact: {sum(1 for x in ws if x > 0)}/{n}")
    print(f"  extra facts left UNFILED: mean {sum(us)/n:.1f}, median {us[n//2]}, max {us[-1]}")
    worst = sorted(rows, key=lambda x: -x[2])[:5]
    print("  worst cases (true noun -> misread noun: wrong, unfiled):", worst)
    drops = []
    for i, k in defs:
        P = {j: p for j, p in P0.items()}
        P[i] = dict(P0[i])
        P[i]["claims"] = [c for kk, c in enumerate(P0[i]["claims"]) if kk != k]
        r, w, u, _ = counts(W, P, R.resolve(P))
        drops.append((P0[i]["claims"][k][2], w - w0, u - u0))
    print(f"\nONE role definition dropped, {len(drops)} cases: extra wrong mean "
          f"{sum(d[1] for d in drops)/len(drops):.1f} (max {max(d[1] for d in drops)}), extra unfiled mean "
          f"{sum(d[2] for d in drops)/len(drops):.1f} (max {max(d[2] for d in drops)})")
    # comparison: one misread PLAIN fact attachment costs at most one fact
    print("\nfor comparison: one misread plain fact misfiles at most 1 fact.")
    # a rate view: definitions misread at rate q, everything else perfect (Monte Carlo, 200 draws per q)
    for q in (0.05, 0.10, 0.20):
        rng = random.Random(11)
        wr = []
        for _ in range(200):
            P = {j: p for j, p in P0.items()}
            for i, k in defs:
                if rng.random() < q:
                    c = P0[i]["claims"][k]
                    alt = rng.choice([x for x in nouns if x != c[2]])
                    P[i] = dict(P[i])
                    P[i]["claims"] = list(P[i]["claims"])
                    P[i]["claims"][k] = (c[0], c[1], alt, c[3])
            r, w, u, _ = counts(W, P, R.resolve(P))
            wr.append(100 * w / max(1, r + w))
        wr.sort()
        print(f"role definitions misread at rate {q:.2f} (all else perfect): wrong among filed "
              f"mean {sum(wr)/len(wr):.2f}%, 90th percentile {wr[int(0.9*len(wr))]:.2f}%, max {wr[-1]:.2f}%")


if __name__ == "__main__":
    main()
