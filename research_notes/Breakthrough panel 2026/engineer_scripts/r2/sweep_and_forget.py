"""Round 2 follow-up: (1) the describer error rates the 1% wrong-filing line needs; (2) 'forget X' by candidate set.
Strict mode leaves ambiguous first names unfiled; forgetting by binding then misses them. Rule tested here: forget X
removes every message with a mention bound to X OR an unfiled bare-name mention whose candidates include X."""
import random, sys
from collections import defaultdict
sys.path.insert(0, ".")
import adversarial_sim as A
R, MW = A.R, A.MW

print("== sweep on the original world (anchor errors at the hard rate) ==")
W0 = MW.World("dev")
for easy in (0.02, 0.05):
    for hardr in (0.02, 0.05, 0.10, 0.20):
        A.run(f"easy {easy:.0%} hard {hardr:.0%}", W0, easy, hardr, False)

print("\n== forget by candidate set, loose world (amb 0.3, early 0.5), perfect describer, strict ==")
W = A.LooseWorld("dev", 0.3, 0.0, 0.5)
P = A.local_parse2(W, 0.0, 0.0, random.Random(7))
bind = A.resolve2(P, strict=True)
full_of = {p["full"].lower(): p["id"] for p in W.people}
pid_full = {p["id"]: p["full"].lower() for p in W.people}
# candidates for unfiled bare names: everyone with that first name (or nickname) introduced by that date
first_seen = {}
for i in sorted(P, key=lambda i: P[i]["order"]):
    for d in P[i]["descs"]:
        for dd in ([d, d[1]] if d[0] == "via" else [d]):
            if dd[0] == "full":
                first_seen.setdefault(dd[1], P[i]["date"])
nick_of = defaultdict(set)
for n in W.notes:
    for pid, kind, v in n["facts"]:
        if kind == "nickname":
            nick_of[v.lower()].add(pid_full[pid])
cands = {}
for i, p in P.items():
    for k, d in enumerate(p["descs"]):
        if d[0] == "name" and not bind.get((i, k)):
            c = {e for e, dt in first_seen.items() if e.split()[0] == d[1] and dt <= p["date"]} | nick_of.get(d[1], set())
            cands[(i, k)] = c
firsts = defaultdict(int)
for p in W.people:
    firsts[p["first"]] += 1
targets = [p["id"] for p in W.people if p["close"]] + [p["id"] for p in W.people if firsts[p["first"]] > 1][:10]
for rule in ("binding only", "binding + candidates"):
    res, over = [], 0
    for pid in targets:
        rem = {i for (i, k), e in bind.items() if e and full_of.get(e) == pid}
        if rule != "binding only":
            rem |= {i for (i, k), c in cands.items() if pid_full[pid] in c}
        truth = {n["id"] for n in W.notes if any(m[1] == pid for m in n["mentions"])}
        res.append(len(rem & truth) / max(1, len(truth)))
        over += len(rem - truth)
    print(f"  {rule:22s} forget average {100*sum(res)/len(res):5.1f}%  worst {100*min(res):5.1f}%  "
          f"messages removed that did not mention the target (all 20 targets): {over}")
print(f"  unfiled bare-name mentions: {len(cands)}; with >1 candidate: {sum(1 for c in cands.values() if len(c) > 1)}")
