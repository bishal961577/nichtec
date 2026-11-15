import random, sys
from collections import Counter, defaultdict
sys.path.insert(0, ".")
import adversarial_sim as A
R, MW = A.R, A.MW
for label, W in (("original world", MW.World("dev")), ("loose world", A.LooseWorld("dev", 0.3, 0.0, 0.5))):
    P = A.local_parse2(W, 0.0, 0.0, random.Random(7))
    bind = A.resolve2(P, strict=False)
    full_of = {p["full"].lower(): p["id"] for p in W.people}
    firsts = defaultdict(int)
    for p in W.people: firsts[p["first"]] += 1
    targets = [p["id"] for p in W.people if p["close"]] + [p["id"] for p in W.people if firsts[p["first"]] > 1][:10]
    miss = Counter(); worst = None
    for pid in targets:
        rem = {i for (i, k), e in bind.items() if e and full_of.get(e) == pid}
        truth = {n["id"] for n in W.notes if any(m[1] == pid for m in n["mentions"])}
        frac = len(rem & truth) / len(truth)
        if worst is None or frac < worst[0]: worst = (frac, pid, truth - rem)
        for i in truth - rem:
            n = W.notes[i]
            miss.update(st for _, q, st in n["mentions"] if q == pid)
    frac, pid, missed = worst
    print(label, "| missed mention styles over all targets:", dict(miss))
    print("  worst target:", W.people[pid]["full"], "close" if W.people[pid]["close"] else "", f"{100*frac:.1f}%",
          "of", len(missed) + int(0), "missed; examples:")
    for i in sorted(missed)[:4]:
        print("   ", W.notes[i]["text"][:90], [m[2] for m in W.notes[i]["mentions"] if m[1] == pid])
