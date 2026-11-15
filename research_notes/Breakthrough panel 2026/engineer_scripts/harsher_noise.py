"""Harsher describer noise: besides the errors in resolver_sim, the describer drops the surname of a full name
(writes "Mira" for "Mira Okafor") with probability e, so the resolver must fall back on first-name rules and recency."""
import random, sys
sys.path.insert(0, "."); sys.path.insert(0, "/home/user/nichtec/brain")
import resolver_sim as R, messy_world as MW
W = MW.World("dev")
for e in (0.05, 0.10, 0.20):
    P = R.local_parse(W, e, random.Random(7))
    rng = random.Random(3)
    for p in P.values():
        p["descs"] = [("name", d[1].split()[0]) if d[0] == "full" and rng.random() < e else d for d in p["descs"]]
    ev = R.evaluate(W, P, R.resolve(P))
    print(f"e={e:.2f}: filed right {ev['filed_right_pct_of_all']}%, wrong among filed {ev['wrong_pct_of_filed']}%,",
          "by style", ev["right_of_filed_by_style"])
