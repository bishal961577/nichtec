"""Learning routes by driving (reports/Learning routes by driving 2026.md): does a memory that learns its routes from
being used and corrected file new facts better than the same memory with fixed routes and spot fixes?

Free, on a CPU, with no model calls: it reuses the cached descriptions of the generated chat worlds (two ends check,
steps 2, 4 and 5) and the resolver as frozen after step 4 (antecedent rule fixed, reporting-verb filter on).

The track: the resolver has worked out who every mention in a message is. A fact claim is filed under the person of the
mention the describer says it is about (the default route), or, after learning, under another person named in the same
message. Every pair (the mention the fact was said to be about, another mention) has a route type built from what the
system can see: both mentions' kinds (pronoun, full name, first name, role, "X's flatmate", group), whether each is the
teller, whether the second is the pronoun's antecedent, whether the message reports speech ("told me", "says"), and
which comes first. A route type is a switch: each correction that says a fact belonged to the other mention adds STEP
to it, each correction that says a switched fact belonged where it was before takes STEP away, and the switch is thrown
when its weight passes MARGIN (two net corrections). No truth is used to route; truth is used only to play the owner,
who corrects facts that are shown to them and wrong.

Arms (the same days, the same facts shown each day, common random numbers):
  fixed   no corrections at all (the step 5 system as it is);
  patch   each correction moves that one fact (a fixed map with errata);
  drive   each correction moves that fact and teaches the switches; each night every earlier fact is re-routed.

    python brain/drive_routes.py --selfcheck     # bookkeeping checks
    python brain/drive_routes.py --stage all     # every dataset, as registered
"""
import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402
import confirm_defs as CD  # noqa: E402

CFG = dict(shown=3, cap=2, step=0.6, margin=1.0, seeds=10, boot=2000, sweep=(1, 10))
OUT = os.path.join(MI.ROOT, "results", "drive_routes")
REPORT_TXT = re.compile(r"\b(told me|tells me|telling me|says|said|heard|mentioned|according to)\b", re.I)
ARMS = ("fixed", "patch", "drive")
FRESH_SEED = 20261117


def fresh_world():
    W = MW.World("dev", seed=FRESH_SEED)
    keep = {n["id"] for n in W.notes[:TE.DEV_LIMIT]}
    W.notes = W.notes[:TE.DEV_LIMIT]
    W.deleted = [d for d in W.deleted if d[2] in keep]
    W.deleted_ids = {d[2] for d in W.deleted}
    W.questions = [q for q in W.questions if all(n in keep for n in q["notes"]) and q["notes"]]
    return W


DATASETS = {   # name: (world, description files, describer)
    "4B dev": (TE.dev_world, None, "Qwen3.5-4B"),
    "27B dev": (TE.dev_world, ("describe_all_27b.jsonl",), "Qwen3.5-27B-FP8"),
    "27B fresh": (fresh_world, ("describe_fresh_27b.jsonl",), "Qwen3.5-27B-FP8"),
}


class Track:
    """One world, one set of descriptions, one definition condition: the resolver's bindings and every fact claim."""

    def __init__(self, W, descs, confirmed):
        self.W = W
        ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
        self.by = {n["id"]: n for n in W.notes}
        DB.ANTE_V2 = True
        if confirmed:
            conf = {i: CD.truth_defs(W, self.by[i]) for i in ids if CD.truth_defs(W, self.by[i])}
            self.P = DB.to_parse(W, descs, ids, keep_defs=False, confirmed=conf, drop_reporting=True)
        else:
            self.P = DB.to_parse(W, descs, ids, drop_reporting=True)
        self.bind = DB.resolve(self.P)
        V = DB.View(W, self.P, self.bind)
        DB.ANTE_V2 = False
        self.V0 = V
        wq = MI.write_quality(W, V.ix)
        self.who = wq.pop("who")
        self.reg0, wrong0, self.reg0_n = CD.everyday_merges(W, V.ix, self.who)
        self.reg0_wrong = len(wrong0)
        self.owners = {n["id"]: {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS
                                                                                else (pid,))} for n in W.notes}
        self.me = {pid for pid, p in V.ix.people.items() if p["full"].lower().strip(" .") in MI.ME_NAMES}
        fact_notes = {n["id"] for n in W.notes if n["facts"]}
        self.claims = []                      # every everyday fact claim, in date order
        for i in sorted(self.P, key=lambda x: self.P[x]["order"]):
            p, ent = self.P[i], V.ent_of[i]
            for ci, c in enumerate(p["claims"]):
                if c[0] != "fact" or i not in fact_notes or CD.is_def(c[2] or "note"):
                    continue
                a = c[1]
                pid = V.pid(ent[a])
                if pid in self.me or pid not in self.who:
                    continue                  # the same rows test 2's everyday count keeps
                cands = [k for k in range(len(ent)) if k != a and V.pid(ent[k]) not in self.me]
                self.claims.append({"i": i, "ci": ci, "a": a, "label": c[2] or "note", "value": c[3],
                                    "date": p["date"], "cands": cands,
                                    "keys": {k: self.route_key(i, a, k) for k in cands}})
        self.days = sorted({c["date"] for c in self.claims})

    def kind(self, i, k):
        return self.P[i]["descs"][k][0] + ("*" if self.P[i]["styles"][k] == "source" else "")

    def route_key(self, i, a, k):
        """What the system can see about sending a fact said to be about mention a to mention k instead."""
        return (self.kind(i, a), self.kind(i, k), "ante" if self.P[i]["ante"].get(a) == ("here", k) else "-",
                "rep" if REPORT_TXT.search(self.by[i]["text"]) else "-", "before" if k < a else "after")

    def person(self, c, k):
        """The true person (or None) of the index person a claim is filed under when routed to mention k."""
        return self.who.get(self.V0.pid(self.V0.ent_of[c["i"]][k]))

    def right(self, c, k):
        return self.person(c, k) in self.owners[c["i"]]

    def owner_of(self, c):
        """Who the owner says the fact is about: the true person whose fact in the message matches the claim's value;
        or the message's only fact owner; or None (the owner cannot say)."""
        n = self.by[c["i"]]
        hits = {pid for pid, s, v in n["facts"] if s in MW.SLOTS and v is not None and
                any(MI.has_key(c["value"], key) for key in MW.value_keys(s, v))}
        if len(hits) == 1:
            return next(iter(hits))
        own = self.owners[c["i"]] & {p["id"] for p in self.W.people}
        return next(iter(own)) if len(own) == 1 else None


def route(c, w, margin):
    """The mention a claim is filed under: the default, unless a switch's weight passes the margin."""
    best, bw = c["a"], margin
    for k in c["cands"]:
        if w.get(c["keys"][k], 0.0) > bw:
            best, bw = k, w[c["keys"][k]]
    return best


def simulate(T, arm, seed, shown, cfg=CFG):
    """Day by day: facts arrive and are routed; `shown` facts filed so far are shown to the owner, who corrects each
    wrong one (at most cap a day); at night the drive arm re-routes everything not corrected."""
    w, pinned, cur = defaultdict(float), {}, {}
    shown_ever, corrections, unknown = set(), 0, 0
    by_day = defaultdict(list)
    for idx, c in enumerate(T.claims):
        by_day[c["date"]].append(idx)
    seen = []
    for day in T.days:
        for idx in by_day[day]:                                   # arrivals
            cur[idx] = route(T.claims[idx], w, cfg["margin"]) if arm == "drive" else T.claims[idx]["a"]
            seen.append(idx)
        rng = random.Random(f"shown-{seed}-{day}")              # the same facts shown in every arm
        today = rng.sample(seen, min(shown, len(seen)))
        fixed_today = 0
        for idx in today:
            shown_ever.add(idx)
            c = T.claims[idx]
            if arm == "fixed" or idx in pinned or T.right(c, cur[idx]) or fixed_today >= cfg["cap"]:
                continue
            q = T.owner_of(c)
            if q is None:
                unknown += 1
                continue
            fixed_today += 1
            corrections += 1
            pinned[idx] = q
            if arm == "drive":
                good = [k for k in [c["a"]] + c["cands"] if T.person(c, k) == q]
                if cur[idx] != c["a"]:
                    w[c["keys"][cur[idx]]] -= cfg["step"]         # the switch sent it the wrong way
                if good and good[0] != c["a"]:
                    w[c["keys"][good[0]]] += cfg["step"]          # it should have gone to that mention
        if arm == "drive":                                        # night: re-route everything not corrected
            for idx in seen:
                if idx not in pinned:
                    cur[idx] = route(T.claims[idx], w, cfg["margin"])
    out = [{"ok": idx in pinned or T.right(c, cur[idx]), "shown": idx in shown_ever, "i": c["i"],
            "moved": cur[idx] != c["a"] and idx not in pinned, "was_right": T.right(c, c["a"])}
           for idx, c in enumerate(T.claims)]
    thrown = sorted(" | ".join(k) for k, v in w.items() if v > cfg["margin"])
    return {"claims": out, "corrections": corrections, "owner_could_not_say": unknown, "switches_thrown": thrown,
            "weights": {" | ".join(k): round(v, 2) for k, v in sorted(w.items()) if v}, "cur": cur, "pinned": pinned}


def oracle_wrong(T):
    """Claims no route in their message could file right: the floor for any router."""
    return sum(not (T.right(c, c["a"]) or any(T.right(c, k) for k in c["cands"])) for c in T.claims)


def registered_final(T, sim):
    """Test 2's everyday wrong-merge count on an arm's final index: routes applied, and each corrected fact filed under
    the person the owner named (the mention naming them, or that person's full name if none does)."""
    P2 = {i: dict(p, claims=list(p["claims"])) for i, p in T.P.items()}
    extra, drop = [], set()
    for idx, c in enumerate(T.claims):
        k = sim["cur"][idx]
        if idx in sim["pinned"]:
            q = sim["pinned"][idx]
            hit = [kk for kk in [c["a"]] + c["cands"] if T.person(c, kk) == q]
            if not hit:
                drop.add((c["i"], c["ci"]))
                extra.append((c["i"], T.W.people[q]["full"].lower(), c["label"], c["value"]))
                continue
            k = hit[0]
        old = P2[c["i"]]["claims"][c["ci"]]
        P2[c["i"]]["claims"][c["ci"]] = ("fact", k, old[2], old[3])
    for i in P2:
        P2[i]["claims"] = [x for ci, x in enumerate(P2[i]["claims"]) if (i, ci) not in drop]
    V = DB.View(T.W, P2, T.bind)
    for i, e, lab, val in extra:
        V.ix.add(i, V.pid(e), lab, val)
    wq = MI.write_quality(T.W, V.ix)
    who = wq.pop("who")
    ev, wrong, n = CD.everyday_merges(T.W, V.ix, who)
    return ev, len(wrong), n


def boot_ci(per_msg_a, per_msg_b, B, seed=7):
    """95% interval of rate(a) - rate(b) over messages resampled with replacement; per_msg_x = {message: [wrong, n]}."""
    msgs = sorted(set(per_msg_a) | set(per_msg_b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(B):
        s = [msgs[rng.randrange(len(msgs))] for _ in msgs]
        wa = sum(per_msg_a.get(m, [0, 0])[0] for m in s)
        na = sum(per_msg_a.get(m, [0, 0])[1] for m in s)
        wb = sum(per_msg_b.get(m, [0, 0])[0] for m in s)
        nb = sum(per_msg_b.get(m, [0, 0])[1] for m in s)
        diffs.append(100 * (wa / max(1, na) - wb / max(1, nb)))
    diffs.sort()
    return diffs[int(0.025 * B)], diffs[int(0.975 * B) - 1]


def verdict(R, ci, patch_wrong_per_seed):
    if patch_wrong_per_seed < 10:
        return "INCONCLUSIVE"
    if R >= 30 and ci[0] > 0:
        return "LEARNS"
    if R < 10 or ci[0] <= 0:
        return "NO GAIN"
    return "BETWEEN"


def run_condition(T, shown, cfg=CFG, final=True):
    """All arms over every seed. Primary: wrong share among facts never shown to the owner (nothing corrected them
    directly), drive against patch."""
    res, per_msg = {}, {}
    for arm in ARMS:
        agg = defaultdict(float)
        pm = defaultdict(lambda: [0, 0])
        thrown = Counter()
        for seed in range(cfg["seeds"]):
            sim = simulate(T, arm, seed, shown, cfg)
            cl = sim["claims"]
            ns = [x for x in cl if not x["shown"]]
            agg["claims"] += len(cl)
            agg["wrong_all"] += sum(not x["ok"] for x in cl)
            agg["never_shown"] += len(ns)
            agg["wrong_never_shown"] += sum(not x["ok"] for x in ns)
            agg["corrections"] += sim["corrections"]
            agg["owner_could_not_say"] += sim["owner_could_not_say"]
            agg["moved"] += sum(x["moved"] for x in cl)
            agg["moved_right_to_wrong"] += sum(x["moved"] and x["was_right"] and not x["ok"] for x in cl)
            agg["moved_wrong_to_right"] += sum(x["moved"] and not x["was_right"] and x["ok"] for x in cl)
            for x in ns:
                pm[x["i"]][0] += not x["ok"]
                pm[x["i"]][1] += 1
            thrown.update(sim["switches_thrown"])
            if final and seed == 0:
                ev, nw, n = registered_final(T, sim)
                agg["registered_final_pct"], agg["registered_final_wrong"], agg["registered_final_rows"] = ev, nw, n
        S = cfg["seeds"]
        r = {k: (v / S if k not in ("registered_final_pct", "registered_final_wrong", "registered_final_rows") else v)
             for k, v in agg.items()}
        r["wrong_all_pct"] = 100 * agg["wrong_all"] / agg["claims"]
        r["wrong_never_shown_pct"] = 100 * agg["wrong_never_shown"] / max(1, agg["never_shown"])
        r["switches_thrown_in_seeds"] = dict(thrown.most_common())
        res[arm], per_msg[arm] = r, pm
    p, d = res["patch"], res["drive"]
    R = 100 * (1 - d["wrong_never_shown"] / p["wrong_never_shown"]) if p["wrong_never_shown"] else float("nan")
    ci = boot_ci(per_msg["patch"], per_msg["drive"], cfg["boot"])
    res["primary"] = {"relative_reduction_pct": R, "diff_pct_points": p["wrong_never_shown_pct"] - d["wrong_never_shown_pct"],
                      "ci95_pct_points": ci, "patch_wrong_never_shown_per_seed": p["wrong_never_shown"],
                      "verdict": verdict(R, ci, p["wrong_never_shown"])}
    return res


def selfcheck(T, log):
    """(1) the fixed arm's claim count and wrong count equal test 2's everyday count on the same index; (2) the fixed
    arm's final index reproduces it; (3) drive with a step of 0 equals patch; (4) routing uses no truth."""
    sim = simulate(T, "fixed", 0, CFG["shown"])
    wrong = sum(not x["ok"] for x in sim["claims"])
    assert (wrong, len(T.claims)) == (T.reg0_wrong, T.reg0_n), (wrong, len(T.claims), T.reg0_wrong, T.reg0_n)
    ev, nw, n = registered_final(T, sim)
    assert (nw, n) == (T.reg0_wrong, T.reg0_n), (nw, n, T.reg0_wrong, T.reg0_n)
    a = simulate(T, "patch", 3, CFG["shown"])
    b = simulate(T, "drive", 3, CFG["shown"], dict(CFG, step=0.0))
    assert [x["ok"] for x in a["claims"]] == [x["ok"] for x in b["claims"]]
    assert all(isinstance(x, str) for c in T.claims for key in c["keys"].values() for x in key)
    log(f"   self-check passed: {len(T.claims)} claims, {wrong} filed wrong by the fixed routes "
        f"({100 * wrong / len(T.claims):.2f}%, test 2's everyday count {T.reg0:.2f}%), routing floor "
        f"{oracle_wrong(T)} wrong")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "selfcheck"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} learning routes by driving, stage {a.stage}, {json.dumps(CFG)}")
    results = {"cfg": CFG}
    for name, (world, files, model) in DATASETS.items():
        W = world()
        descs = CD.load_descs(None if files is None else tuple(os.path.join(TE.OUT, f) for f in files))
        assert all(n["id"] in descs for n in W.notes), name
        for confirmed in (False, True):
            label = f"{name}, {'every definition confirmed' if confirmed else 'own definitions'}"
            t0 = time.time()
            T = Track(W, descs, confirmed)
            log(f"{label} ({model}): {len(T.claims)} everyday fact claims over {len(T.days)} days")
            selfcheck(T, log)
            if a.stage == "selfcheck":
                continue
            r = run_condition(T, CFG["shown"])
            r["sweep"] = {}
            for s in CFG["sweep"]:
                rs = run_condition(T, s, final=False)
                r["sweep"][s] = {"primary": rs["primary"], "corrections_drive": rs["drive"]["corrections"],
                                 "corrections_patch": rs["patch"]["corrections"]}
            r.update(claims=len(T.claims), days=len(T.days), routing_floor_wrong=oracle_wrong(T),
                     registered_start_pct=T.reg0, model=model)
            results[label] = r
            pr = r["primary"]
            log(f"   fixed {r['fixed']['wrong_all_pct']:.2f}% wrong | patch {r['patch']['wrong_all_pct']:.2f}% "
                f"({r['patch']['corrections']:.1f} corrections) | drive {r['drive']['wrong_all_pct']:.2f}% "
                f"({r['drive']['corrections']:.1f} corrections; moved {r['drive']['moved']:.1f}: "
                f"{r['drive']['moved_wrong_to_right']:.1f} wrong->right, {r['drive']['moved_right_to_wrong']:.1f} "
                f"right->wrong)")
            log(f"   never shown: patch {r['patch']['wrong_never_shown_pct']:.2f}% "
                f"({r['patch']['wrong_never_shown']:.1f} of {r['patch']['never_shown']:.0f}) | drive "
                f"{r['drive']['wrong_never_shown_pct']:.2f}% | reduction {pr['relative_reduction_pct']:.1f}% "
                f"(CI of the difference {pr['ci95_pct_points'][0]:.2f} to {pr['ci95_pct_points'][1]:.2f} points) "
                f"-> {pr['verdict']}  [{time.time() - t0:.0f} s]")
            for s, x in r["sweep"].items():
                log(f"   sweep, {s} shown a day: reduction {x['primary']['relative_reduction_pct']:.1f}% "
                    f"(CI {x['primary']['ci95_pct_points'][0]:.2f} to {x['primary']['ci95_pct_points'][1]:.2f}), "
                    f"{x['corrections_drive']:.1f} corrections -> {x['primary']['verdict']}")
            log(f"   switches thrown (seeds of {CFG['seeds']}): {r['drive']['switches_thrown_in_seeds']}")
    if a.stage == "all":
        with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
