"""Engineer's CPU check (no model): can identity be decided by CODE from per-message LOCAL descriptions?

The design under test ("describe, don't decide"): the phone model never sees the memory while writing. For each
message (plus the one before it, for pronouns) it emits a local description: the mentions as written ("Kofi's
flatmate", "my old boss", "Mira from work", "she" -> the person of the previous message), and the claims with a
pointer to the mention they are about. A deterministic resolver then decides who every mention is, as a function of
the whole SET of descriptions, re-run whenever the set changes (a new message, a deletion).

Here the local descriptions are simulated from the generator's truth (surface strings only; the resolver never sees
person ids), with injected parser errors. So this measures the resolver and the decomposition, NOT the 4B.

Run: python resolver_sim.py   (reads brain/messy_world.py from the repo, writes nothing there)
"""
import random
import sys
import time
from collections import Counter, defaultdict

sys.path.insert(0, "/home/user/nichtec/brain")
import messy_world as MW  # noqa: E402

PRON = {"he", "she", "they", "him", "her", "them"}
CIRCLE = {"work": "work", "climbing": "climbing", "choir": "choir", "street": "street", "spanish": "spanish",
          "football": "football"}
FAR = "9999-12-31"


def classify(s):
    """What a parser can tell from the mention's own words."""
    s = s.strip()
    low = s.lower()
    if low in PRON:
        return ("pron",)
    if "'s " in s:
        anchor, noun = s.rsplit("'s ", 1)
        return ("via", classify(anchor), noun.lower())
    if low.startswith("my old "):
        return ("role", low[7:], "old")
    if low.startswith("my "):
        return ("role", low[3:], "cur")
    if " from " in s:
        first, ctx = s.split(" from ", 1)
        key = next((v for k, v in CIRCLE.items() if k in ctx.lower()), ctx.lower())
        return ("fctx", first.lower(), key)
    if len(s.split()) >= 2:
        return ("full", low)
    return ("name", low)


def local_parse(W, err, rng, defer_frac=0.0):
    """One description per message, from that message (and its predecessor) alone. Returns {note id: parse}.
    defer_frac: share of nickname and relation messages moved to the last day (a use-before-definition world)."""
    notes = W.notes
    out = {}
    prev_ments = []
    rels = list(MW.RELS)
    for i, n in enumerate(notes):
        ms = [(s, pid, st) for s, pid, st in n["mentions"] if st != "via_anchor"]
        descs, truth_pid, styles = [], [], []
        for s, pid, st in ms:
            d = classify(s)
            descs.append(d)
            truth_pid.append(pid)
            styles.append(st)
        ante = {}
        for k, d in enumerate(descs):           # pronoun -> a mention in this message, else in the previous one
            if d[0] != "pron":
                continue
            here = [j for j, dd in enumerate(descs) if dd[0] != "pron" and styles[j] != "source"]
            if here:
                ante[k] = ("here", here[0])
            elif prev_ments:
                tgt = prev_ments[0]
                if rng.random() < err and len(prev_ments) > 1:
                    tgt = prev_ments[1]          # parser error: wrong antecedent
                ante[k] = ("prev", tgt)
        claims = []

        def subj(pid):
            cand = [j for j, p in enumerate(truth_pid) if p == pid]
            cand.sort(key=lambda j: descs[j][0] == "pron")
            if not cand:
                return None
            j = cand[0]
            others = [x for x in range(len(descs)) if truth_pid[x] != pid]
            if others and rng.random() < err:    # parser error: claim attached to the wrong mention (anchor, source)
                j = rng.choice(others)
            return j

        for pid, kind, value in n["facts"]:
            if rng.random() < err and kind in MW.SLOTS:
                continue                         # parser error: fact missed
            if kind in MW.RELS:                  # (a, rel, b): "b's rel is a"; symmetric relations both ways
                a, b = subj(pid), subj(value)
                noun = kind
                if rng.random() < err:
                    a, b = b, a                  # parser error: direction swapped
                if a is not None and b is not None:
                    claims.append(("rel", a, noun, b, MW.RELS[kind][0]))
            elif kind == "nickname":
                claims.append(("nick", subj(pid), value.lower()))
            elif kind == "known from":
                claims.append(("circle", subj(pid), value))
            elif kind == "relation to me":
                claims.append(("role", subj(pid), value, "new" in n["text"].lower()))
            else:
                claims.append(("fact", subj(pid), kind, value, pid))
        date = n["date"]
        if n["type"] in ("nick", "rel") and rng.random() < defer_frac:
            date = "2026-10-31"
        out[n["id"]] = {"date": date, "order": (date, n["id"]), "descs": descs, "ante": ante, "claims": claims,
                        "truth": truth_pid, "styles": styles}
        prev_ments = [j for j, d in enumerate(descs) if d[0] != "pron" and styles[j] != "source"]
        prev_ments = [(n["id"], j) for j in prev_ments]
        ante = {k: (v if v[0] == "here" else ("prev", v[1])) for k, v in ante.items()}
        out[n["id"]]["ante"] = ante
    return out


def resolve(P, online=False, iters=4):
    """Who every mention is, as a function of the set P. online=True: a definition (nickname, relation, role) is only
    usable by mentions dated after it (write-time binding, never revised)."""
    order = sorted(P, key=lambda i: P[i]["order"])
    first_seen = {}
    for i in order:
        for d in P[i]["descs"]:
            for dd in ([d, d[1]] if d[0] == "via" else [d]):
                if dd[0] == "full":
                    first_seen.setdefault(dd[1], P[i]["date"])
    first_of = defaultdict(set)
    for e in first_seen:
        first_of[e.split()[0]].add(e)
    bind = {}
    for _ in range(iters):
        nick, circle, roles, relof = defaultdict(list), defaultdict(set), defaultdict(list), defaultdict(list)
        for i in order:
            p = P[i]
            for c in p["claims"]:
                if c[0] == "nick" and bind.get((i, c[1])):
                    nick[c[2]].append((p["date"], bind[(i, c[1])]))
                elif c[0] == "circle" and bind.get((i, c[1])):
                    circle[bind[(i, c[1])]].add(c[2])
                elif c[0] == "role" and bind.get((i, c[1])):
                    roles[c[2]].append((p["date"] if c[3] else "0000", bind[(i, c[1])]))
                elif c[0] == "rel":
                    a, b = bind.get((i, c[1])), bind.get((i, c[3]))
                    if a and b:
                        relof[(b, c[2])].append((p["date"], a))
                        if c[4]:
                            relof[(a, c[2])].append((p["date"], b))
        recent = []
        new = {}

        def usable(lst, date):
            return [e for d, e in lst if (not online or d <= date)]

        def rec_pick(cands):
            for e in reversed(recent[-30:]):
                if e in cands:
                    return e
            return None

        def res(d, date):
            t = d[0]
            if t == "full":
                return d[1]
            if t == "role":
                hs = sorted((dd, e) for dd, e in roles.get(d[1], []) if (not online or dd <= date))
                hs = [(dd, e) for dd, e in hs]
                cur = [k for k, (dd, e) in enumerate(hs) if dd <= date]
                if not hs:
                    return None
                k = cur[-1] if cur else 0
                if d[2] == "old":
                    k -= 1
                return hs[k][1] if 0 <= k < len(hs) else None
            if t == "name":
                nk = set(usable(nick.get(d[1], []), date))
                fn = {e for e in first_of.get(d[1], ()) if first_seen[e] <= date}
                cands = nk or fn
                if len(cands) == 1:
                    return next(iter(cands))
                return rec_pick(cands) if cands else None
            if t == "fctx":
                cands = {e for e in first_of.get(d[1], ()) if d[2] in circle.get(e, ())}
                known = {e for e in cands if first_seen[e] <= date}
                if len(known) == 1:
                    return next(iter(known))
                if len(cands) == 1:
                    return next(iter(cands))
                return rec_pick(cands)
            if t == "via":
                a = res(d[1], date)
                if not a:
                    return None
                cands = set(usable(relof.get((a, d[2]), []), date))
                if len(cands) == 1:
                    return next(iter(cands))
                return rec_pick(cands)
            return None

        for i in order:
            p = P[i]
            for k, d in enumerate(p["descs"]):
                if d[0] == "pron":
                    continue
                e = res(d, p["date"])
                new[(i, k)] = e
                if d[0] == "via":                # the anchor ("Kofi" in "Kofi's flatmate") is a mention too
                    new[(i, ("anchor", k))] = res(d[1], p["date"])
            for k, a in p["ante"].items():
                if a[0] == "here":
                    new[(i, k)] = new.get((i, a[1]))
                else:
                    new[(i, k)] = new.get(a[1]) if a[1] in new else bind.get(a[1])
            for k in range(len(p["descs"])):
                if new.get((i, k)) and p["styles"][k] != "source":
                    recent.append(new[(i, k)])
        bind = new
    return bind


def evaluate(W, P, bind):
    full_of = {p["full"].lower(): p["id"] for p in W.people}
    by = defaultdict(lambda: [0, 0, 0])         # right, wrong, unfiled
    for i, p in P.items():
        for c in p["claims"]:
            if c[0] != "fact" or c[1] is None or c[3] is None:
                continue
            true_pid = c[4]
            st = next((p["styles"][j] for j, t in enumerate(p["truth"]) if t == true_pid and
                       p["descs"][j][0] != "pron"), None) or next(
                (p["styles"][j] for j, t in enumerate(p["truth"]) if t == true_pid), "?")
            e = bind.get((i, c[1]))
            if e is None:
                by[st][2] += 1
            elif full_of.get(e) == true_pid:
                by[st][0] += 1
            else:
                by[st][1] += 1
    tot = [sum(v[k] for v in by.values()) for k in range(3)]
    filed = tot[0] + tot[1]
    return {"right_of_filed_by_style": {k: round(100 * v[0] / max(1, v[0] + v[1]), 1) for k, v in sorted(by.items())},
            "unfiled_by_style": {k: round(100 * v[2] / max(1, sum(v)), 1) for k, v in sorted(by.items())},
            "wrong_pct_of_filed": round(100 * tot[1] / max(1, filed), 2),
            "filed_right_pct_of_all": round(100 * tot[0] / max(1, sum(tot)), 1), "n": sum(tot)}


def forget(W, P, bind, targets):
    """'Forget X': delete every message with any mention bound to X (any style). Recall vs the truth's mentions."""
    full_of = {p["full"].lower(): p["id"] for p in W.people}
    res = []
    for pid in targets:
        removed = {i for (i, k), e in bind.items() if e and full_of.get(e) == pid}
        truth = {n["id"] for n in W.notes if any(m[1] == pid for m in n["mentions"])}
        res.append(len(removed & truth) / max(1, len(truth)))
    return round(100 * sum(res) / len(res), 1), round(100 * min(res), 1)


def main():
    W = MW.World("dev")
    firsts = Counter(p["first"] for p in W.people)
    targets = [p["id"] for p in W.people if p["close"]] + [p["id"] for p in W.people if firsts[p["first"]] > 1][:10]
    print("dev world:", len(W.notes), "messages")
    for err in (0.0, 0.02, 0.05, 0.10):
        rng = random.Random(7)
        P = local_parse(W, err, rng)
        t0 = time.time()
        bind = resolve(P)
        dt_ = time.time() - t0
        ev = evaluate(W, P, bind)
        fg = forget(W, P, bind, targets)
        print(f"\nparser error rate {err:.2f}: full re-solve of the whole set {dt_:.2f}s (Python, 1 core)")
        print("  facts filed right (of all facts the parser kept): %.1f%%; wrong person among filed: %.2f%%" %
              (ev["filed_right_pct_of_all"], ev["wrong_pct_of_filed"]))
        print("  right person among filed, by mention style:", ev["right_of_filed_by_style"])
        print("  left unfiled (deferred, not guessed), by style:", ev["unfiled_by_style"])
        print("  forget a person (20 targets): average %.1f%%, worst %.1f%% of their messages" % fg)
    # order invariance: shuffle the dict order; the resolver sorts by (date, id), so the result is identical
    rng = random.Random(7)
    P = local_parse(W, 0.05, rng)
    b1 = resolve(P)
    keys = list(P)
    random.Random(1).shuffle(keys)
    b2 = resolve({k: P[k] for k in keys})
    print("\norder invariance (same set, shuffled arrival):", b1 == b2)
    # exact deletion of a definition: drop every nickname message; facts bound only through a nickname become unbound
    P2 = {i: p for i, p in P.items() if not any(c[0] == "nick" for c in p["claims"])}
    b3 = resolve(P2)
    nick_bound = sum(1 for (i, k), e in b1.items() if e and isinstance(k, int) and P[i]["descs"][k][0] == "name"
                     and P[i]["styles"][k] == "nick")
    nick_after = sum(1 for (i, k), e in b3.items() if e and isinstance(k, int) and i in P2 and
                     P2[i]["descs"][k][0] == "name" and P2[i]["styles"][k] == "nick")
    print(f"delete all nickname-defining messages: nickname mentions bound {nick_bound} -> {nick_after}")
    # deferred definitions: 40% of nickname and relation messages moved to the last day
    for online in (True, False):
        rng = random.Random(7)
        P = local_parse(W, 0.0, rng, defer_frac=0.4)
        ev = evaluate(W, P, resolve(P, online=online))
        print(f"\nuse-before-definition world (40% of nick/rel messages arrive last), {'ONLINE write-time binding' if online else 'SET re-solve (deferred)'}:")
        print("  facts filed right: %.1f%%; wrong among filed %.2f%%" % (ev["filed_right_pct_of_all"], ev["wrong_pct_of_filed"]))
        print("  unfiled by style:", {k: v for k, v in ev["unfiled_by_style"].items() if v})


if __name__ == "__main__":
    main()
