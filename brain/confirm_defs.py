"""Step 3 of the two ends check (reports/Two ends check 2026.md): the owner confirms definitions.

A free CPU simulation on the 4B's cached descriptions of the 1,500 dev messages (no model calls), through the resolver
in brain/describe_bind.py:
  none: step 2 as is (the 4B's own definitions);
  A:    every definition confirmed: the describer's definitions are dropped and the truth's definitions in each message
        are added, entity-level, by full name, as the owner would give them;
  B:    a realistic asking policy: (1) the AI asks about each message where the 4B proposed a definition that survives
        the resolver's filters, and the owner corrects it to the message's true definitions (or none); (2) the AI asks
        "who is this?" at the first unresolved occurrence of a role, nickname, group, "X's flatmate" or unknown first
        name; the answer binds that mention and defines the form for later messages. Pronouns are never asked about.

    python brain/confirm_defs.py
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402

OUT = TE.OUT
DEF_KINDS = ("relation to me", "known from", "nickname")


def is_def(kind):
    k = kind.strip().lower()
    return k in DEF_KINDS or k.startswith("relation") or k.startswith("nick") or k.startswith("known from")


def load_descs():
    descs = {}
    for path in (os.path.join(OUT, "writing.jsonl"), DB.DESC):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            descs[r["id"]] = r.get("description")
    return descs


def everyday_merges(W, ix, who):
    """test 2's wrong-merge count (MI.write_quality) with definition rows left out: of the everyday-fact rows (job,
    city, pet, ...) on messages that state a fact, the share filed under someone the fact is not about."""
    owners = {n["id"]: {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS else (pid,))}
              for n in W.notes}
    me = {pid for pid, p in ix.people.items() if p["full"].lower().strip(" .") in MI.ME_NAMES}
    fact_notes = {n["id"] for n in W.notes if n["facts"]}
    rows = [r for r in ix.rows if r["pid"] not in me and r["pid"] in who and not r["link"] and
            r["kind"] != "mentioned" and r["note"] in fact_notes and not is_def(r["kind"])]
    wrong = [r for r in rows if who[r["pid"]] not in owners.get(r["note"], set())]
    return 100 * len(wrong) / max(1, len(rows)), wrong, len(rows)


def wrong_by_style(W, wrong):
    """Each wrong everyday row, by how the person the fact is really about was mentioned in its message."""
    by = {n["id"]: n for n in W.notes}
    c = Counter()
    for r in wrong:
        n = by[r["note"]]
        st = {m[1]: m[2] for m in n["mentions"]}
        about = [pid for pid, kind, v in n["facts"] if kind in MW.SLOTS]
        c[st.get(about[0], "?") if about else "no fact"] += 1
    return dict(c.most_common())


def truth_defs(W, n):
    """The definitions in message n, entity-level, as the owner would confirm them (people by full name)."""
    out = []
    for pid, kind, v in n["facts"]:
        p = W.people[pid]
        full = p["full"].lower()
        if kind == "relation to me":
            since = p["role_from"].isoformat() if p.get("role_from") else "0000"
            out.append(("role", full, DB.norm_role(str(v)), since))
        elif kind == "nickname":
            out.append(("nick", full, str(v).lower().strip('"').split()[0]))
        elif kind == "known from":
            out.append(("circle", full, DB.gwords(str(v))))
        elif kind in MW.RELS and isinstance(v, int):
            noun = DB.norm_rel(kind)[0]
            out.append(("rel", full, noun, W.people[v]["full"].lower(), noun in DB.SYM))
    return out


def owner_answer(W, n, m, d):
    """The owner's answer to "who is <this mention>?" in message n: (entity or None, definitions for later)."""
    t = TE.match_mention_v2(m.get("text", "") or "", n["mentions"])
    if t is None:
        return None, []
    pid = t[1]
    p = W.people[pid]
    e = p["full"].lower()
    defs = []
    if d[0] == "role":
        since = p["role_from"].isoformat() if p.get("role_from") else "0000"
        defs.append(("role", e, d[1], since))
    elif d[0] == "name":
        if p["nick"] and p["nick"].lower().strip('"').split()[0] == d[1]:
            defs.append(("nick", e, d[1]))
    elif d[0] == "fctx":
        defs.append(("circle", e, d[2] | DB.gwords(p["circle"] or "")))
    elif d[0] == "via":
        anc = next((x for x in n["mentions"] if x[2] == "via_anchor"), None)
        if anc and anc[1] != pid:
            defs.append(("rel", e, d[2], W.people[anc[1]]["full"].lower(), d[2] in DB.SYM))
    return e, defs


def policy_B(W, descs, ids, log):
    by = {n["id"]: n for n in W.notes}
    order = sorted(ids, key=lambda i: (by[i]["date"], i))
    # (1) confirm what the 4B proposed
    P_own = DB.to_parse(W, descs, ids)
    asked0 = [i for i in order if any(c[0] in ("nick", "circle", "role", "rel") for c in P_own[i]["claims"])]
    confirmed = {i: truth_defs(W, by[i]) for i in asked0}
    q = Counter({"confirm a proposed definition": len(asked0)})
    q_none = sum(1 for i in asked0 if not confirmed[i])
    # (2) "who is this?" at the first unresolved occurrence; a form asked before and still unresolved is asked at
    # every unresolved occurrence in the next round (its answer did not help later messages)
    binds, dead, asked, asked_keys = defaultdict(dict), set(), set(), set()
    rounds = 0
    while True:
        rounds += 1
        P = DB.to_parse(W, descs, ids, keep_defs=False, confirmed=confirmed, binds=binds)
        bind = DB.resolve(P)
        new, seen = [], set()
        for i in order:
            for k, d in enumerate(P[i]["descs"]):
                if d[0] in ("pron", "full") or bind.get((i, k)) or (i, k) in asked or d in dead:
                    continue
                if d in seen and d not in asked_keys:
                    continue
                seen.add(d)
                new.append((i, k, d))
        if not new or rounds > 40:
            break
        for i, k, d in new:
            asked.add((i, k))
            asked_keys.add(d)
            e, defs = owner_answer(W, by[i], (descs[i] or {}).get("mentions", [])[k], d)
            q["who is " + {"role": "my <role>", "name": "<first name or nickname>", "fctx": "<name> from <group>",
                           "via": "X's <relation>"}.get(d[0], d[0])] += 1
            if e is None:
                dead.add(d)
                q["  of which: not a person"] += 1
                continue
            binds[i][k] = e
            if defs:
                confirmed.setdefault(i, []).extend(defs)
    total = sum(v for k, v in q.items() if not k.startswith("  "))
    log(f"B: {total} questions in {rounds} rounds; {q_none} confirmations found no true definition")
    return confirmed, binds, q, total, q_none, rounds


def measure(W, P, emb, label, log):
    t0 = time.time()
    bind = DB.resolve(P)
    V = DB.View(W, P, bind)
    wq = MI.write_quality(W, V.ix)
    who = wq.pop("who")
    ev, wrong, n_ev = everyday_merges(W, V.ix, who)
    reader = MI.Reader(V.ix, W, emb)
    fg = MI.forget_test(W, V.ix, reader)
    left = MI.deletion_audit(W, V.ix, who)
    r = {"everyday_wrong_pct": ev, "everyday_rows": n_ev, "everyday_wrong_rows": len(wrong),
         "wrong_by_owner_style": wrong_by_style(W, wrong), "registered_wrong_pct": wq["wrong_merge_pct"],
         "hard_merge_pct": wq["hard_merge_pct"], "via_pct": wq["filed_right_by_style"].get("via", float("nan")),
         "forget_worst_pct": fg["forget_worst_pct"], "forget_avg_pct": fg["forget_recall_pct"],
         "facts_filed_pct": wq["facts_filed_pct"], "link_precision_pct": wq["link_precision_pct"],
         "unresolved_pct": 100 * V.unresolved / max(1, V.mentions), "filed_right_by_style": wq["filed_right_by_style"],
         "filed_n_by_style": wq["filed_n_by_style"], "rows_left_after_deletion": left,
         "people_split_into_several_ids": wq["people_split_into_several_ids"]}
    log(f"{label}: everyday wrong {ev:.2f}% ({len(wrong)}/{n_ev}); registered {wq['wrong_merge_pct']:.2f}%; via "
        f"{r['via_pct']}; forgetting worst {fg['forget_worst_pct']:.1f} avg {fg['forget_recall_pct']:.1f}; filed "
        f"{wq['facts_filed_pct']:.1f}; links {wq['link_precision_pct']:.1f}; unresolved {r['unresolved_pct']:.1f}% "
        f"({time.time() - t0:.0f} s)")
    return r, bind


def verdict(r, questions=None, days=None):
    ok = (r["everyday_wrong_pct"] <= 1.5 and r["via_pct"] >= 75 and r["forget_worst_pct"] >= 90 and
          r["facts_filed_pct"] >= 85 and (questions is None or questions / days <= 2))
    return "PASS" if ok else ("STOP" if r["everyday_wrong_pct"] > 3 else "BETWEEN")


def main():
    W = TE.dev_world()
    assert all(p["id"] == k for k, p in enumerate(W.people))
    logf = open(os.path.join(OUT, "confirm.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} step 3: the owner confirms definitions")
    descs = load_descs()
    assert all(n["id"] in descs for n in W.notes)
    ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
    days = (date.fromisoformat(W.notes[-1]["date"]) - date.fromisoformat(W.notes[0]["date"])).days + 1
    emb = MI.Embedder()
    res = {"days": days, "messages": len(ids)}

    # B's own index (writes with memory in view), on the same yardstick
    bix, _ = MI.build_index(None, W, os.path.join(MI.ROOT, "results", "messy_index_dev_dev1500", "write.jsonl"),
                            lambda s: None)
    bwq = MI.write_quality(W, bix)
    bwho = bwq.pop("who")
    bev, bwrong, bn = everyday_merges(W, bix, bwho)
    res["B_index"] = {"registered_wrong_pct": bwq["wrong_merge_pct"], "everyday_wrong_pct": bev, "everyday_rows": bn,
                      "everyday_wrong_rows": len(bwrong)}
    log(f"B's dev index: registered wrong {bwq['wrong_merge_pct']:.2f}% (step 2 reference 6.27%); everyday wrong "
        f"{bev:.2f}% ({len(bwrong)}/{bn})")

    # step 2 as is (regression: must give 9.42% registered)
    r0, _ = measure(W, DB.to_parse(W, descs, ids), emb, "none (step 2 as is)", log)
    res["none"] = r0

    # A: every definition confirmed
    by = {n["id"]: n for n in W.notes}
    confA = {i: truth_defs(W, by[i]) for i in ids if truth_defs(W, by[i])}
    rA, _ = measure(W, DB.to_parse(W, descs, ids, keep_defs=False, confirmed=confA), emb, "A (every definition)", log)
    rA["questions"] = len(confA)
    res["A"] = rA

    # B: realistic asking policy
    confB, bindsB, qB, totB, q_none, rounds = policy_B(W, descs, ids, log)
    rB, _ = measure(W, DB.to_parse(W, descs, ids, keep_defs=False, confirmed=confB, binds=bindsB), emb,
                    "B (asking policy)", log)
    rB.update(questions=totB, questions_by_type=dict(qB), confirmations_with_nothing_true=q_none, rounds=rounds,
              true_definitions_recovered=sum(1 for i in confB for c in confB[i] if c in confA.get(i, [])),
              true_definitions_total=sum(len(v) for v in confA.values()))
    res["B"] = rB
    for k in ("none", "A", "B"):
        res[k]["verdict"] = verdict(res[k], res[k].get("questions"), days) if k != "none" else verdict(res[k])

    def row(name, key, fmt, line, ref=""):
        return f"| {name} | " + " | ".join(fmt(res[c][key]) if key in res[c] else "–" for c in ("none", "A", "B")) + \
            f" | {line} | {ref} |"
    pct = lambda v: f"{v:.2f}%"
    p1 = lambda v: f"{v:.1f}%"
    L = ["# Step 3: the owner confirms definitions (dev, first 1,500 messages, 1,465 after deletion)", "",
         "Pre-registration: reports/Two ends check 2026.md, step 3. Free CPU simulation on the 4B's cached descriptions.",
         "", "| Measure | Step 2 as is | A: every definition confirmed | B: asking policy | Line | B's index (test 2) |",
         "|---|---|---|---|---|---|",
         row("Wrong merges, everyday facts (primary)", "everyday_wrong_pct", pct, "<= 1.5% (stop > 3%)",
             f"{bev:.2f}%"),
         row("Wrong merges, registered count", "registered_wrong_pct", pct, "reported", f"{bwq['wrong_merge_pct']:.2f}%"),
         row("\"X's flatmate\" filed right", "via_pct", p1, ">= 75%", "11.1%"),
         row("Worst-person forgetting", "forget_worst_pct", p1, ">= 90%", "62.2%"),
         row("Average forgetting", "forget_avg_pct", p1, "reported", ""),
         row("Facts filed right", "facts_filed_pct", p1, ">= 85%", "78.2%"),
         row("Hard-case merges", "hard_merge_pct", p1, "reported", "33.3%"),
         row("Link precision", "link_precision_pct", p1, "reported", "57.1%"),
         row("Mentions left unresolved", "unresolved_pct", p1, "reported", ""),
         row("Owner questions", "questions", lambda v: f"{v} ({v / days:.2f} a day, {100 * v / len(ids):.1f} per 100 "
             "messages)", "B: <= 2 a day", ""),
         row("Rows left carrying a deleted value", "rows_left_after_deletion", str, "reported", ""),
         row("Verdict (dev)", "verdict", str, "", ""), "",
         "Filed right by how the person was mentioned (n facts):", "",
         "| Mention | Step 2 as is | A | B |", "|---|---|---|---|"]
    for st in sorted(set(r0["filed_right_by_style"]) | set(rA["filed_right_by_style"])):
        L.append(f"| {st} | " + " | ".join(f"{res[c]['filed_right_by_style'].get(st, float('nan')):.1f}% "
                                            f"({res[c]['filed_n_by_style'].get(st, 0)})" for c in ("none", "A", "B"))
                 + " |")
    L += ["", "Wrong everyday rows, by how the person the fact is really about was mentioned:", ""]
    for c in ("none", "A", "B"):
        L.append(f"- {c}: {json.dumps(res[c]['wrong_by_owner_style'])}")
    L += ["", f"B's questions by type: {json.dumps(dict(qB))}; rounds: {rounds}; confirmations where the message held "
              f"no true definition: {q_none}; true definitions recovered: {rB['true_definitions_recovered']} of "
              f"{rB['true_definitions_total']}."]
    with open(os.path.join(OUT, "confirm_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "confirm_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("\n".join(L))


if __name__ == "__main__":
    main()
