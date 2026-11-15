"""Founder round 2: does cross-message binding explain B's accuracy gap, or only its wrong rows?
(1) every true fact in dev1500: no row written / written under wrong person / right person, by mention style;
(2) B's failed hard questions (thinking on): were all needed messages handed to the reader? For each needed
message that was missed: did B's index hold a row from it under the right person, the wrong person, or none?
No model call; reads saved writer replies and saved answers only."""
import json, os, sys, shutil, tempfile
from collections import Counter, defaultdict
sys.path.insert(0, "/home/user/nichtec/brain")
import messy_world as MW
import messy_index as MI

W = MW.World("dev")
lim = 1500
keep = {n["id"] for n in W.notes[:lim]}
W.notes = W.notes[:lim]
W.deleted = [d for d in W.deleted if d[2] in keep]
W.deleted_ids = {d[2] for d in W.deleted}
tmp = tempfile.mkdtemp(); p = os.path.join(tmp, "write.jsonl")
shutil.copy("/home/user/nichtec/results/messy_index_dev_dev1500/write.jsonl", p)
ix, st = MI.build_index(None, W, p, lambda s: None)
wq = MI.write_quality(W, ix); who = wq.pop("who")
print("check: wrong merge", round(wq["wrong_merge_pct"], 2), "facts filed", round(wq["facts_filed_pct"], 1))
note = {n["id"]: n for n in W.notes}
rows_by_note = defaultdict(list)
for r in ix.rows:
    rows_by_note[r["note"]].append(r)

# (1) facts
cat = Counter(); bystyle = defaultdict(Counter)
for n in W.notes:
    style = {m[1]: m[2] for m in n["mentions"]}
    for pid, s, v in n["facts"]:
        if s not in MW.SLOTS or v is None:
            continue
        keys = MW.value_keys(s, v)
        match = [r for r in rows_by_note[n["id"]] if any(MI.has_key(r["value"], k) for k in keys)]
        if not match:
            c = "not written"
        elif any(who.get(r["pid"]) == pid for r in match):
            c = "right person"
        elif any(r["pid"] not in who for r in match):
            c = "written, person unidentifiable"
        else:
            c = "written, wrong person"
        cat[c] += 1; bystyle[style.get(pid, "?")][c] += 1
tot = sum(cat.values())
print("\n(1) true facts:", tot)
for k, v in cat.most_common():
    print(f"  {k}: {v} ({100*v/tot:.1f}%)")
for s, c in sorted(bystyle.items()):
    t = sum(c.values()); print(f"  {s:13s} n={t:4d} " + ", ".join(f"{k} {100*v/t:.0f}%" for k, v in c.most_common()))

# (2) B's failed hard questions
ans = [json.loads(l) for l in open("/home/user/nichtec/results/messy_index_dev_dev1500/answers.jsonl")]
qs = {q["qid"]: q for q in W.questions}
for arm in ("B", "oracle"):
    hard = [a for a in ans if a["arm"] == arm and a["mode"] == "think" and a["kind"] in MW.HARD]
    fail = [a for a in hard if not a["correct"]]
    allfound = [a for a in fail if a.get("gold_total") and len(a["gold_found"]) == a["gold_total"]]
    nogold = [a for a in fail if not a.get("gold_total")]
    print(f"\n(2) {arm} think hard: n={len(hard)} wrong={len(fail)}; all needed messages handed over yet wrong={len(allfound)}; "
          f"no needed messages defined={len(nogold)}; some needed message missing={len(fail)-len(allfound)-len(nogold)}")
    if arm != "B":
        continue
    miss = Counter(); misskind = Counter(); per_q = Counter()
    for a in fail:
        if not a.get("gold_total") or len(a["gold_found"]) == a["gold_total"]:
            continue
        q = qs[a["qid"]]
        kinds = set()
        for nid in set(q["notes"]) - set(a["gold_found"]):
            n = note.get(nid)
            if n is None:
                kinds.add("beyond slice"); continue
            owners = {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS else (pid,))} or {m[1] for m in n["mentions"]}
            rs = [r for r in rows_by_note[nid] if r["kind"] != "mentioned"]
            rsall = rows_by_note[nid]
            if not rsall:
                k = "no row at all from that message"
            elif any(who.get(r["pid"]) in owners for r in rs):
                k = "row under right person, still not handed over"
            elif rs:
                k = "rows only under wrong/unidentified person"
            else:
                k = "only a 'mentioned' tag"
            miss[(n["type"], k)] += 1; kinds.add(k)
        for k in kinds:
            per_q[k] += 1
        misskind[a["kind"]] += 1
    print("  missed needed messages, by message type and what B's index held:")
    for k, v in miss.most_common():
        print("   ", v, k)
    print("  failed questions touched by each cause:", dict(per_q))
    print("  by question kind:", dict(misskind))

# (3) the largest bucket: the row was under the right person, yet the reader was not handed the message. Why?
ids_of = defaultdict(set)
for pid, t in who.items():
    ids_of[t].add(pid)
print("\n(3) B think hard failures with a needed message whose row sat under the right person:")
why = Counter(); bykind = defaultdict(Counter)
for a in ans:
    if a["arm"] != "B" or a["mode"] != "think" or a["kind"] not in MW.HARD or a["correct"]:
        continue
    q = qs[a["qid"]]
    missing = set(q["notes"]) - set(a["gold_found"])
    hit = False
    split = False
    for nid in missing:
        n = note.get(nid)
        if n is None: continue
        owners = {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS else (pid,))} or {m[1] for m in n["mentions"]}
        rs = [r for r in rows_by_note[nid] if r["kind"] != "mentioned" and who.get(r["pid"]) in owners]
        if rs:
            hit = True
            split |= any(len(ids_of[who[r["pid"]]]) > 1 for r in rs)
    if not hit:
        continue
    target = q["person"]
    handed = set(a["notes"])
    target_notes = {n["id"] for n in W.notes if any(m[1] == target for m in n["mentions"])}
    got_target = len(handed & target_notes)
    k = ("target person split over several index ids" if split else
         "reader found none of the target's messages" if got_target == 0 else
         "reader found some of the target's messages, not the needed one")
    why[k] += 1; bykind[k][a["kind"]] += 1
for k, v in why.most_common():
    print(f"  {v} {k}: {dict(bykind[k])}")

# (4) one cause per failed question, write-time identity first
print("\n(4) B think hard failures, one cause each (priority order):")
excl = Counter(); ex = defaultdict(list)
for a in ans:
    if a["arm"] != "B" or a["mode"] != "think" or a["kind"] not in MW.HARD or a["correct"]:
        continue
    q = qs[a["qid"]]
    missing = set(q["notes"]) - set(a["gold_found"])
    if not missing:
        excl["E reading: every needed message handed over"] += 1; continue
    flags = set()
    for nid in missing:
        n = note.get(nid)
        if n is None: continue
        owners = {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS else (pid,))} or {m[1] for m in n["mentions"]}
        rs = [r for r in rows_by_note[nid] if r["kind"] != "mentioned"]
        right = [r for r in rs if who.get(r["pid"]) in owners]
        if rs and not right:
            flags.add("A")
        elif right and any(len(ids_of[who[r["pid"]]]) > 1 for r in right):
            flags.add("B")
        elif not rs:
            flags.add("C")
        else:
            flags.add("D")
    c = min(flags) if flags else "?"
    label = {"A": "A write-time identity: filed under wrong/unidentified person",
             "B": "B write-time identity: person split over several ids",
             "C": "C writer coverage: fact not written from its own message",
             "D": "D question-time: row filed right, reader did not find it"}[c]
    excl[label] += 1
    if len(ex[label]) < 2: ex[label].append((a["kind"], q["q"]))
for k in sorted(excl):
    print(f"  {excl[k]:3d}  {k}")
print("  total", sum(excl.values()))
for k in sorted(ex):
    print(" ", k[:1], ex[k])
