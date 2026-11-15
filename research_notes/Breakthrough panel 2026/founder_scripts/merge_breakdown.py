"""Founder scratch: rebuild B's dev1500 index from the saved writer replies (no model call) and break the wrong
fact rows down by how the true owner was mentioned in that message. Question: are the writer's mistakes local
perception errors, or cross-message binding errors (the part a recurrent writer does in its head)?"""
import json, os, sys
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
src = "/home/user/nichtec/results/messy_index_dev_dev1500/write.jsonl"
import shutil, tempfile
tmp = tempfile.mkdtemp()
p = os.path.join(tmp, "write.jsonl"); shutil.copy(src, p)
ix, st = MI.build_index(None, W, p, lambda s: None)
wq = MI.write_quality(W, ix)
who = wq.pop("who")
print("wrong merge %", round(wq["wrong_merge_pct"], 2), "fact rows", wq["fact_rows"], "split", wq["people_split_into_several_ids"])
truth_people = {n["id"]: {m[1] for m in n["mentions"]} for n in W.notes}
owners = {n["id"]: {x for pid, kind, value in n["facts"] for x in ((pid, value) if kind in MW.RELS else (pid,))} for n in W.notes}
note = {n["id"]: n for n in W.notes}
me = {pid for pid, pp in ix.people.items() if pp["full"].lower().strip(" .") in MI.ME_NAMES}
fact_notes = {n["id"] for n in W.notes if n["facts"]}
rows = [r for r in ix.rows if r["pid"] not in me and r["pid"] in who and not r["link"] and r["kind"] != "mentioned" and r["note"] in fact_notes]
wrong = [r for r in rows if who[r["pid"]] not in owners.get(r["note"], set())]
cat = Counter(); styles_all = Counter(); ex = defaultdict(list)
for r in wrong:
    n = note[r["note"]]
    st_of = {m[1]: m[2] for m in n["mentions"]}
    own = owners[r["note"]]
    s = sorted({st_of.get(o, "unmentioned") for o in own})
    key = "+".join(s)
    # was the wrong index person someone mentioned in this message (wrong person in the message) or not mentioned at all?
    inmsg = who[r["pid"]] in truth_people[r["note"]]
    cat[(key, "wrong-person-in-msg" if inmsg else "person-not-in-msg")] += 1
    if len(ex[key]) < 3:
        ex[key].append((n["text"], r["kind"], r["value"], W.people[who[r["pid"]]]["full"]))
for r in rows:
    n = note[r["note"]]; st_of = {m[1]: m[2] for m in n["mentions"]}
    styles_all["+".join(sorted({st_of.get(o, "unmentioned") for o in owners[r["note"]]}))] += 1
print("wrong rows", len(wrong))
for k, v in cat.most_common():
    print(v, k, "of", styles_all[k[0]], "rows with that style")
for k, v in ex.items():
    print("==", k)
    for e in v: print("   ", e)
