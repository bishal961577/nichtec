import sys, json
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import messy_world as MW, messy_index as MI, two_ends as TE, describe_bind as DB
W = TE.dev_world()
descs = {}
prev = None
for n in W.notes:
    ms, idx = [], {}
    for k, (s, pid, st) in enumerate(n["mentions"]):
        m = {"text": s, "kind": "full name", "name": "", "through": "", "relation": "", "previous": ""}
        if st in ("pronoun", "pronoun_prev"):
            m["kind"] = "pronoun"
            if st == "pronoun_prev" and prev:
                m["previous"] = next((x[0] for x in prev["mentions"] if x[1] == pid), "")
        elif st == "via":
            anchor, noun = s.rsplit("'s ", 1)
            m.update(kind="through someone", through=anchor, relation=noun)
        elif st in ("role", "old_role"):
            m.update(kind="my relation", relation=s.lower().replace("my old ", "").replace("my ", ""))
        elif st == "fname_ctx":
            m.update(kind="first name with group", name=s.split()[0], relation=s.split(" from ", 1)[1])
        elif st == "nick":
            m.update(kind="nickname", name=s)
        elif st == "fname":
            m.update(kind="first name", name=s)
        elif st == "via_anchor":
            m.update(kind="first name with group" if " from " in s else ("full name" if len(s.split()) > 1 else "first name"),
                     name=s.split()[0] if " from " in s else s, relation=s.split(" from ", 1)[1] if " from " in s else "")
        else:
            m.update(name=s)
        ms.append(m)
    subj = {}
    for k, (s, pid, st) in enumerate(n["mentions"]):
        if st != "via_anchor" and pid not in subj:
            subj[pid] = k
    for k, (s, pid, st) in enumerate(n["mentions"]):
        subj.setdefault(pid, k)
    claims, links = [], []
    for pid, kind, v in n["facts"]:
        if pid not in subj:
            continue
        if kind in MW.SLOTS:
            claims.append({"about": subj[pid], "label": kind, "value": v or "", "status": "true now" if v else "no longer"})
        elif kind in TE.DEFS:
            claims.append({"about": subj[pid], "label": kind, "value": str(v), "status": "true now"})
        elif kind in MW.RELS and v in subj:
            links.append({"from": subj[pid], "relation": kind, "to": subj[v]})
    tb = next((k for k, m in enumerate(n["mentions"]) if m[2] == "source"), -1)
    descs[n["id"]] = {"mentions": ms, "claims": claims, "links": links, "told_by": tb}
    prev = n
ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
P = DB.to_parse(W, descs, ids)
bind = DB.resolve(P)
V = DB.View(W, P, bind)
wq = MI.write_quality(W, V.ix); wq.pop("who")
fg = MI.forget_test(W, V.ix, MI.Reader(V.ix, W, MI.Embedder()))
print("perfect describer -> wrong merges", round(wq["wrong_merge_pct"], 2), "| hard", round(wq["hard_merge_pct"], 1),
      "| filed", round(wq["facts_filed_pct"], 1), "| via", wq["filed_right_by_style"].get("via"),
      "| forget", round(fg["forget_recall_pct"], 1), round(fg["forget_worst_pct"], 1),
      "| unresolved", V.unresolved, "of", V.mentions, "| link precision", round(wq["link_precision_pct"], 1))
print(json.dumps(wq["filed_right_by_style"]))
