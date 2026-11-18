"""Step 8 of the two ends check (reports/Two ends check 2026.md): describe each message with the conversation, and
let models work out who is who instead of hand-written chat rules.

Systems (all bind with step 5's resolver; one memory per owner):
  A  conversation describer v5 on the 27B: it sees the conversation before the message (both people) and writes, for
     each person mentioned, "who" they are (a name, "my crush", "Ana's boss", or "nobody" for a general "you"). The
     code binds the "who" words. Step 7's chat rules (v4) are removed. Claims marked past, plan or tried are kept.
  B  A's descriptions, but every he/she/they mention's "who" comes from the 4B taught by itself (brain/whoquiz.py,
     brain/modal_who4b.py), not from the 27B.
  0  step 7's frozen system (v4b, one message at a time), as a baseline.

Data: REALTALK chats 1-10 (development; github.com/danny911kr/REALTALK, Apache-2.0) and the exam, EmpatheticDialogues'
test split (Rashkin et al. 2019, CC BY-NC 4.0; dl.fbaipublicfiles.com/parlai/empatheticdialogues), read from local
copies outside the repository. Nothing quoting either is committed: only ids, labels and numbers.

    python brain/convchat.py --stage dev --data DIR --work WORK --dry
    ENDPOINT_TOKEN=... python brain/convchat.py --stage dev --data DIR --work WORK --base-url URL/v1
    python brain/convchat.py --stage dev --data DIR --work WORK --export | --score
"""
import argparse
import datetime as dt
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_index as MI  # noqa: E402
import order_gap as G  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402
import realchat as R  # noqa: E402

K = 12                                         # messages of conversation shown before the latest one

# ---- describer v5: step 6's base describer, with the conversation and a "who" field; no chat rules ----
HEAD5 = ("You read the latest message that I, {owner}, sent to {partner} in a chat, and describe it, so that a program "
         "can later work out who is who. Above it you see the conversation before it, from both of us, so you can tell "
         "who each person is. In \"text\", write each person exactly as the message names them; in \"who\", say who "
         "they are.")
WHO = ('  "who": who this person is, using the conversation: their name if the chat gives one ("Julio"); else how I '
       'would name them ("my crush", "my roommate"), or through someone ("{partner}\'s sister", "Julio\'s mom"); '
       '"{partner}" when it is {partner}; "nobody" when the words mean no particular person (a "you" or "they" that '
       'means people in general). Write the same words each time for the same person.')
OLD_PREV = ('  "previous": for a pronoun that points to someone in the previous message, the words naming them there, '
            'else "".')
OLD_ONLY = ('- Describe only the message, never the previous message: the previous message is there only to tell you '
            'who "he", "she" or "they" is.')
NEW_ONLY = "- Describe only the latest message, never the earlier ones: they are there only to tell you who is who."


def _v5():
    s = TE.DESCRIBE
    first = s.split("\n", 1)[0]
    s = s.replace(first, HEAD5, 1)
    for a, b in ((OLD_PREV, WHO), (OLD_ONLY, NEW_ONLY),
                 ('"previous": ""}}, {{"text": "Tessa\'s flatmate"', '"who": "Jonas"}}, {{"text": "Tessa\'s flatmate"'),
                 ('"relation": "flatmate", "previous": ""}}', '"relation": "flatmate", "who": "Tessa\'s flatmate"}}'),
                 ('"name": "Tessa", "through": "", "relation": "", "previous": ""}}',
                  '"name": "Tessa", "through": "", "relation": "", "who": "Tessa"}}'),
                 ('"relation": "niece", "previous": ""}}', '"relation": "niece", "who": "Lena Park"}}'),
                 ('"relation": "work", "previous": ""}}', '"relation": "work", "who": "Omar from work"}}'),
                 ("Example (made-up people). Previous message:", "Example (made-up people). Conversation before:"),
                 ("Second example. Previous message:", "Second example. Conversation before:"),
                 ("\nMessage: [2026-03-03", "\nLatest message: [2026-03-03"),
                 ("\nMessage: [2026-04-02", "\nLatest message: [2026-04-02"),
                 ("Previous message: {prev}\nMessage: {msg}", "Conversation before (oldest first):\n{window}\n\n"
                                                              "Latest message: {msg}")):
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    assert '"previous"' not in s
    return s


DESCRIBE5 = _v5()
KW5 = json.loads(json.dumps(TE.DESCRIBE_KW))
_m = KW5["response_format"]["json_schema"]["schema"]["properties"]["mentions"]["items"]
_m["properties"].pop("previous")
_m["properties"]["who"] = {"type": "string"}
_m["required"] = ["text", "kind", "name", "through", "relation", "who"]
KW5["max_tokens"] = 900

NOBODY = {"nobody", "no one", "noone", "none", "anyone", "anybody", "everyone", "everybody", "people", "someone",
          "somebody", "people in general", "n/a", "general", "generic"}
UNSURE = {"unclear", "unknown", "not sure", "can't tell", "cannot tell"}     # no "who": the mention stays unattached
THIRD = {"he", "she", "they", "him", "her", "them", "his", "hers", "their", "theirs"}

STAGES = {
    "dev": dict(data="realtalk", chats=(5, 6, 7, 8, 9, 10), desc="describe_v5.jsonl", out="conv_dev",
                sample=(450, 150, 300, 8), recall=(150, 70)),
    "exam": dict(data="ed", split="test", desc="describe_v5.jsonl", out="conv_exam",
                 sample=(450, 150, 300, 80), recall=(150, 81)),
}
STAGE = STAGES["dev"]


# ---- data: messages in realchat's shape (chat, speaker, partner, time, text, prev) ----

def load_ed(path):
    """EmpatheticDialogues csv. Each conversation is its own chat between "Speaker A" (who opens) and "Speaker B";
    times are made up (a conversation a minute) since the data has none."""
    rows = open(path, encoding="utf-8").read().split("\n")[1:]
    by = defaultdict(list)
    for r in rows:
        f = r.split(",")
        if len(f) < 6:
            continue
        by[f[0]].append((int(f[1]), f[4], f[5].replace("_comma_", ",").strip()))
    msgs = []
    t0 = dt.datetime(2019, 1, 1)
    for n, cid in enumerate(sorted(by)):
        us = sorted(by[cid])
        first = us[0][1]
        chat = []
        for j, (_, s, text) in enumerate(us):
            sp = "Speaker A" if s == first else "Speaker B"
            chat.append({"chat": cid, "speaker": sp, "partner": "Speaker B" if sp == "Speaker A" else "Speaker A",
                         "time": t0 + dt.timedelta(minutes=n, seconds=j), "text": text, "dia": f"{cid}:{j}"})
        for j, m in enumerate(chat):
            m["prev"] = chat[j - 1] if j else None
        msgs += chat
    return msgs


def load(data):
    if STAGE["data"] == "ed":
        return load_ed(os.path.join(data, f"{STAGE['split']}.csv"))
    R.STAGE = dict(R.STAGES["step7"], chats=STAGE["chats"])
    return R.load(data)


def owner_key(m):
    return f"{m['chat']}|{m['speaker']}" if STAGE["data"] == "ed" else m["speaker"]


def journals(msgs):
    """{owner: Journal}: REALTALK, a participant's messages across their chats; EmpatheticDialogues, one speaker's
    messages in one conversation (strangers, so nothing links two conversations)."""
    keep = [m for m in msgs if m["text"]]
    keep.sort(key=lambda m: (m["time"], str(m["chat"]), str(m["dia"])))
    for i, m in enumerate(keep):
        m["id"] = i
        m["owner"] = owner_key(m)
    out = {}
    by = defaultdict(list)
    for m in keep:
        by[m["owner"]].append(m)
    for o, ms in sorted(by.items()):
        out[o] = R.Journal(o, [{"id": m["id"], "date": m["time"].strftime("%Y-%m-%d"), "text": m["text"], "m": m}
                               for m in ms])
    return out, keep


def window(m, k=K):
    out, p = [], m["prev"]
    while p is not None and len(out) < k:
        if p["text"]:
            out.append(R.line(p))
        p = p["prev"]
    return "\n".join(reversed(out)) or "(none)"


def prompt(m):
    return DESCRIBE5.format(owner=m["speaker"], partner=m["partner"], window=window(m), msg=R.line(m))


def describe(llm, msgs, log, workers, work, desc_name):
    path = os.path.join(work, desc_name)
    done = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            done[r["id"]] = r
    todo = [m for m in msgs if m["id"] not in done]
    f = open(path, "a", encoding="utf-8")

    def one(m):
        r = MI.ask(llm, prompt(m), KW5, False, m["id"])
        try:
            d = json.loads(r["text"])
            d["links"] = [{"from": lk.get("who", -1), "relation": lk.get("is_the", ""), "to": lk.get("of", -1)}
                          for lk in d.get("links", [])]
        except (json.JSONDecodeError, TypeError, AttributeError):
            d = None
        return {"id": m["id"], "description": d, "completion_tokens": r["completion_tokens"],
                "prompt_tokens": r["prompt_tokens"], "cut": r.get("cut", False)}
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for k, rec in enumerate(ex.map(one, todo)):
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            done[rec["id"]] = rec
            if k % 500 == 499:
                f.flush()
                print(f"  {k + 1}/{len(todo)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    f.flush()
    log(f"described: {len(done)} messages ({len(todo)} new, {(time.time() - t0) / 60:.1f} min)")
    return {i: r["description"] for i, r in done.items()}


# ---- binding the "who" words ----

def is_third(mm):
    t = (mm.get("text") or "").strip().lower()
    return mm.get("kind") == "pronoun" and t not in ("you", "your", "u", "ur", "yours") or t in THIRD


def rewrite(d, owner, partner, override=None):
    """The describer's mentions rewritten as their "who" words, so the resolver binds who they are; "nobody" and the
    writer are dropped with their claims. override: {mention index: who} (system B). Returns (desc, info) where info
    lists each kept mention's original kind (pronoun or not) and whether it is the partner."""
    if not d:
        return d, []
    ms = d.get("mentions", [])
    keep, info, new = {}, [], []                       # keep: original mention -> list of new mention indexes
    clean = lambda w: re.sub(r"\s*\(.*?\)\s*", " ", (w or "")).strip().strip('"').strip()
    gone = lambda low: (low in NOBODY or low.startswith("nobody") or low.startswith("no one")
                        or low in (owner.lower(), "me", "i", "myself", "the writer", "writer", "us", "we"))
    group = lambda w: bool(re.search(r"\s(and|&|plus)\s", w))
    for k, mm in enumerate(ms):
        who = clean(mm.get("who"))
        # system B: the 27B still decides whether a he/she/they is one particular person (not "nobody", the writer
        # or a group); the 4B decides which person. Its answer never drops or splits a claim, so A and B hold the
        # same claims and differ only in where they are filed.
        ov = bool(override) and k in override and not gone(who.lower()) and not group(who)
        if ov:
            who = clean(override[k])
            if who.lower() in UNSURE or gone(who.lower()) or group(who):
                who = ""
        if who.lower() in UNSURE:
            who = ""
        # a group ("Julio and Nebraas") is one mention per member; the writer is not one of them. (Dev fix: as one
        # two-word "name", a group took over every later "Julio".) Commas split only in a list with "and".
        parts = [who]
        if group(who):
            parts = [x for x in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+|\s*&\s*|\s+plus\s+", who) if x.strip()]
        third = is_third(mm)
        for w in parts:
            w = w.strip()
            low = w.lower()
            if low.startswith("your "):
                w = f"{partner}'s {w[5:]}"
                low = w.lower()
            if gone(low):
                continue
            if not w and len(parts) == 1:
                m2 = dict(mm, previous="")
            elif low in (partner.lower(), "you"):
                m2 = {"text": partner, "kind": "full name" if len(partner.split()) > 1 else "first name",
                      "name": partner, "through": "", "relation": "", "previous": ""}
            elif not w:
                continue
            else:
                m2 = {"text": w, "kind": "", "name": "", "through": "", "relation": "", "previous": ""}
            keep.setdefault(k, []).append(len(new))
            new.append(m2)
            info.append({"third": third, "partner": low in (partner.lower(), "you")})
    claims = [dict(c, about=j) for c in d.get("claims", []) for j in keep.get(c.get("about"), [])]
    links = [dict(lk, **{"from": keep[lk["from"]][0], "to": keep[lk["to"]][0]}) for lk in d.get("links", [])
             if len(keep.get(lk.get("from"), [])) == 1 and len(keep.get(lk.get("to"), [])) == 1]
    tb = keep.get(d.get("told_by", -1), [-1])
    return {"mentions": new, "claims": claims, "links": links, "told_by": tb[0] if len(tb) == 1 else -1}, info


def bind_all(J, descs, override=None, system="A"):
    """As realchat.bind_all, with the who-rewrite (A, B) or without it (0, step 7's system)."""
    DB.ANTE_V2 = True
    recs, ents = [], {}
    for owner, W in J.items():
        ids = [n["id"] for n in W.notes]
        infos = {}
        dd = {}
        for n in W.notes:
            m = n["m"]
            if system == "0":
                dd[n["id"]] = descs.get(n["id"])
            else:
                dd[n["id"]], infos[n["id"]] = rewrite(descs.get(n["id"]), m["speaker"], m["partner"],
                                                      (override or {}).get(n["id"]))
        P = DB.to_parse(W, dd, ids, drop_reporting=True, keep_status=(system != "0"))
        V = DB.View(W, P, DB.resolve(P))
        for n in W.notes:
            i, m = n["id"], n["m"]
            for c in P[i]["claims"]:
                if c[0] != "fact":
                    continue
                e = V.ent_of[i][c[1]]
                dsc = P[i]["descs"][c[1]]
                if system == "0":
                    third = dsc[0] == "pron"
                    part = dsc[0] == "name" and dsc[1] == m["partner"].lower()
                else:
                    third, part = infos[i][c[1]]["third"], infos[i][c[1]]["partner"]
                recs.append({"cid": f"{i}.{len(recs)}", "id": i, "owner": owner, "label": c[2], "value": c[3],
                             "entity": f"{owner}|{e}", "mention": P[i]["texts"][c[1]],
                             "kind": "pron" if third else dsc[0], "mention_is_partner": part})
        for i in ids:
            for k, e in enumerate(V.ent_of[i]):
                ents.setdefault(f"{owner}|{e}", Counter())[P[i]["texts"][k]] += 1
    DB.ANTE_V2 = False
    return recs, ents


def stats(recs, ents, J, log, tag):
    """What a memory needs besides filing right: how many person records are one-message pronoun placeholders, and
    how many records a journal has per person-word."""
    pron = sum(1 for e in ents if re.search(r"\|\?(he|she|they|him|her|them|his|their|hers|hes|shes)\b.*#\d+$", e))
    log(f"{tag}: {len(recs)} claims on {len({r['entity'] for r in recs})} records; {len(ents)} records in all, "
        f"{pron} unattached pronoun placeholders ({100 * pron / max(1, len(ents)):.1f}%)")


# ---- system B: the 4B's answer for every he/she/they mention ----

def who_items(J, descs, k=10):
    """One quiz item (brain/whoquiz.py's format) per third-person pronoun mention in the descriptions: the pronoun is
    marked in the message, as in the quiz the 4B learned from."""
    import whoquiz as Q
    out = []
    for owner, W in J.items():
        for n in W.notes:
            m, d = n["m"], descs.get(n["id"])
            if not d:
                continue
            seen = Counter()
            for j, mm in enumerate(d.get("mentions", [])):
                t = (mm.get("text") or "").strip()
                if not is_third(mm) or not t:
                    continue
                occ = seen[t.lower()]
                seen[t.lower()] += 1
                hits = list(re.finditer(r"\b" + re.escape(t) + r"\b", m["text"], re.I))
                if occ >= len(hits):
                    continue
                h = hits[occ]
                msg = m["text"][:h.start()] + "⟦" + m["text"][h.start():h.end()] + "⟧" + m["text"][h.end():]
                win, p = [], m["prev"]
                while p is not None and len(win) < k:
                    if p["text"]:
                        win.append((p["speaker"], p["text"]))
                    p = p["prev"]
                it = {"window": list(reversed(win)), "writer": m["speaker"], "partner": m["partner"], "message": msg,
                      "pronoun": m["text"][h.start():h.end()]}
                out.append({"mid": n["id"], "k": j, "prompt": Q.prompt_of(it)})
    return out


def ask_who(url, items, adapter, path, log):
    """Answers from the who-endpoint (brain/modal_who4b.py), cached in WORK."""
    import urllib.request
    done = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            done[(r["mid"], r["k"])] = r["answer"]
    todo = [it for it in items if (it["mid"], it["k"]) not in done]
    f = open(path, "a", encoding="utf-8")
    t0 = time.time()
    for s in range(0, len(todo), 32):
        chunk = todo[s:s + 32]
        req = urllib.request.Request(url.rstrip("/") + "/who", method="POST",
                                     data=json.dumps({"prompts": [c["prompt"] for c in chunk],
                                                      "adapter": adapter}).encode(),
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + os.environ["ENDPOINT_TOKEN"]})
        for attempt in range(5):
            try:
                ans = json.loads(urllib.request.urlopen(req, timeout=900).read())["answers"]
                break
            except Exception as e:                       # cold start or a dropped connection: wait and retry
                log(f"who endpoint: {type(e).__name__}; retry {attempt + 1}")
                time.sleep(30)
        else:
            raise SystemExit("who endpoint did not answer")
        for c, a in zip(chunk, ans):
            f.write(json.dumps({"mid": c["mid"], "k": c["k"], "answer": a}, ensure_ascii=False) + "\n")
            done[(c["mid"], c["k"])] = a
        f.flush()
    log(f"who answers ({'trained' if adapter else 'untrained'} 4B): {len(done)} ({len(todo)} new, "
        f"{(time.time() - t0) / 60:.1f} min)")
    over = defaultdict(dict)
    for (mid, k), a in done.items():
        over[mid][k] = a.strip().strip('"').strip()
    return over


# ---- the exam: export for judging, and scoring (step 7's rules, four systems) ----

def sample(recs, cap, n_part, n_other, seed):
    if len(recs) <= cap:
        return recs, {}
    by = {m: [r for r in recs if r["mention_is_partner"] == m] for m in (True, False)}
    rng = random.Random(seed)
    pick_p = rng.sample(by[True], min(n_part, len(by[True])))
    pick_o = rng.sample(by[False], min(n_other, len(by[False])))
    w = {r["cid"]: len(by[True]) / len(pick_p) for r in pick_p}
    w.update({r["cid"]: len(by[False]) / len(pick_o) for r in pick_o})
    return pick_o + pick_p, w


def recall_ids(keep):
    n, seed = STAGE["recall"]
    return set(random.Random(seed).sample(sorted(m["id"] for m in keep), n))


def export_exam(systems, keep, J, work, log):
    """systems: {name: (recs, ents)}. A, B and B0 share their claims (only the filing differs), so one pass-1 file
    serves all three; system 0 (step 7's) has its own claims and a smaller sample (150, seed 82)."""
    by = {m["id"]: m for m in keep}
    rid = recall_ids(keep)
    sig = lambda rs: [(r["cid"], r["label"], r["value"]) for r in rs]
    shared = [k for k in systems if k != "0"]
    for k in shared[1:]:
        assert sig(systems[k][0]) == sig(systems[shared[0]][0]), f"system {k}'s claims differ from {shared[0]}'s"
    cap, n_part, n_other, seed = STAGE["sample"]
    plans = {"A": sample(systems[shared[0]][0], cap, n_part, n_other, seed)}
    if "0" in systems:
        plans["0"] = (random.Random(82).sample(systems["0"][0], min(150, len(systems["0"][0]))), {})
    for tag, (todo, w) in plans.items():
        suffix = "" if tag == "A" else "_0"
        recs = systems[shared[0]][0] if tag == "A" else systems["0"][0]
        with open(os.path.join(work, f"recall_claims{suffix}.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                if r["id"] in rid:
                    f.write(json.dumps({"cid": r["cid"], "mid": r["id"], "claim": f"{r['label']}: {r['value']}"},
                                       ensure_ascii=False) + "\n")
        order = list(todo)
        random.Random(61).shuffle(order)
        with open(os.path.join(work, f"claims_to_label{suffix}.jsonl"), "w", encoding="utf-8") as f:
            for r in order:
                m = by[r["id"]]
                f.write(json.dumps({"cid": r["cid"], "owner": r["owner"], "partner": m["partner"],
                                    "before": R.context(m), "message": R.line(m),
                                    "claim": f"{r['label']}: {r['value']}"}, ensure_ascii=False) + "\n")
        need = {r["cid"] for r in todo} | {r["cid"] for r in recs if r["id"] in rid}
        names = shared if tag == "A" else ["0"]
        words, held, ex = defaultdict(Counter), Counter(), defaultdict(list)
        for k in names:
            rs, ents = systems[k]
            for r in rs:
                if r["cid"] in need:
                    held[r["entity"]] += 1
                    if len(ex[r["entity"]]) < 3 and r["id"] not in [e[0] for e in ex[r["entity"]]]:
                        ex[r["entity"]].append((r["id"], {"before": R.context(by[r["id"]], 2),
                                                          "message": R.line(by[r["id"]])}))
            for e in held:
                words[e] |= ents.get(e, Counter())
        with open(os.path.join(work, f"entities_to_label{suffix}.jsonl"), "w", encoding="utf-8") as f:
            for e in sorted(held):
                f.write(json.dumps({"entity": e, "owner": e.rsplit("|", 1)[0], "words": dict(words[e].most_common(8)),
                                    "claims": held[e], "examples": [x for _, x in ex[e]]}, ensure_ascii=False) + "\n")
        log(f"export {tag}: {len(recs)} claims ({len(todo)} to judge), {len(held)} records holding them")


def score_exam(systems, keep, work, out, log):
    lab = json.load(open(os.path.join(work, "labels_claims.json"), encoding="utf-8"))
    facts = json.load(open(os.path.join(work, "labels_recall_facts.json"), encoding="utf-8"))
    pct = lambda a, b: 100 * a / b if b else float("nan")
    res = {}
    for k, (recs, ents) in systems.items():
        suffix = "_0" if k == "0" else ""
        lab_k = lab if k != "0" else json.load(open(os.path.join(work, "labels_claims_0.json"), encoding="utf-8"))
        ment = json.load(open(os.path.join(work, f"labels_entities{suffix}.json"), encoding="utf-8"))
        match = json.load(open(os.path.join(work, f"labels_recall_match{suffix}.json"), encoding="utf-8"))
        if k == "0":
            todo, w = random.Random(82).sample(recs, min(150, len(recs))), {}
            grp_of = {r["cid"]: r for r in recs}
        else:                                          # A's sample, weights and groups for A, B and B0 alike
            todo_a, w = sample(systems["A"][0], *STAGE["sample"])
            mine = {r["cid"]: r for r in recs}
            todo = [mine[r["cid"]] for r in todo_a]
            grp_of = {r["cid"]: r for r in systems["A"][0]}
        n, rows = Counter(), []
        kept = defaultdict(Counter)
        for r in todo:
            if r["cid"] not in lab_k:
                continue
            truth, filed, wt = lab_k[r["cid"]], ment.get(r["entity"], "UNKNOWN"), w.get(r["cid"], 1.0)
            if truth == "NONE":
                n["none"] += wt
                continue
            want = {x for x in truth.split("|") if x != "OWNER"}
            ok = bool(want & set(filed.split("|")))
            ga = grp_of[r["cid"]]
            grp = "partner" if ga["mention_is_partner"] else ("pronoun" if ga["kind"] == "pron" else "third party")
            for key, v in (("person", wt), ("wrong", wt * (not ok)), (f"{grp} person", wt), (f"{grp} wrong", wt * (not ok)),
                           ("owner as claim", wt * (truth == "OWNER"))):
                n[key] += v
            if ok:
                kept[(r["owner"], truth)][r["entity"]] += 1
            rows.append({"cid": r["cid"], "truth": truth, "filed": filed, "ok": ok, "group": grp})
        rmap = {r["cid"]: r for r in recs}
        got = sum(1 for fid, fct in facts.items() if match.get(fid, "MISS") in rmap and
                  ({x for x in fct["person"].split("|") if x != "OWNER"} &
                   set(ment.get(rmap[match[fid]]["entity"], "UNKNOWN").split("|"))))
        multi = [c for c in kept.values() if sum(c.values()) >= 2]
        r_ = {"claims": len(recs), "judged": len(rows) + int(sum(1 for r in todo if lab_k.get(r["cid"]) == "NONE")),
              "person_claims": len(rows), "wrong_person_pct": pct(n["wrong"], n["person"]),
              "not_a_fact_pct": pct(n["none"], n["none"] + n["person"]),
              "owner_fact_as_claim_pct": pct(n["owner as claim"], n["person"]),
              "by_group": {g: {"n": n[f"{g} person"], "wrong_pct": pct(n[f"{g} wrong"], n[f"{g} person"])}
                           for g in ("partner", "third party", "pronoun")},
              "recall_facts": len(facts), "recall_right": got, "recall_pct": pct(got, len(facts)),
              "people_in_one_record_pct": pct(sum(1 for c in multi if len(c) == 1), len(multi)),
              "claims_per_message": len(recs) / len(keep)}
        wp, rc = r_["wrong_person_pct"], r_["recall_pct"]
        r_["verdict"] = ("PASS" if wp <= 1.5 and rc >= 70 else "STOP" if wp > 3 or rc < 50 else "BETWEEN")
        res[k] = r_
        json.dump(rows, open(os.path.join(out, f"scored_claims_{k}.json"), "w"), indent=0)
        log(f"system {k}: " + json.dumps(r_))
    json.dump(res, open(os.path.join(out, "result.json"), "w"), indent=1)
    return res


def read_descs(path):
    out = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        out[r["id"]] = r["description"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default="dev")
    ap.add_argument("--data", required=True, help="REALTALK data/ (dev) or the EmpatheticDialogues folder (exam)")
    ap.add_argument("--work", required=True, help="private folder outside the repository")
    ap.add_argument("--base-url", help="the 27B (describes for systems A and 0)")
    ap.add_argument("--who-url", help="the 4B who-endpoint (systems B and B0)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, help="describe only the first N messages (a smoke test)")
    ap.add_argument("--baseline", action="store_true", help="also describe with step 7's frozen v4b (system 0)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--recall-export", action="store_true", help="the recall sample, before anything is described")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    global STAGE
    STAGE = STAGES[a.stage]
    out = os.path.join(TE.OUT, STAGE["out"])
    msgs = load(a.data)
    J, keep = journals(msgs)
    os.makedirs(out, exist_ok=True)
    logf = open(os.path.join(out, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {a.stage}: {len(keep)} messages with text, {len(J)} journals")
    if a.dry:
        print(prompt(keep[min(len(keep) - 1, 4000)]))
        return
    assert not os.path.abspath(a.work).startswith(os.path.abspath(MI.ROOT)), "WORK must be outside the repository"
    os.makedirs(a.work, exist_ok=True)
    if a.recall_export:
        by = {m["id"]: m for m in keep}
        with open(os.path.join(a.work, "recall_to_label.jsonl"), "w", encoding="utf-8") as f:
            for mid in sorted(recall_ids(keep)):
                m = by[mid]
                f.write(json.dumps({"mid": mid, "owner": m["owner"], "partner": m["partner"], "before": R.context(m),
                                    "message": R.line(m)}, ensure_ascii=False) + "\n")
        log(f"recall export: {STAGE['recall'][0]} messages (seed {STAGE['recall'][1]}); file in WORK")
        return
    todo = keep[:a.limit] if a.limit else keep
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        describe(llm, todo, log, a.workers, a.work, STAGE["desc"])
        if a.baseline:
            R.STAGE = dict(prompt=R.CHAT_DESCRIBE_V4, desc="describe_v4b.jsonl")
            R.describe(llm, todo, log, a.workers, a.work)
    descs = read_descs(os.path.join(a.work, STAGE["desc"]))
    if len(descs) < len(keep):
        raise SystemExit(f"{len(keep) - len(descs)} messages not described yet")
    systems = {"A": bind_all(J, descs)}
    if a.who_url:
        items = who_items(J, descs)
        log(f"who items (he/she/they mentions): {len(items)}")
        systems["B"] = bind_all(J, descs, ask_who(a.who_url, items, True, os.path.join(a.work, "who_tuned.jsonl"), log))
        systems["B0"] = bind_all(J, descs, ask_who(a.who_url, items, False, os.path.join(a.work, "who_base.jsonl"),
                                                   log))
    p0 = os.path.join(a.work, "describe_v4b.jsonl")
    if os.path.exists(p0):
        d0 = read_descs(p0)
        if len(d0) >= len(keep):
            systems["0"] = bind_all(J, d0, system="0")
    for k, (recs, ents) in systems.items():
        stats(recs, ents, J, log, f"system {k}")
    if a.export:
        export_exam(systems, keep, J, a.work, log)
    if a.score:
        score_exam(systems, keep, a.work, out, log)


if __name__ == "__main__":
    main()
