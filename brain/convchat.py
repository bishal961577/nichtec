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
          "somebody", "people in general", "unclear", "unknown", "n/a", "general", "generic"}
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
    keep, info, new = {}, [], []
    for k, mm in enumerate(ms):
        who = (override or {}).get(k, mm.get("who"))
        who = re.sub(r"\s*\(.*?\)\s*", " ", (who or "")).strip().strip('"').strip()
        low = who.lower()
        if low.startswith("your "):
            who = f"{partner}'s {who[5:]}"
            low = who.lower()
        drop = (low in NOBODY or low.startswith("nobody") or low.startswith("no one")
                or low in (owner.lower(), "me", "i", "myself", "the writer", "writer"))
        if drop:
            continue
        third = is_third(mm)
        if not who:
            m2 = dict(mm, previous="")
        elif low in (partner.lower(), "you"):
            m2 = {"text": partner, "kind": "full name" if len(partner.split()) > 1 else "first name", "name": partner,
                  "through": "", "relation": "", "previous": ""}
        else:
            m2 = {"text": who, "kind": "", "name": "", "through": "", "relation": "", "previous": ""}
        keep[k] = len(new)
        new.append(m2)
        part = low in (partner.lower(), "you")
        info.append({"third": third, "partner": part})
    claims = [dict(c, about=keep[c["about"]]) for c in d.get("claims", []) if c.get("about") in keep]
    links = [dict(lk, **{"from": keep[lk["from"]], "to": keep[lk["to"]]}) for lk in d.get("links", [])
             if lk.get("from") in keep and lk.get("to") in keep]
    tb = keep.get(d.get("told_by", -1), -1)
    return {"mentions": new, "claims": claims, "links": links, "told_by": tb}, info


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=sorted(STAGES), default="dev")
    ap.add_argument("--data", required=True, help="REALTALK data/ (dev) or the EmpatheticDialogues folder (exam)")
    ap.add_argument("--work", required=True, help="private folder outside the repository")
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, help="describe only the first N messages (a smoke test)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--stats", action="store_true")
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
    todo = keep[:a.limit] if a.limit else keep
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        describe(llm, todo, log, a.workers, a.work, STAGE["desc"])
    if a.stats:
        descs = {}
        for ln in open(os.path.join(a.work, STAGE["desc"]), encoding="utf-8"):
            r = json.loads(ln)
            descs[r["id"]] = r["description"]
        recs, ents = bind_all(J, descs)
        stats(recs, ents, J, log, "system A")


if __name__ == "__main__":
    main()
