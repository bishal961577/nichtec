"""Step 6 of the two ends check (reports/Two ends check 2026.md): the real-chat check on REALTALK.

REALTALK (Lee et al. 2025, arXiv 2502.13270; github.com/danny911kr/REALTALK, Apache-2.0) is 21 days of real messaging
between consenting participants. The data are read from a local clone outside the repository and are never copied
into it: only ids, labels and numbers are committed.

Setting: each participant of chats 1-4 (Emi, Elise, Kevin, Paola) is an owner. An owner's journal is their own messages
across their two chats, in time order. Each message is described memory-blind from its writer's point of view, with the
chat's previous message (either speaker) for context, by describer v3 plus one registered change for chat: "you" is the
chat partner, a named person. The resolver is frozen at step 5's version (antecedent rule fixed, reporting-verb filter
on); each journal is bound on its own (one memory per person).

    python brain/realchat.py --data DIR --work WORK --dry               # journals, one prompt (no model call)
    ENDPOINT_TOKEN=... python brain/realchat.py --data DIR --work WORK --base-url URL/v1   # describe (cached)
    python brain/realchat.py --data DIR --work WORK --export            # the two judging files
    python brain/realchat.py --data DIR --work WORK --score             # compare the judge's labels
WORK holds everything that quotes REALTALK (descriptions, judging files): keep it outside the repository. Only the
labels (claim ids and person ids), the summary and the log, which quote no message, are committed.

Judging, registered before any description was seen (reports/Two ends check 2026.md, step 6):
  pass 1, claims_to_label.jsonl: each claim with its message and the chat's previous messages, but not the mention it
          was attached to or where it was filed; the judge writes who it is about (a person id from people.json, or
          OWNER for a fact about the writer, or NONE for something that is no fact about a person);
  pass 2, entities_to_label.jsonl: each of the resolver's people that holds a claim, with its mention words and
          example messages; the judge writes which person id it is.
  A claim is filed right when the two agree. If there are more than 600 claims, all claims not attached to the
  partner's mention are judged, plus 200 of the partner's drawn at random (seed 6), and the overall rate is weighted.
"""
import argparse
import datetime as dt
import glob
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

OUT = os.path.join(TE.OUT, "realchat")         # committed: summary, log, labels (no message text)
CHATS = (1, 2, 3, 4)
FIRST = TE.DESCRIBE.split("\n", 1)[0]
assert FIRST.startswith("You read one message from a person's private notebook")
# The one registered change for chat (step 6): the head names the writer and the listener; "you" is a mention.
HEAD = ("You read one message that I, {owner}, sent to {partner} in a chat, and describe it, so that a program can "
        "later work out who is who. Do not guess who anyone is: write each person exactly as the message names them. "
        "You see no other messages.\n\n"
        "In this chat, \"you\", \"your\", \"u\" and \"ur\" mean {partner}: when the message says something about "
        "{partner}, list it as a mention with \"text\" as written, \"kind\" \"first name\" and \"name\" \"{partner}\". "
        "\"We\" and \"us\" are not mentions.")
CHAT_DESCRIBE = TE.DESCRIBE.replace(FIRST, HEAD, 1)


class Journal:
    """A W-like object for describe_bind: notes with id, date, text; nothing deleted."""

    def __init__(self, owner, notes):
        self.owner, self.notes, self.deleted_ids = owner, notes, set()


def load(data):
    msgs = []
    for c in CHATS:
        f = glob.glob(os.path.join(data, f"Chat_{c}_*.json"))
        assert len(f) == 1, (c, f)
        d = json.load(open(f[0], encoding="utf-8"))
        names = {d["name"]["speaker_1"].strip().lower(): d["name"]["speaker_1"].strip().capitalize(),
                 d["name"]["speaker_2"].strip().lower(): d["name"]["speaker_2"].strip().capitalize()}
        chat = []
        for k in sorted((k for k in d if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])):
            for m in d[k]:
                t = dt.datetime.strptime(m["date_time"].strip(), "%d.%m.%Y, %H:%M:%S")
                chat.append({"chat": c, "speaker": names[m["speaker"].strip().lower()], "time": t,
                             "text": (m.get("clean_text") or "").strip(), "dia": m.get("dia_id")})
        chat.sort(key=lambda m: m["time"])
        people = sorted(set(names.values()))
        for j, m in enumerate(chat):
            m["partner"] = next(p for p in people if p != m["speaker"])
            m["prev"] = chat[j - 1] if j else None
        msgs += chat
    return msgs


def journals(msgs):
    """{owner: Journal} of the owner's own messages with text, in time order; ids are global and stable."""
    keep = [m for m in msgs if m["text"]]
    keep.sort(key=lambda m: (m["time"], m["chat"], str(m["dia"])))
    for i, m in enumerate(keep):
        m["id"] = i
    out = {}
    for owner in sorted({m["speaker"] for m in keep}):
        notes = [{"id": m["id"], "date": m["time"].strftime("%Y-%m-%d"), "text": m["text"], "m": m}
                 for m in keep if m["speaker"] == owner]
        out[owner] = Journal(owner, notes)
    return out, keep


def line(m):
    return f"[{MI.dshow(m['time'].strftime('%Y-%m-%d'))}] {m['speaker']}: {m['text']}" if m else "(none)"


def prompt(m):
    prev = m["prev"]
    while prev is not None and not prev["text"]:
        prev = prev["prev"]
    return CHAT_DESCRIBE.format(owner=m["speaker"], partner=m["partner"], prev=line(prev), msg=line(m))


def describe(llm, msgs, log, workers, work):
    path = os.path.join(work, "describe.jsonl")
    done = {}
    if os.path.exists(path):
        for ln in open(path, encoding="utf-8"):
            r = json.loads(ln)
            done[r["id"]] = r
    todo = [m for m in msgs if m["id"] not in done]
    f = open(path, "a", encoding="utf-8")

    def one(m):
        r = MI.ask(llm, prompt(m), TE.DESCRIBE_KW, False, m["id"])
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
        for rec in ex.map(one, todo):
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            done[rec["id"]] = rec
    f.flush()
    log(f"described: {len(done)} messages ({len(todo)} new, {(time.time() - t0) / 60:.1f} min)")
    return {i: r["description"] for i, r in done.items()}


def bind_all(J, descs):
    """Each journal bound on its own, with step 5's frozen resolver. Returns per-claim records."""
    DB.ANTE_V2 = True
    recs, ents = [], {}
    for owner, W in J.items():
        ids = [n["id"] for n in W.notes]
        P = DB.to_parse(W, descs, ids, drop_reporting=True)
        V = DB.View(W, P, DB.resolve(P))
        for i in ids:
            for c in P[i]["claims"]:
                if c[0] != "fact":
                    continue
                e = V.ent_of[i][c[1]]
                m = W.notes[[n["id"] for n in W.notes].index(i)]["m"]
                d = P[i]["descs"][c[1]]
                recs.append({"cid": f"{i}.{len(recs)}", "id": i, "owner": owner, "label": c[2], "value": c[3],
                             "entity": f"{owner}|{e}", "mention": P[i]["texts"][c[1]], "kind": d[0],
                             "mention_is_partner": d[0] == "name" and d[1] == m["partner"].lower()})
        for i in ids:
            for k, e in enumerate(V.ent_of[i]):
                ents.setdefault(f"{owner}|{e}", Counter())[P[i]["texts"][k]] += 1
    DB.ANTE_V2 = False
    return recs, ents


def sample(recs):
    """Claims to judge: all of them up to 600; else all not on the partner's mention plus 200 of the partner's."""
    if len(recs) <= 600:
        return recs, {}
    by = {m: [r for r in recs if (r["mention_is_partner"]) == m] for m in (True, False)}
    pick = random.Random(6).sample(by[True], min(200, len(by[True])))
    w = {r["cid"]: len(by[True]) / len(pick) for r in pick}
    return by[False] + pick, w


def context(m, k=4):
    out, p = [], m["prev"]
    while p is not None and len(out) < k:
        if p["text"]:
            out.append(line(p))
        p = p["prev"]
    return list(reversed(out))


def export(recs, ents, keep, J, work, log):
    by = {m["id"]: m for m in keep}
    todo, w = sample(recs)
    rng = random.Random(61)
    order = list(todo)
    rng.shuffle(order)                                  # judged in random order, not grouped by filing
    with open(os.path.join(work, "claims_to_label.jsonl"), "w", encoding="utf-8") as f:
        for r in order:
            m = by[r["id"]]
            f.write(json.dumps({"cid": r["cid"], "owner": r["owner"], "partner": m["partner"], "before": context(m),
                                "message": line(m), "claim": f"{r['label']}: {r['value']}"}, ensure_ascii=False) + "\n")
    held = Counter(r["entity"] for r in recs)
    with open(os.path.join(work, "entities_to_label.jsonl"), "w", encoding="utf-8") as f:
        for e, n in sorted(held.items()):
            owner = e.split("|", 1)[0]
            ex = []
            for nt in J[owner].notes:
                if len(ex) >= 3:
                    break
                if any(r["entity"] == e and r["id"] == nt["id"] for r in recs):
                    ex.append({"before": context(nt["m"], 2), "message": line(nt["m"])})
            f.write(json.dumps({"entity": e, "owner": owner, "words": dict(ents[e].most_common(8)), "claims": n,
                                "examples": ex}, ensure_ascii=False) + "\n")
    log(f"export: {len(recs)} claims ({len(todo)} to judge{', sampled' if w else ''}), {len(held)} people holding "
        f"claims; files in WORK")


def score(recs, ents, keep, descs, work, log):
    lab = json.load(open(os.path.join(work, "labels_claims.json"), encoding="utf-8"))
    ment = json.load(open(os.path.join(work, "labels_entities.json"), encoding="utf-8"))
    todo, w = sample(recs)
    n = Counter()
    rows = []
    for r in todo:
        if r["cid"] not in lab:
            continue
        truth, filed = lab[r["cid"]], ment.get(r["entity"], "UNKNOWN")
        wt = w.get(r["cid"], 1.0)
        if truth == "NONE":
            n["none"] += wt
            continue
        # judging conventions (step 6), set before scoring:
        # - a label may be a set "a|b|c": identical claims about several people named together ("I think it was:
        #   Harvey, Mike, Jessica and Donna"), or an entity whose mentions the text cannot tell apart (two unnamed
        #   friends who may be one person); a claim is right when the two sets overlap;
        # - as registered, a fact about the writer (OWNER) is wrong-person wherever it is filed: OWNER never counts
        #   towards a match.
        want = {x for x in truth.split("|") if x != "OWNER"}
        ok = bool(want & set(filed.split("|")))
        n["person"] += wt
        n["wrong"] += wt * (not ok)
        grp = "partner" if r["mention_is_partner"] else ("pronoun" if r["kind"] == "pron" else "third party")
        n[f"{grp} person"] += wt
        n[f"{grp} wrong"] += wt * (not ok)
        n["owner as claim"] += wt * (truth == "OWNER")
        rows.append({"cid": r["cid"], "truth": truth, "filed": filed, "ok": ok, "group": grp})
    pct = lambda a, b: 100 * a / b if b else float("nan")
    res = {"claims": len(recs), "judged": len(rows) + int(sum(1 for r in todo if lab.get(r["cid"]) == "NONE")),
           "wrong_person_pct": pct(n["wrong"], n["person"]), "not_a_fact_pct": pct(n["none"], n["none"] + n["person"]),
           "owner_fact_as_claim_pct": pct(n["owner as claim"], n["person"]),
           "by_group": {g: {"n": n[f"{g} person"], "wrong_pct": pct(n[f"{g} wrong"], n[f"{g} person"])}
                        for g in ("partner", "third party", "pronoun")},
           "claims_per_message": len(recs) / len(keep),
           "unparsed": sum(1 for m in keep if descs.get(m["id"]) is None)}
    res["verdict"] = ("PASS" if res["wrong_person_pct"] <= 1.5 else "STOP" if res["wrong_person_pct"] > 3 else "BETWEEN")
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=1)
    json.dump(rows, open(os.path.join(OUT, "scored_claims.json"), "w"), indent=0)
    log(json.dumps(res))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="REALTALK's data/ folder (a local clone, outside the repository)")
    ap.add_argument("--work", required=True, help="private folder outside the repository")
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    msgs = load(a.data)
    J, keep = journals(msgs)
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} step 6: REALTALK chats {CHATS}: {len(keep)} messages with text "
        f"(of {len(msgs)}); journals: " + ", ".join(f"{o} {len(W.notes)}" for o, W in J.items()))
    if a.dry:
        print(prompt(keep[40]))
        return
    assert not os.path.abspath(a.work).startswith(os.path.abspath(MI.ROOT)), "WORK must be outside the repository"
    os.makedirs(a.work, exist_ok=True)
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        describe(llm, keep, log, a.workers, a.work)
    path = os.path.join(a.work, "describe.jsonl")
    if not os.path.exists(path):
        raise SystemExit("no descriptions yet: run with --base-url")
    descs = {}
    for ln in open(path, encoding="utf-8"):
        r = json.loads(ln)
        descs[r["id"]] = r["description"]
    missing = [m["id"] for m in keep if m["id"] not in descs]
    if missing:
        raise SystemExit(f"{len(missing)} messages not described yet")
    recs, ents = bind_all(J, descs)
    if a.export:
        export(recs, ents, keep, J, a.work, log)
    if a.score:
        score(recs, ents, keep, descs, a.work, log)


if __name__ == "__main__":
    main()
