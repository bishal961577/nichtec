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
# Step 7: describer v4 for chat, written after step 6's diagnosis (non-facts written down as claims; "your X" read as
# "my X"; the writer's own facts handed to the partner; pets, companies, countries and "we" as people).
RULES_V4 = (
    "\n\nRules for chat, before the general ones below:\n"
    "- \"You\", \"your\", \"u\" and \"ur\" mean {partner}. When the message says something about {partner}, "
    "list \"you\" as a mention with \"kind\" \"first name\" and \"name\" \"{partner}\".\n"
    "- {partner}'s relatives and friends are named through {partner}: \"your mom\", \"ur sister\", \"your friend\" "
    "are \"through someone\" mentions with \"through\" \"{partner}\" and the relation word (\"mom\"). Only "
    "\"my ...\" is my relation.\n"
    "- I am {owner}, the writer: \"I\", \"me\", \"my\", \"we\" and \"us\" are never mentions, and nothing about "
    "me is a claim, even when the message also asks {partner} something (\"Yes, I've been to Italy, have you?\" has "
    "no claim). When \"we\" or \"me and\" includes someone else (\"me and my mom\"), that person is a mention.\n"
    "- Only people are mentions: not pets, sports teams, companies, brands, countries or places.\n"
    "- A claim is a fact the message tells or repeats about a person. A message that only asks, greets, thanks, "
    "compliments, reacts (\"that sounds amazing\") or advises has no claim: \"How was your trip?\", \"Did you get "
    "the job?\". But a reaction that names a fact keeps it: \"Congrats on the new job!\" (a new job), \"Enjoy your "
    "anniversary trip to Santorini!\" (an anniversary trip to Santorini).")
# Dev log (chats 1-4): v4a, the first version, had "A question, greeting, thanks, compliment, reaction ("that sounds
# amazing"), wish ("enjoy your trip") or advice is not a claim" and no "me and my mom" sentence. It cut claims on
# step 6's no-fact messages from 272 to 43, but 102 of step 6's 314 facts judged filed right were on messages where it
# wrote no claim at all (40 of them questions; the rest mostly reactions that repeat a fact, "your anniversary trip",
# and "me and my mom"). v4 (this text, called v4b in the report) is the version frozen for step 7.
CHAT_DESCRIBE_V4 = TE.DESCRIBE.replace(FIRST, HEAD + RULES_V4, 1)
# stage -> chats, describer, description file, results folder, judging sample (cap, partner n, other n, seed)
STAGES = {
    "step6": dict(chats=(1, 2, 3, 4), prompt=CHAT_DESCRIBE, desc="describe.jsonl", out="realchat",
                  sample=(600, 200, None, 6)),
    "step7dev": dict(chats=(1, 2, 3, 4), prompt=CHAT_DESCRIBE_V4, desc="describe_v4b.jsonl", out="realchat_v4_dev",
                     sample=(450, 150, 300, 7)),
    "step7": dict(chats=(5, 6, 7, 8, 9, 10), prompt=CHAT_DESCRIBE_V4, desc="describe_v4b.jsonl", out="realchat_v4",
                  sample=(450, 150, 300, 7), recall=(150, 70)),
}
STAGE = STAGES["step6"]


class Journal:
    """A W-like object for describe_bind: notes with id, date, text; nothing deleted."""

    def __init__(self, owner, notes):
        self.owner, self.notes, self.deleted_ids = owner, notes, set()


def load(data):
    msgs = []
    for c in STAGE["chats"]:
        f = glob.glob(os.path.join(data, f"Chat_{c}_*.json"))
        assert len(f) == 1, (c, f)
        d = json.load(open(f[0], encoding="utf-8"))
        up = lambda x: x.strip()[:1].upper() + x.strip()[1:]       # "elise" -> "Elise"; "Fahim Khan" kept
        names = {d["name"]["speaker_1"].strip().lower(): up(d["name"]["speaker_1"]),
                 d["name"]["speaker_2"].strip().lower(): up(d["name"]["speaker_2"])}
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
    return STAGE["prompt"].format(owner=m["speaker"], partner=m["partner"], prev=line(prev), msg=line(m))


def describe(llm, msgs, log, workers, work):
    path = os.path.join(work, STAGE["desc"])
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
    """Claims to judge (the stage's registered rule). Step 6: all up to 600, else all not on the partner's mention plus
    200 of the partner's (seed 6). Step 7: all up to 450, else up to 300 not on the partner's mention and 150 on it,
    each drawn at random (seed 7). Each stratum is weighted by its size over its sample."""
    cap, n_part, n_other, seed = STAGE["sample"]
    if len(recs) <= cap:
        return recs, {}
    by = {m: [r for r in recs if (r["mention_is_partner"]) == m] for m in (True, False)}
    rng = random.Random(seed)
    pick_p = rng.sample(by[True], min(n_part, len(by[True])))
    pick_o = by[False] if n_other is None else rng.sample(by[False], min(n_other, len(by[False])))
    w = {r["cid"]: len(by[True]) / len(pick_p) for r in pick_p}
    w.update({r["cid"]: len(by[False]) / len(pick_o) for r in pick_o})
    return pick_o + pick_p, w


def context(m, k=4):
    out, p = [], m["prev"]
    while p is not None and len(out) < k:
        if p["text"]:
            out.append(line(p))
        p = p["prev"]
    return list(reversed(out))


def recall_messages(keep):
    """Step 7's recall sample: messages drawn at random (registered size and seed), labelled for the facts about people
    they tell before the descriptions are seen."""
    n, seed = STAGE["recall"]
    return random.Random(seed).sample(sorted(m["id"] for m in keep), n)


def recall_export(keep, work, log):
    by = {m["id"]: m for m in keep}
    with open(os.path.join(work, "recall_to_label.jsonl"), "w", encoding="utf-8") as f:
        for mid in recall_messages(keep):
            m = by[mid]
            f.write(json.dumps({"mid": mid, "owner": m["speaker"], "partner": m["partner"], "before": context(m),
                                "message": line(m)}, ensure_ascii=False) + "\n")
    log(f"recall export: {STAGE['recall'][0]} messages; file in WORK")


def export(recs, ents, keep, J, work, log):
    by = {m["id"]: m for m in keep}
    todo, w = sample(recs)
    rec_ids = set(recall_messages(keep)) if STAGE.get("recall") else set()
    if rec_ids:                                         # the system's claims on the recall messages, for matching
        with open(os.path.join(work, "recall_claims.jsonl"), "w", encoding="utf-8") as f:
            for r in recs:
                if r["id"] in rec_ids:
                    f.write(json.dumps({"cid": r["cid"], "mid": r["id"], "claim": f"{r['label']}: {r['value']}",
                                        "entity": r["entity"]}, ensure_ascii=False) + "\n")
    rng = random.Random(61)
    order = list(todo)
    rng.shuffle(order)                                  # judged in random order, not grouped by filing
    with open(os.path.join(work, "claims_to_label.jsonl"), "w", encoding="utf-8") as f:
        for r in order:
            m = by[r["id"]]
            f.write(json.dumps({"cid": r["cid"], "owner": r["owner"], "partner": m["partner"], "before": context(m),
                                "message": line(m), "claim": f"{r['label']}: {r['value']}"}, ensure_ascii=False) + "\n")
    need = {r["cid"] for r in todo} | {r["cid"] for r in recs if r["id"] in rec_ids}
    held = Counter(r["entity"] for r in recs if r["cid"] in need)
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
    facts_p, match_p = os.path.join(work, "labels_recall_facts.json"), os.path.join(work, "labels_recall_match.json")
    if STAGE.get("recall") and os.path.exists(facts_p) and os.path.exists(match_p):
        # step 7: recall = facts about people in the recall sample (labelled before the descriptions were seen) that
        # the system filed under the right person; a fact counts once, through the claim matched to it, or MISS
        facts, match = json.load(open(facts_p)), json.load(open(match_p))
        rmap = {r["cid"]: r for r in recs}
        got = 0
        for fid, fct in facts.items():
            cid = match.get(fid, "MISS")
            if cid != "MISS" and cid in rmap:
                want = {x for x in fct["person"].split("|") if x != "OWNER"}
                got += bool(want & set(ment.get(rmap[cid]["entity"], "UNKNOWN").split("|")))
        res["recall_facts"] = len(facts)
        res["recall_pct"] = pct(got, len(facts))
        wp, rc = res["wrong_person_pct"], res["recall_pct"]
        res["verdict"] = ("PASS" if wp <= 1.5 and rc >= 70 else "STOP" if wp > 3 or rc < 50 else "BETWEEN")
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
    ap.add_argument("--stage", choices=sorted(STAGES), default="step6")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--recall-export", action="store_true", help="step 7: the recall sample, before describing")
    a = ap.parse_args()
    global STAGE, OUT
    STAGE = STAGES[a.stage]
    OUT = os.path.join(TE.OUT, STAGE["out"])
    msgs = load(a.data)
    J, keep = journals(msgs)
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} {a.stage}: REALTALK chats {STAGE['chats']}: {len(keep)} messages with text "
        f"(of {len(msgs)}); journals: " + ", ".join(f"{o} {len(W.notes)}" for o, W in J.items()))
    if a.dry:
        print(prompt(keep[40]))
        return
    if a.recall_export:
        os.makedirs(a.work, exist_ok=True)
        recall_export(keep, a.work, log)
        return
    assert not os.path.abspath(a.work).startswith(os.path.abspath(MI.ROOT)), "WORK must be outside the repository"
    os.makedirs(a.work, exist_ok=True)
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        describe(llm, keep, log, a.workers, a.work)
    path = os.path.join(a.work, STAGE["desc"])
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
