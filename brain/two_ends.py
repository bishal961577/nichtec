"""The "two ends" check (reports/Two ends check 2026.md): the first step of the breakthrough panel's plan.

Reading end: can Qwen3.5-4B, thinking off, answer test 2's dev hard questions from pages where code has already bound
names, current values and links (V0 plain, V1 names, V2 + "true now" header, V3 + links)?
Writing end: can it describe single messages (previous message as context, no memory) with every claim attached to
the right mention, relations the right way round, lures tagged and definitions captured?

    python brain/two_ends.py --dry                               # print sample prompts, no model call
    python brain/two_ends.py --base-url URL/v1 --part reading
    python brain/two_ends.py --base-url URL/v1 --part writing
"""
import argparse
import json
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402
import order_gap as G  # noqa: E402

ROOT = MI.ROOT
OUT = os.path.join(ROOT, "results", "two_ends")
DEV_LIMIT = 1500
DEFS = ("relation to me", "known from", "nickname")
BOOKKEEPING = ("two", "retract", "role", "event_last", "nick")


def dev_world():
    """Test 2's dev world, first 1,500 messages, exactly as the dev run (messy_index.main with --dev-limit 1500)."""
    W = MW.World("dev", seed=101)
    keep = {n["id"] for n in W.notes[:DEV_LIMIT]}
    W.notes = W.notes[:DEV_LIMIT]
    W.deleted = [d for d in W.deleted if d[2] in keep]
    W.deleted_ids = {d[2] for d in W.deleted}
    W.questions = [q for q in W.questions if all(n in keep for n in q["notes"]) and q["notes"]]
    return W


# ============================================================================ reading end: pages bound by code
def full(W, pid):
    return W.people[pid]["full"]


class Truth:
    """What a perfect binding engine would know from all (non-deleted) messages so far."""

    def __init__(self, W):
        self.W = W
        self.slots = defaultdict(list)            # (pid, slot) -> [(date, value)]
        self.defs = defaultdict(dict)             # pid -> {"relation to me": .., "known from": .., "nickname": ..}
        self.events = defaultdict(list)           # pid -> [(date, activity)]
        self.links = []                           # (a, rel, b, date)
        for n in W.notes:
            if n["id"] in W.deleted_ids:
                continue
            for pid, kind, value in n["facts"]:
                if kind in MW.SLOTS:
                    self.slots[(pid, kind)].append((n["date"], value))
                elif kind in DEFS:
                    self.defs[pid][kind] = value
                elif kind == "event":
                    self.events[pid].append((value[1], value[0]))
                elif kind in MW.RELS:
                    self.links.append((pid, kind, value, n["date"]))
        self.end = W.notes[-1]["date"]

    def header(self, pid, with_links):
        W, p = self.W, self.W.people[pid]
        bits = []
        d = self.defs.get(pid, {})
        if d.get("nickname"):
            bits.append(f'also called "{d["nickname"]}"')
        role = d.get("relation to me")
        if p.get("role_from") and p["role_from"].isoformat() <= self.end:
            role = f'{p["role"]} (since {p["role_from"].isoformat()})'
        if p.get("role_until") and p["role_until"].isoformat() <= self.end:
            role = f'old {p["role"]} (until {p["role_until"].isoformat()})'
        if role:
            bits.append(f"my {role}")
        if d.get("known from"):
            bits.append(f'known from {d["known from"]}')
        now = []
        for slot in MW.SLOTS:
            hist = self.slots.get((pid, slot))
            if not hist:
                continue
            date, cur = hist[-1]
            earlier = list(dict.fromkeys(v for _, v in hist[:-1] if v is not None and v != cur))
            if cur is None:
                was = f"; before: {', '.join(earlier)}" if earlier else ""
                now.append(f"{slot}: none any more (stopped {date}{was})")
            else:
                since = next(dt for dt, v in reversed(hist) if v == cur) if cur else date
                first = next(dt for dt, v in hist if v == cur)
                start = first if all(v == cur for _, v in hist[[dt for dt, _ in hist].index(first):]) else since
                was = f"; before: {', '.join(earlier)}" if earlier else ""
                now.append(f"{slot}: {cur} (since {start}{was})")
        line = f"- {full(W, pid)}" + (f" ({'; '.join(bits)})" if bits else "")
        if now:
            line += ". Now: " + "; ".join(now)
        ev = sorted(self.events.get(pid, []))
        if ev:
            line += ". Times with me: " + "; ".join(f"{dt} {act}" for dt, act in ev)
        if with_links:
            ls = []
            for a, rel, b, dt in self.links:
                sym = MW.RELS[rel][0]
                if pid == a:
                    ls.append(f"{full(W, a)}'s {rel} is {full(W, b)}" if sym else f"{full(W, a)} is {full(W, b)}'s {rel}")
                elif pid == b:
                    ls.append(f"{full(W, b)}'s {rel} is {full(W, a)}")
            if ls:
                line += ". Links: " + "; ".join(dict.fromkeys(ls))
        return line + "."


def bound_prompt(W, T, ids, q, version):
    """V0 plain pages; V1 + names; V2 + "true now" header; V3 + links."""
    by = {n["id"]: n for n in W.notes}
    shown = [i for i in sorted(ids) if i in by and i not in W.deleted_ids]
    if version == "V0":
        return MI.answer_prompt(W, shown, q)
    lines, people = [], []
    for i in shown:
        n = by[i]
        names = []
        for surf, pid, style in n["mentions"]:
            names.append(f"{full(W, pid)}, as '{surf}'")
            if pid not in people:
                people.append(pid)
        tag = f" [mentions: {'; '.join(names)}]" if names else ""
        lines.append(f"[{MI.dshow(n['date'])}] {n['text']}{tag}")
    block = "\n".join(lines)
    if version in ("V2", "V3"):
        head = "\n".join(T.header(pid, with_links=(version == "V3")) for pid in people)
        block = ("What the notebook's index says about the people in these notes (worked out from all notes so far):\n"
                 f"{head}\n\nThe notes:\n{block}")
    return MI.ANSWER.format(notes=block or "(no notes found)", today=MI.TODAY, q=q["q"])


def run_reading(llm, W, log, workers):
    T = Truth(W)
    qs = [q for q in W.questions if q["kind"] in MW.HARD]
    path = os.path.join(OUT, "reading.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[r["key"]] = r
    f = open(path, "a", encoding="utf-8")

    def one(args):
        q, v = args
        key = f"{v}|{q['qid']}"
        if key in done:
            return done[key]
        r = MI.ask(llm, bound_prompt(W, T, MI.pages(W, q), q, v), MI.NOTHINK, False,
                   zlib.crc32(key.encode()) % 2**31)
        rec = {"key": key, "version": v, "qid": q["qid"], "kind": q["kind"], "answer": r["text"],
               "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"], "cut": r["cut"]}
        rec.update(MI.score(W, q, r["text"]))
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        f.flush()
        return rec
    for v in ("V0", "V1", "V2", "V3"):
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            recs = list(ex.map(one, [(q, v) for q in qs]))
        for r in recs:
            done[r["key"]] = r
        log(f"reading {v}: {len(recs)} answers, {100 * np.mean([r['correct'] for r in recs]):.1f}% right, "
            f"median {np.median([r['completion_tokens'] for r in recs]):.0f} tokens, {(time.time() - t0) / 60:.1f} min")
    return [r for r in done.values()]


def summarize_reading(recs):
    L = ["## Reading end (288 dev hard questions, thinking off)", "",
         "| Version | Hard set | " + " | ".join(BOOKKEEPING) + " | desc | who | median tokens | cut |",
         "|---|---|" + "---|" * len(BOOKKEEPING) + "---|---|---|---|"]
    res = {}
    for v in ("V0", "V1", "V2", "V3"):
        rs = [r for r in recs if r["version"] == v]
        if not rs:
            continue
        acc = lambda ks: 100 * np.mean([r["correct"] for r in rs if r["kind"] in ks]) if any(
            r["kind"] in ks for r in rs) else float("nan")
        row = {"hard": acc(MW.HARD), **{k: acc([k]) for k in BOOKKEEPING + ("desc", "who")},
               "median_tokens": float(np.median([r["completion_tokens"] for r in rs])),
               "cut": int(sum(r["cut"] for r in rs))}
        res[v] = row
        L.append(f"| {v} | {row['hard']:.1f} | " + " | ".join(f"{row[k]:.1f}" for k in BOOKKEEPING) +
                 f" | {row['desc']:.1f} | {row['who']:.1f} | {row['median_tokens']:.0f} | {row['cut']} |")
    v3 = res.get("V3")
    if v3:
        ok = v3["hard"] >= 88 and all(v3[k] >= 75 for k in BOOKKEEPING) and v3["median_tokens"] <= 64
        verdict = "PASS" if ok else ("STOP" if v3["hard"] < 80 else "BETWEEN")
        L += ["", f"**Reading-end verdict (V3, as registered): {verdict}** (pass: hard >= 88, each of "
              f"{', '.join(BOOKKEEPING)} >= 75, median <= 64 tokens; stop: hard < 80)"]
    return L, res


# ============================================================================ writing end: the memory-blind describer
MENTION_KINDS = ["full name", "first name", "nickname", "my relation", "first name with group", "through someone",
                 "pronoun"]
STATUSES = ["true now", "no longer", "correction", "plan or wish", "tried or temporary", "past", "about someone else"]
SCHEMA = {
    "type": "object",
    "properties": {
        "mentions": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "kind": {"type": "string", "enum": MENTION_KINDS},
            "name": {"type": "string"}, "through": {"type": "string"}, "relation": {"type": "string"},
            "previous": {"type": "string"}},
            "required": ["text", "kind", "name", "through", "relation", "previous"], "additionalProperties": False}},
        "claims": {"type": "array", "items": {"type": "object", "properties": {
            "about": {"type": "integer"}, "label": {"type": "string"}, "value": {"type": "string"},
            "status": {"type": "string", "enum": STATUSES}},
            "required": ["about", "label", "value", "status"], "additionalProperties": False}},
        "links": {"type": "array", "items": {"type": "object", "properties": {
            "who": {"type": "integer"}, "is_the": {"type": "string"}, "of": {"type": "integer"}},
            "required": ["who", "is_the", "of"], "additionalProperties": False}},
        "told_by": {"type": "integer"}},
    "required": ["mentions", "claims", "links", "told_by"], "additionalProperties": False}

DESCRIBE = """You read one message from a person's private notebook and describe it, so that a program can later work out \
who is who. Do not guess who anyone is: write each person exactly as the message names them. You see no other notes.

Return JSON:
- "mentions": every person the message refers to, other than me (the writer), in order. For each:
  "text": the exact words used ("Ugo Mensah", "Ana", "my sister", "my old boss", "Mira from work", "Kofi's flatmate", "she");
  "kind": one of "full name", "first name", "nickname", "my relation", "first name with group", "through someone", "pronoun";
  "name": the person's own name as written, else "";
  "through": for "through someone", the words for the other person (for "Kofi's flatmate": "Kofi"), else "";
  "relation": the relation or group word, if any ("sister" for "my sister", "flatmate" for "Kofi's flatmate", "work" for \
"Mira from work"), else "";
  "previous": for a pronoun that points to someone in the previous message, the words naming them there, else "".
  When someone is named through another person, list both: "Kofi's flatmate" and "Kofi".
- "claims": what the message says about a mentioned person. For each:
  "about": the number of the mention it is about (0 for the first);
  "label": what it is about, in one to three words;
  "value": the thing itself, in a word or two;
  "status": "true now"; "no longer" (stopped, gave up, got rid of); "correction" (corrects an earlier note); "plan or wish"; \
"tried or temporary" (tried once, rented, borrowed, test-drove); "past" (used to, grew up in); "about someone else" (their \
unnamed relative).
  How I know someone is a claim too: label "relation to me" (value "sister"), "known from" (value "choir"), or "nickname" \
(value the nickname). Facts about me, the writer, are not claims: a claim must be about a mention.
- One person, one mention:
  - an introduction is one mention, the name, with the relation in "relation": "my dad Chloe Constantin", "my brother, \
Gaia Okafor", "Teo Rojas, a friend of a friend", "Bodhi Sithole is a colleague of mine";
  - a first name with a group is one mention: "Rosalind from work", "Chloe from down the street";
  - "Uma from football's kid" is someone else, named through Uma: a "through someone" mention, not Uma;
  - a relation between two people always goes in "links", never in "claims".
- Visiting, a trip, or something only for this week is "tried or temporary".
- Every message that says something about a person gets at least one claim. "He adopted a cat" is a mention "He" \
(pronoun) and a claim about it. Something said in passing counts: "Dev's new bike is red" tells you Dev has a bike.
- Describe only the message, never the previous message: the previous message is there only to tell you who "he", \
"she" or "they" is.
- "links": a relation between two mentioned people, as "who" is_the "<relation>" "of" whom: "who" (mention number), \
"is_the" (one word), "of" (mention number). "Ana rents from Kofi": Kofi is the landlord of Ana, so who = Kofi, is_the = \
"landlord", of = Ana. "Ana's mentor is Kofi": who = Kofi, is_the = "mentor", of = Ana.
- "told_by": the number of the mention who told me this, if the message passes on what someone said, else -1.

Example (made-up people). Previous message: [2026-03-02 Mon] Had tea with Tessa Brandt.
Message: [2026-03-03 Tue] Jonas says Tessa's flatmate quit law school and studies music now.
{{"mentions": [{{"text": "Jonas", "kind": "first name", "name": "Jonas", "through": "", "relation": "", "previous": ""}}, \
{{"text": "Tessa's flatmate", "kind": "through someone", "name": "", "through": "Tessa", "relation": "flatmate", \
"previous": ""}}, {{"text": "Tessa", "kind": "first name", "name": "Tessa", "through": "", "relation": "", "previous": ""}}], \
"claims": [{{"about": 1, "label": "studies", "value": "law", "status": "no longer"}}, {{"about": 1, "label": "studies", \
"value": "music", "status": "true now"}}], "links": [{{"who": 1, "is_the": "flatmate", "of": 2}}], "told_by": 0}}

Second example. Previous message: [2026-04-01 Wed] Busy day.
Message: [2026-04-02 Thu] My niece Lena Park, a friend of a friend's daughter, rents from Omar from work.
{{"mentions": [{{"text": "My niece Lena Park", "kind": "full name", "name": "Lena Park", "through": "", "relation": "niece", \
"previous": ""}}, {{"text": "Omar from work", "kind": "first name with group", "name": "Omar", "through": "", "relation": \
"work", "previous": ""}}], "claims": [{{"about": 0, "label": "relation to me", "value": "niece", "status": "true now"}}, \
{{"about": 1, "label": "known from", "value": "work", "status": "true now"}}], "links": [{{"who": 1, "is_the": "landlord", \
"of": 0}}], "told_by": -1}}

Previous message: {prev}
Message: {msg}"""

DESCRIBE_KW = {"max_tokens": 700, "temperature": 0,
               "response_format": {"type": "json_schema", "json_schema": {"name": "description", "schema": SCHEMA}}}


def is_hard(n):
    return bool(n.get("confusable")) or any(m[2] in ("via", "source", "pronoun_prev") for m in n["mentions"])


def sample_messages(W):
    rng = random.Random(20261116)
    notes = W.notes
    slot_fact = lambda n: any(k in MW.SLOTS for _, k, _ in n["facts"])
    pick, chosen = [], set()

    def take(pool, k):
        pool = [n for n in pool if n["id"] not in chosen and n["id"] > 0]
        for n in rng.sample(pool, min(k, len(pool))):
            pick.append(n)
            chosen.add(n["id"])
    take([n for n in notes if slot_fact(n) and is_hard(n)], 100)
    take([n for n in notes if n.get("lure")], 60)
    take([n for n in notes if any(k in DEFS for _, k, _ in n["facts"])], 60)
    take([n for n in notes if n["type"] == "rel"], 40)
    take([n for n in notes if n["facts"]], 300 - len(pick))
    return sorted(pick, key=lambda n: n["id"])


def norm_t(s):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", s.lower().replace("'s", " s")).split() if w]


def match_mention(text, truth):
    """The true mention (surface, pid, style) whose words best match the describer's mention text, or None."""
    a = set(norm_t(text))
    best, score = None, 0.0
    for m in truth:
        b = set(norm_t(m[0]))
        if not a or not b:
            continue
        j = len(a & b) / len(a | b)
        if j > score:
            best, score = m, j
    return best if score >= 0.34 else None


def val_hit(slot, true_value, text):
    if true_value is None:
        return False
    keys = MI.answer_keys(true_value, MW.value_keys(slot, true_value)) if slot in MW.SLOTS else [str(true_value)]
    return any(MI.has_key(text, k) for k in keys)


def score_description(W, n, d):
    """Per-message scores against the truth. d = parsed JSON (or None when unparseable)."""
    out = {"facts": [], "defs": [], "rels": [], "lure_ok": None}
    if d is None:
        d = {"mentions": [], "claims": [], "links": [], "told_by": -1}
    ms = d.get("mentions", [])
    mpid = [(match_mention(m.get("text", ""), n["mentions"]) or (None, None, None))[1] for m in ms]

    def about(c):
        i = c.get("about", -1)
        return mpid[i] if isinstance(i, int) and 0 <= i < len(mpid) else None
    claims = d.get("claims", [])
    for pid, kind, value in n["facts"]:
        if kind in MW.SLOTS:
            if value is None:
                cand = [c for c in claims if c.get("status") == "no longer"]
            else:
                cand = [c for c in claims if val_hit(kind, value, c.get("value", "") + " " + c.get("label", ""))]
            right = any(about(c) == pid for c in cand)
            out["facts"].append({"pid": pid, "slot": kind, "recalled": bool(cand), "right": right,
                                 "hard": is_hard(n)})
        elif kind in DEFS:
            v = str(value).lower()
            hit = any(about(c) == pid and v in (c.get("value", "") + " " + c.get("label", "")).lower() for c in claims)
            hit = hit or any(mpid[i] == pid and v in (m.get("relation", "") + " " + m.get("text", "")).lower()
                             for i, m in enumerate(ms))
            out["defs"].append({"kind": kind, "right": hit})
        elif kind in MW.RELS and not MW.RELS[kind][0]:            # one-way: from = pid ("pid is value's <rel>")
            fw = bw = False
            for lk in d.get("links", []):
                i, j = lk.get("from", -1), lk.get("to", -1)
                if not (isinstance(i, int) and isinstance(j, int) and 0 <= i < len(mpid) and 0 <= j < len(mpid)):
                    continue
                if kind[:4] not in lk.get("relation", "").lower() and not (kind == "landlord" and "rent" in
                                                                          lk.get("relation", "").lower()):
                    continue
                fw = fw or (mpid[i] == pid and mpid[j] == value)
                bw = bw or (mpid[i] == value and mpid[j] == pid)
            out["rels"].append({"rel": kind, "right": fw and not bw, "wrong": bw and not fw, "captured": fw or bw})
    if n.get("lure"):
        slot, value = n["lure"]
        out["lure_ok"] = not any(c.get("status") == "true now" and val_hit(slot, value, c.get("value", ""))
                                 for c in claims)
    return out


# ---------------------------------------------------------------------------- corrected scorer (added after run 1)
# Four scorer false negatives found by reading run 1's failures (reports/Two ends check 2026.md, results). The registered
# scorer above is kept and reported unchanged; this one is reported beside it. Self-checks re-run for both.
ARTICLES = re.compile(r"^(a|an|the|my|some)\s+")
STOPW = {"a", "an", "the", "my", "of", "from", "at", "in", "on", "our", "someone", "one", "mine", "to", "with"}
# the generator's own surface phrases for each "known from" label (messy_world.CIRCLES and the party introductions)
DEF_WORDS = {c: {w for ph in phrases for w in norm_t(ph) if w not in STOPW} | {c}
             for c, phrases in MW.CIRCLES.items()}
DEF_WORDS["a party"] = {"party", "friend of a friend"}


def match_mention_v2(text, truth):
    """As match_mention, but a describer mention whose words all occur in one true mention also matches ("Rosalind" for
    "Rosalind from work"); the best overlap wins, so "Uma" still matches the anchor "Uma from football", not the
    flatmate "Uma from football's colleague"."""
    a = set(norm_t(text))
    best, score = None, 0.0
    for m in truth:
        b = set(norm_t(m[0]))
        if not a or not b:
            continue
        j = len(a & b) / len(a | b)
        if j >= 0.34 or a <= b:
            if j > score:
                best, score = m, j
    return best


def score_description_v2(W, n, d):
    out = {"facts": [], "defs": [], "rels": [], "lure_ok": None}
    if d is None:
        d = {"mentions": [], "claims": [], "links": [], "told_by": -1}
    ms = d.get("mentions", [])
    mpid = [(match_mention_v2(m.get("text", ""), n["mentions"]) or (None, None, None))[1] for m in ms]
    valid = lambda i: isinstance(i, int) and 0 <= i < len(mpid)
    about = lambda c: mpid[c["about"]] if valid(c.get("about", -1)) else None
    claims = d.get("claims", [])
    for pid, kind, value in n["facts"]:
        if kind in MW.SLOTS:
            if value is None:
                cand = [c for c in claims if c.get("status") == "no longer"]
            else:
                cand = [c for c in claims if val_hit(kind, value, c.get("value", "") + " " + c.get("label", ""))]
            out["facts"].append({"pid": pid, "slot": kind, "recalled": bool(cand),
                                 "right": any(about(c) == pid for c in cand), "hard": is_hard(n)})
        elif kind in DEFS:
            v = ARTICLES.sub("", str(value).lower())
            words = DEF_WORDS.get(str(value).lower(), {v}) if kind == "known from" else {v}
            has = lambda t: any(w in t.lower() for w in words)
            hit = any(about(c) == pid and has(c.get("value", "") + " " + c.get("label", "")) for c in claims)
            hit = hit or any(mpid[i] == pid and has(m.get("relation", "") + " " + m.get("text", ""))
                             for i, m in enumerate(ms))
            out["defs"].append({"kind": kind, "right": hit})
        elif kind in MW.RELS and not MW.RELS[kind][0]:
            # a relation may come as a link or as a claim whose value names another mention; its direction is read
            # from its wording: "<x> rents from <y>" / "<x>'s mentor is <y>" make y the landlord / mentor
            pairs = []
            for lk in d.get("links", []):
                if valid(lk.get("from", -1)) and valid(lk.get("to", -1)):
                    pairs.append((mpid[lk["from"]], lk.get("relation", "").lower(), mpid[lk["to"]]))
            for c in claims:
                tgt = [i for i, m in enumerate(ms) if m.get("text") and m["text"].lower() in c.get("value", "").lower()]
                if about(c) is not None and tgt:
                    pairs.append((about(c), c.get("label", "").lower(), mpid[tgt[0]]))
            fw = bw = False
            for x, rel, y in pairs:
                passive = any(w in rel for w in ("rents from", "rent from", "tenant", "mentee", "is mentored", "godchild",
                                                 "mentor is", "landlord is", "godparent is"))
                if kind[:4] not in rel and "rent" not in rel and "mentor" not in rel and "godp" not in rel:
                    continue
                a_, b_ = (y, x) if passive else (x, y)       # a_ is the landlord / mentor / godparent
                fw = fw or (a_ == pid and b_ == value)
                bw = bw or (a_ == value and b_ == pid)
            out["rels"].append({"rel": kind, "right": fw and not bw, "wrong": bw and not fw, "captured": fw or bw})
    if n.get("lure"):
        slot, value = n["lure"]
        out["lure_ok"] = not any(c.get("status") == "true now" and about(c) is not None and
                                 val_hit(slot, value, c.get("value", "")) for c in claims)
    return out


def run_writing(llm, W, log, workers):
    msgs = sample_messages(W)
    by_pos = {n["id"]: i for i, n in enumerate(W.notes)}
    path = os.path.join(OUT, "writing.jsonl")
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    f = open(path, "a", encoding="utf-8")

    def one(n):
        if n["id"] in done:
            return done[n["id"]]
        prev = W.notes[by_pos[n["id"]] - 1]
        prompt = DESCRIBE.format(prev=f"[{MI.dshow(prev['date'])}] {prev['text']}",
                                 msg=f"[{MI.dshow(n['date'])}] {n['text']}")
        r = MI.ask(llm, prompt, DESCRIBE_KW, False, n["id"])
        try:
            d = json.loads(r["text"])
        except (json.JSONDecodeError, TypeError):
            d = None
        if d is not None:
            d["links"] = [{"from": lk.get("who", lk.get("from", -1)), "relation": lk.get("is_the", lk.get("relation", "")),
                           "to": lk.get("of", lk.get("to", -1))} for lk in d.get("links", [])]
        rec = {"id": n["id"], "text": n["text"], "raw": r["text"], "parsed": d is not None, "description": d,
               "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"], "cut": r["cut"],
               "score": score_description(W, n, d), "score_v2": score_description_v2(W, n, d)}
        f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        f.flush()
        return rec
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        recs = list(ex.map(one, msgs))
    log(f"writing: {len(recs)} messages described, {sum(r['parsed'] for r in recs)} parsed, "
        f"{(time.time() - t0) / 60:.1f} min")
    return recs


def summarize_writing(recs, key="score"):
    facts = [x for r in recs for x in r[key]["facts"]]
    defs = [x for r in recs for x in r[key]["defs"]]
    rels = [x for r in recs for x in r[key]["rels"]]
    lures = [r[key]["lure_ok"] for r in recs if r[key]["lure_ok"] is not None]
    pct = lambda a, b: 100 * a / b if b else float("nan")
    rec_e = [x for x in facts if not x["hard"] and x["recalled"]]
    rec_h = [x for x in facts if x["hard"] and x["recalled"]]
    m = {
        "attached_easy": pct(sum(x["right"] for x in rec_e), len(rec_e)), "n_easy": len(rec_e),
        "attached_hard": pct(sum(x["right"] for x in rec_h), len(rec_h)), "n_hard": len(rec_h),
        "rel_direction": pct(sum(x["right"] for x in rels if x["captured"]), sum(x["captured"] for x in rels)),
        "rel_captured": pct(sum(x["captured"] for x in rels), len(rels)), "n_rel": len(rels),
        "lures_tagged": pct(sum(lures), len(lures)), "n_lures": len(lures),
        "definitions": pct(sum(x["right"] for x in defs), len(defs)), "n_defs": len(defs),
        "fact_recall": pct(sum(x["recalled"] for x in facts), len(facts)), "n_facts": len(facts),
        "parsed": pct(sum(r["parsed"] for r in recs), len(recs)),
        "tokens_per_message": float(np.mean([r["completion_tokens"] for r in recs])),
        "prompt_tokens_per_message": float(np.mean([r["prompt_tokens"] for r in recs])),
    }
    lines = {"attached_easy": 98, "attached_hard": 93, "rel_direction": 95, "lures_tagged": 90, "definitions": 98}
    ok = {k: m[k] >= v for k, v in lines.items()}
    L = [f"## Writing end (300 dev messages, memory-blind, thinking off){' - corrected scorer' if key != 'score' else ''}", "",
         "| Measure | Result | Line |", "|---|---|---|",
         f"| Claims on the right mention, easy | {m['attached_easy']:.1f}% of {m['n_easy']} | >= 98 |",
         f"| Claims on the right mention, hard | {m['attached_hard']:.1f}% of {m['n_hard']} | >= 93 (stop < 85 after a week) |",
         f"| Relation direction (one-way, of captured) | {m['rel_direction']:.1f}% (captured {m['rel_captured']:.1f}% of {m['n_rel']}) | >= 95 |",
         f"| Lures tagged | {m['lures_tagged']:.1f}% of {m['n_lures']} | >= 90 |",
         f"| Definitions captured | {m['definitions']:.1f}% of {m['n_defs']} | >= 98 |",
         f"| Fact recall | {m['fact_recall']:.1f}% of {m['n_facts']} | reported (>= 90 suggested) |",
         f"| Valid JSON | {m['parsed']:.1f}% | - |",
         f"| Tokens per message | {m['tokens_per_message']:.0f} out, {m['prompt_tokens_per_message']:.0f} in | - |",
         "", f"**Writing-end lines met:** {json.dumps(ok)}"]
    return L, m


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--part", choices=["reading", "writing", "both"], default="both")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dry", action="store_true", help="print sample prompts and the message sample; no model call")
    a = ap.parse_args()
    W = dev_world()
    if a.dry:
        T = Truth(W)
        qs = [q for q in W.questions if q["kind"] in MW.HARD]
        print(len(qs), "hard questions;", Counter(q["kind"] for q in qs))
        for kind in ("two", "retract", "who"):
            q = next(q for q in qs if q["kind"] == kind)
            print("=" * 30, kind, q["q"], "| gold", q["gold"])
            print(bound_prompt(W, T, MI.pages(W, q), q, "V3")[:3000])
        msgs = sample_messages(W)
        print("writing sample:", len(msgs), "hard", sum(is_hard(n) for n in msgs), "lures",
              sum(bool(n.get("lure")) for n in msgs), "defs",
              sum(any(k in DEFS for _, k, _ in n["facts"]) for n in msgs), "rel", sum(n["type"] == "rel" for n in msgs))
        return
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "run.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} two ends check, part {a.part}")
    llm = G.Endpoint(a.base_url, a.model, 1)
    L = ["# Two ends check: results (dev world, first 1,500 messages)", "",
         "Pre-registration: reports/Two ends check 2026.md", ""]
    result = {}
    if a.part in ("reading", "both"):
        recs = run_reading(llm, W, log, a.workers)
        lr, result["reading"] = summarize_reading(recs)
        L += lr + [""]
    if a.part in ("writing", "both"):
        recs = run_writing(llm, W, log, a.workers)
        lw, result["writing"] = summarize_writing(recs)
        lw2, result["writing_corrected"] = summarize_writing(recs, "score_v2")
        L += lw + [""] + lw2
    with open(os.path.join(OUT, f"summary_{a.part}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, f"result_{a.part}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    log("\n".join(L))


if __name__ == "__main__":
    main()
