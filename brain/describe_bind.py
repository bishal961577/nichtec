"""Step 2 of the two ends check (reports/Two ends check 2026.md): the whole "describe, don't decide" chain on the dev
world.

1. describe: describer v3 (brain/two_ends.py) on all 1,500 dev messages, memory-blind, thinking off;
2. bind: the panel engineer's resolver (research_notes/Breakthrough panel 2026/engineer_scripts/resolver_sim.py),
   ported to read describer JSON: re-solves who everyone is over the whole set, in date order, until nothing changes;
   an unresolved mention becomes a placeholder person, never a guess;
3. index: bindings -> rows in test 2's format, so its filing, forgetting and deletion measures run unchanged;
4. answer: test 2's reader retrieves; pages are bound by the resolver's view; the 4B answers with thinking off.

    python brain/describe_bind.py --base-url URL/v1          # describe (cached), bind, measure, answer
    python brain/describe_bind.py --no-answer                 # bind and measure filing only (needs cached descriptions)
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
import two_ends as TE  # noqa: E402

OUT = TE.OUT
DESC = os.path.join(OUT, "describe_all.jsonl")
PRON = {"he", "she", "they", "him", "her", "them", "his", "hers", "their"}
STOP = {"from", "the", "my", "a", "an", "at", "in", "of", "down", "our", "one", "someone", "mine", "with", "to", "on"}
# general English, not the generator's lists
ROLE_SYN = {"mom": "mum", "mother": "mum", "mommy": "mum", "father": "dad", "daddy": "dad", "grandmother": "grandma",
            "granny": "grandma", "gran": "grandma", "grandfather": "grandpa", "girlfriend": "partner",
            "boyfriend": "partner", "wife": "partner", "husband": "partner", "spouse": "partner", "bestie": "best friend",
            "bff": "best friend", "roommate": "flatmate", "housemate": "flatmate", "manager": "boss"}
SYM = {"neighbour", "flatmate", "colleague", "cousin", "best friend", "teammate", "bandmate", "friend", "sibling",
       "partner", "classmate"}
REVERSE = {"tenant": "landlord", "renter": "landlord", "rents from": "landlord", "mentee": "mentor",
           "protege": "mentor", "student": "mentor", "mentored by": "mentor", "godchild": "godparent",
           "godson": "godparent", "goddaughter": "godparent"}
FORWARD = {"rents to": "landlord", "rents out": "landlord", "mentoring": "mentor", "mentors": "mentor",
           "is mentoring": "mentor", "godmother": "godparent", "godfather": "godparent", "works with": "colleague",
           "lives next door": "neighbour", "next door": "neighbour", "shares a flat": "flatmate", "neighbor": "neighbour",
           "coworker": "colleague", "co-worker": "colleague", "band": "bandmate", "team": "teammate",
           "best friends": "best friend"}


def gwords(s):
    return frozenset(w for w in re.sub(r"[^a-z ]", " ", s.lower()).split() if w not in STOP and len(w) > 2)


def norm_role(s):
    r = re.sub(r"^(my|the|is my|is the)\s+", "", s.strip().lower())
    r = re.sub(r"^old\s+", "", r)
    return ROLE_SYN.get(r, r)


def norm_rel(s):
    """(noun, reversed): a relation word as written -> a canonical noun, and whether who/of must be swapped."""
    r = s.strip().lower().replace("’", "'")
    r = re.sub(r"^(is the|is a|the|a|is)\s+", "", r)
    r = re.sub(r"\s+(of|to|with)$", "", r) if r not in FORWARD and r not in REVERSE else r
    if r in REVERSE:
        return REVERSE[r], True
    if r in FORWARD:
        return FORWARD[r], False
    if r.endswith("s") and r[:-1] in SYM | {"landlord", "mentor", "godparent"}:
        r = r[:-1]
    return r, False


def cap_name(text):
    """The longest run of capitalised words ("My dad Chloe Constantin" -> "Chloe Constantin")."""
    best, cur = [], []
    for w in re.findall(r"[A-Za-z][\w'-]*", text):
        if w[0].isupper() and w.lower() not in ("my", "the", "a", "met", "had", "got", "having"):
            cur.append(w)
            best = cur if len(cur) > len(best) else best
        else:
            cur = []
    return " ".join(best)


def classify(s):
    """The engineer's parser of a mention's own words, with groups as word sets."""
    s = s.strip()
    low = s.lower()
    if low in PRON:
        return ("pron",)
    if "'s " in s:
        anchor, noun = s.rsplit("'s ", 1)
        return ("via", classify(anchor), norm_rel(noun)[0])
    if low.startswith("my old "):
        return ("role", norm_role(low[7:]), "old")
    if low.startswith("my "):
        return ("role", norm_role(low[3:]), "cur")
    if " from " in s:
        first, ctx = s.split(" from ", 1)
        return ("fctx", first.lower().split()[0], gwords(ctx))
    if len(s.split()) >= 2:
        return ("full", low)
    return ("name", low)


def desc_of(m):
    kind, text = m.get("kind", ""), (m.get("text") or "").strip()
    name, rel, thr = (m.get("name") or "").strip(), (m.get("relation") or "").strip(), (m.get("through") or "").strip()
    low = text.lower()
    if kind == "pronoun" or low in PRON:
        return ("pron",)
    if low.startswith("my ") or name.lower().startswith("my "):        # (dev fix 5) "my best friend" is a role
        return classify(text if low.startswith("my ") else name)
    if kind == "through someone" and thr:
        noun = rel or (text.rsplit("'s ", 1)[-1] if "'s " in text else "")
        return ("via", classify(thr), norm_rel(noun)[0])
    if kind == "my relation":
        r = rel or re.sub(r"^my\s+(old\s+)?", "", low)
        return ("role", norm_role(r), "old" if re.search(r"\bold\b", low) else "cur")
    if kind == "first name with group":
        first = (name or text).split()[0].lower()
        return ("fctx", first, gwords(rel or (text.split(" from ", 1)[-1] if " from " in text else "")))
    if kind == "full name":
        nm = name if len(name.split()) >= 2 else (cap_name(text) or name or text)
        return ("full", nm.lower()) if len(nm.split()) >= 2 else ("name", nm.lower().split()[0] if nm else low)
    if kind in ("first name", "nickname"):
        nm = (name or text).strip().strip('"')
        return ("name", nm.lower().split()[0]) if nm else classify(text)
    return classify(text)


def grounded(val, text):
    """(dev fix 4) A claim's value must be supported by its own message: one of its content words (or a 4-letter
    stem of one) occurs in the message, so a definition carried over from the previous message is dropped."""
    words = [w for w in re.sub(r"[^a-z ]", " ", val.lower()).split() if len(w) > 2 and w not in STOP]
    t = text.lower()
    return not words or any(w in t or (len(w) > 4 and w[:4] in t) for w in words)


# Post-hoc dev change after step 4 (off unless set; every registered number is computed with it off): the describer's
# "previous" may name a mention in the same message, the teller included ("Fadil Morozov says his street ..."), and
# names match as whole words ("he" is not part of "Hedda").
ANTE_V2 = False


def wmatch(t, pw):
    if not t or not pw:
        return False
    if t == pw:
        return True
    if t in PRON or pw in PRON:
        return False
    return bool(re.search(r"\b" + re.escape(t) + r"\b", pw) or re.search(r"\b" + re.escape(pw) + r"\b", t))


REPORTING = re.compile(r"^(told|tells|says|said|heard|hears|reported|reports|mentioned|mentions|according)\b")


def to_parse(W, descs_by_id, ids, keep_defs=True, confirmed=None, binds=None, drop_reporting=False):
    """Describer JSON -> the engineer's parse format: {note id: {date, order, descs, ante, claims, styles, texts}}.

    Step 3 (owner confirms definitions): keep_defs=False drops every definition the describer or the apposition rule
    proposes (roles, nicknames, groups, links; the apposition *binding* stays); confirmed = {note id: [entity-level
    definitions the owner gave at that message]}; binds = {note id: {mention index: entity}} (the owner's answer to
    "who is this?"). See confirmed_claims() in resolve() for the formats."""
    by = {n["id"]: n for n in W.notes}
    P = {}
    prev = None
    for i in ids:
        n = by[i]
        d = descs_by_id.get(i) or {"mentions": [], "claims": [], "links": [], "told_by": -1}
        ms = d.get("mentions", [])
        descs = [desc_of(m) for m in ms]
        tb = d.get("told_by", -1)
        styles = ["source" if k == tb else "" for k in range(len(ms))]
        ante = {}
        prev_ments = [] if prev is None else [(prev["id"], j) for j, dd in enumerate(prev["descs"])
                                                if dd[0] != "pron" and prev["styles"][j] != "source"]
        for k, dd in enumerate(descs):
            if dd[0] != "pron":
                continue
            pw = (ms[k].get("previous") or "").strip().lower()
            here = [j for j, x in enumerate(descs) if x[0] != "pron" and styles[j] != "source"]
            hit = None
            if ANTE_V2 and pw:
                texts_here = [(m.get("text") or "").strip().lower() for m in ms]
                same = next((j for j, x in enumerate(descs) if j != k and x[0] != "pron" and wmatch(texts_here[j], pw)),
                            None)
                if same is not None:
                    ante[k] = ("here", same)
                    continue
                if prev is not None:
                    hit = next(((prev["id"], j) for j, t in enumerate(prev["texts"]) if wmatch(t, pw)), None)
            elif pw and prev is not None:
                for j, t in enumerate(prev["texts"]):
                    if t and (t == pw or pw in t or t in pw):
                        hit = (prev["id"], j)
                        break
            if hit:
                ante[k] = ("prev", hit)
            elif here:
                ante[k] = ("here", here[0])
            elif prev_ments:
                ante[k] = ("prev", prev_ments[0])
        valid = lambda a: isinstance(a, int) and 0 <= a < len(ms)
        # (dev fix 1) apposition: in "my dad Chloe Constantin" / "my sister, Ugo Mensah" the role mention and the full
        # name are one person, whether or not the describer split them; the role is defined on the full name
        alias = {}
        fulls = [j for j, x in enumerate(descs) if x[0] == "full"]
        low = n["text"].lower()
        for j, x in enumerate(descs):
            if x[0] == "role" and len(fulls) == 1:
                last = descs[fulls[0]][1].split()[0]
                if re.search(r"\bmy (old |new )?" + re.escape(ms[j].get("text", "").lower().split()[-1] if ms[j].get("text")
                                                         else x[1]) + r"\b,?\s+" + re.escape(last), low):
                    alias[j] = fulls[0]
                    ante[j] = ("here", fulls[0])
        claims = []
        for j, f_ in alias.items():
            if keep_defs:
                claims.append(("role", f_, descs[j][1], bool(re.search(r"\bnew\b", low))))
        for c in d.get("claims", []):
            a = c.get("about", -1)
            if not valid(a):
                continue
            a = alias.get(a, a)
            lab, val, st = (c.get("label") or "").strip().lower(), (c.get("value") or "").strip(), c.get("status")
            if lab in ("nickname", "known from", "relation to me") and not keep_defs:
                continue
            if drop_reporting and REPORTING.match(lab):        # (step 4 dev change, registered before its run) a
                continue                                       # reporting verb is not a fact about the teller
            if lab in ("nickname", "known from", "relation to me") and not grounded(val, n["text"]):
                continue                                 # definitions only: facts may be paraphrased ("flies planes")
            if lab == "nickname":
                claims.append(("nick", a, val.lower().strip('"').split()[0] if val else ""))
            elif lab == "known from":
                claims.append(("circle", a, gwords(val)))
            elif lab == "relation to me":
                if descs[a][0] == "role":               # (dev fix 2) "my dad" cannot define who "my dad" is
                    continue
                claims.append(("role", a, norm_role(val), bool(re.search(r"\bnew\b", n["text"].lower()))))
            elif st in ("true now", "correction"):
                claims.append(("fact", a, lab, val))
            elif st == "no longer":
                claims.append(("fact", a, lab, "none"))
        for lk in ([] if not keep_defs else d.get("links", [])):
            a, b = lk.get("from", -1), lk.get("to", -1)
            if not (valid(a) and valid(b)) or a == b:
                continue
            a, b = alias.get(a, a), alias.get(b, b)
            noun, rev = norm_rel(lk.get("relation", ""))
            if rev:
                a, b = b, a
            claims.append(("rel", a, noun, b, noun in SYM))
        P[i] = {"date": n["date"], "order": (n["date"], i), "descs": descs, "ante": ante, "claims": claims,
                "styles": styles, "texts": [(m.get("text") or "").strip().lower() for m in ms],
                "confirmed": list((confirmed or {}).get(i, [])), "bind": dict((binds or {}).get(i, {}))}
        prev = {"id": i, "descs": descs, "styles": styles, "texts": P[i]["texts"]}
    return P


def resolve(P, iters=5):
    """The engineer's resolver (re-solve over the whole set, in date order, until nothing changes), with groups as word
    sets. Returns {(note id, mention index): entity name or None}."""
    order = sorted(P, key=lambda i: P[i]["order"])
    first_seen = {}
    for i in order:
        for d in P[i]["descs"]:
            for dd in ([d, d[1]] if d[0] == "via" else [d]):
                if dd[0] == "full":
                    first_seen.setdefault(dd[1], P[i]["date"])
        # step 3: the owner names people by full name, so a confirmed person is known from that message on
        for c in P[i].get("confirmed", ()):
            for e in ((c[1], c[3]) if c[0] == "rel" else (c[1],)):
                first_seen.setdefault(e, P[i]["date"])
        for e in P[i].get("bind", {}).values():
            first_seen.setdefault(e, P[i]["date"])
    first_of = defaultdict(set)
    for e in first_seen:
        first_of[e.split()[0]].add(e)
    bind = {}
    for _ in range(iters):
        nick, circle, roles, relof = defaultdict(list), defaultdict(set), defaultdict(list), defaultdict(list)
        for i in order:
            p = P[i]
            for c in p["claims"]:
                if c[0] == "nick" and bind.get((i, c[1])) and c[2]:
                    nick[c[2]].append((p["date"], bind[(i, c[1])]))
                elif c[0] == "circle" and bind.get((i, c[1])):
                    circle[bind[(i, c[1])]] |= c[2]
                elif c[0] == "role" and bind.get((i, c[1])) and c[2]:
                    roles[c[2]].append((p["date"] if c[3] else "0000", bind[(i, c[1])]))
                elif c[0] == "rel":
                    a, b = bind.get((i, c[1])), bind.get((i, c[3]))
                    if a and b and a != b:
                        relof[(b, c[2])].append((p["date"], a))     # "a is b's noun"
                        if c[4]:
                            relof[(a, c[2])].append((p["date"], b))
            # step 3: definitions the owner confirmed, entity-level (no mention binding needed)
            for c in p.get("confirmed", ()):
                if c[0] == "role":                                  # ("role", entity, role, since)
                    roles[c[2]].append((c[3], c[1]))
                elif c[0] == "nick":                                # ("nick", entity, nickname)
                    nick[c[2]].append((p["date"], c[1]))
                elif c[0] == "circle":                              # ("circle", entity, words)
                    circle[c[1]] |= c[2]
                elif c[0] == "rel":                                 # ("rel", a, noun, b, symmetric): a is b's noun
                    relof[(c[3], c[2])].append((p["date"], c[1]))
                    if c[4]:
                        relof[(c[1], c[2])].append((p["date"], c[3]))
        recent, new = [], {}
        # (dev fix 3) definitions by majority over the whole history, so one misread definition cannot redirect every
        # later mention; a role changes hands only when a message says "new" (my new boss)
        role_base, role_new = {}, defaultdict(list)
        for r, lst in roles.items():
            base = Counter(e for d, e in lst if d == "0000")
            news = sorted((d, e) for d, e in lst if d != "0000")
            if base:
                top = max(base.values())
                role_base[r] = sorted(e for e, c in base.items() if c == top)[0]
            elif news:
                role_base[r] = news[0][1]
            role_new[r] = [(d, e) for d, e in news if e != role_base.get(r)]
        nick_major = {}
        for x, lst in nick.items():
            c = Counter(e for _, e in lst)
            top = max(c.values())
            nick_major[x] = sorted(e for e, k in c.items() if k == top)[0]
        rel_major = {}
        for key, lst in relof.items():
            c = Counter(e for _, e in lst)
            top = max(c.values())
            rel_major[key] = {e for e, k in c.items() if k == top}

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
                if d[1] not in role_base:
                    return None
                holders = [("0000", role_base[d[1]])] + role_new[d[1]]
                cur = [k for k, (dd, e) in enumerate(holders) if dd <= date]
                k = cur[-1] if cur else 0
                if d[2] == "old":
                    k -= 1
                return holders[k][1] if 0 <= k < len(holders) else None
            if t == "name":
                nk = {nick_major[d[1]]} if d[1] in nick_major else set()
                fn = {e for e in first_of.get(d[1], ()) if first_seen[e] <= date}
                cands = nk or fn or set(first_of.get(d[1], ()))
                if len(cands) == 1:
                    return next(iter(cands))
                return rec_pick(cands) if cands else None
            if t == "fctx":
                cands = {e for e in first_of.get(d[1], ()) if d[2] & circle.get(e, set())}
                if len(cands) == 1:
                    return next(iter(cands))
                if not cands:
                    return res(("name", d[1]), date)
                return rec_pick(cands)
            if t == "via":
                a = res(d[1], date)
                if not a:
                    return None
                cands = rel_major.get((a, d[2]), set())
                if len(cands) == 1:
                    return next(iter(cands))
                return rec_pick(cands)
            return None

        for i in order:
            p = P[i]
            for k, d in enumerate(p["descs"]):
                if d[0] == "pron":
                    continue
                new[(i, k)] = res(d, p["date"])
            for k, e in p.get("bind", {}).items():             # step 3: the owner said who this mention is
                new[(i, k)] = e
            for k, a in p["ante"].items():
                new[(i, k)] = new.get((i, a[1])) if a[0] == "here" else (new.get(a[1]) or bind.get(a[1]))
            for k in range(len(p["descs"])):
                if new.get((i, k)) and p["styles"][k] != "source":
                    recent.append(new[(i, k)])
        if new == bind:
            break
        bind = new
    return bind


def placeholder(P, i, k, bind):
    d = P[i]["descs"][k]
    if d[0] == "via":
        a = None
        return f"?{d[2]} of {P[i]['texts'][k] or 'someone'}"
    if d[0] == "role":
        return f"?my {'old ' if d[2] == 'old' else ''}{d[1]}"
    if d[0] in ("name", "fctx"):
        return f"?{d[1]}" + (f" from {' '.join(sorted(d[2]))}" if d[0] == "fctx" else "")
    return f"?{P[i]['texts'][k] or 'someone'} #{i}"


class View:
    """The resolver's view of the journal (all non-deleted messages): index rows in test 2's format, plus a per-entity
    header for bound pages."""

    def __init__(self, W, P, bind):
        self.W, self.P, self.bind = W, P, bind
        self.ix = MI.Index()
        self.pid_of, self.ent_of = {}, {}
        self.facts = defaultdict(list)          # entity -> [(date, label, value, note)]
        self.meta = defaultdict(dict)           # entity -> nickname / role / circle
        self.links = []                          # (a, noun, b, date)
        self.unresolved, self.mentions = 0, 0
        for i in sorted(P, key=lambda x: P[x]["order"]):
            p = P[i]
            ent = []
            for k in range(len(p["descs"])):
                e = bind.get((i, k))
                self.mentions += 1
                if not e:
                    self.unresolved += 1
                    e = placeholder(P, i, k, bind)
                ent.append(e)
            self.ent_of[i] = ent
            used = set()
            for c in p["claims"]:
                if c[0] == "rel":
                    a, b = ent[c[1]], ent[c[3]]
                    self.ix.add(i, self.pid(a), c[2], self.title(b), link=self.pid(b))
                    self.links.append((a, c[2], b, p["date"]))
                    used |= {c[1], c[3]}
                    continue
                e = ent[c[1]]
                used.add(c[1])
                if c[0] == "fact":
                    self.ix.add(i, self.pid(e), c[2] or "note", c[3])
                    self.facts[e].append((p["date"], c[2], c[3], i))
                elif c[0] == "nick" and c[2]:
                    self.ix.add(i, self.pid(e), "nickname", c[2])
                    self.meta[e]["nickname"] = c[2]
                elif c[0] == "role" and c[2]:
                    self.ix.add(i, self.pid(e), "relation to me", c[2])
                    self.meta[e]["role"] = c[2]
                elif c[0] == "circle" and c[2]:
                    self.ix.add(i, self.pid(e), "known from", " ".join(sorted(c[2])))
                    self.meta[e]["circle"] = " ".join(sorted(c[2]))
            for c in p.get("confirmed", ()):                   # step 3: definitions the owner confirmed
                if c[0] == "role":
                    self.ix.add(i, self.pid(c[1]), "relation to me", c[2])
                    self.meta[c[1]]["role"] = c[2]
                elif c[0] == "nick":
                    self.ix.add(i, self.pid(c[1]), "nickname", c[2])
                    self.meta[c[1]]["nickname"] = c[2]
                elif c[0] == "circle" and c[2]:
                    self.ix.add(i, self.pid(c[1]), "known from", " ".join(sorted(c[2])))
                    self.meta[c[1]]["circle"] = " ".join(sorted(c[2]))
                elif c[0] == "rel":
                    self.ix.add(i, self.pid(c[1]), c[2], self.title(c[3]), link=self.pid(c[3]))
                    self.links.append((c[1], c[2], c[3], p["date"]))
            for k, e in enumerate(ent):
                if k not in used and p["styles"][k] != "source" or (k not in used and p["styles"][k] == "source"):
                    if not any(r["note"] == i and r["pid"] == self.pid(e) for r in self.ix.rows[-8:]):
                        self.ix.add(i, self.pid(e), "mentioned", "-")

    @staticmethod
    def title(e):
        return e if e.startswith("?") else " ".join(w.capitalize() for w in e.split())

    def pid(self, e):
        if e not in self.pid_of:
            self.pid_of[e] = self.ix.person(self.title(e))
        return self.pid_of[e]

    def header(self, e):
        bits = []
        m = self.meta.get(e, {})
        if m.get("nickname"):
            bits.append(f'also called "{m["nickname"]}"')
        if m.get("role"):
            bits.append(f"my {m['role']}")
        if m.get("circle"):
            bits.append(f"known from {m['circle']}")
        by_label = defaultdict(list)
        for date, lab, val, _ in sorted(self.facts.get(e, [])):
            by_label[lab].append((date, val))
        now = []
        for lab, hist in by_label.items():
            date, cur = hist[-1]
            earlier = list(dict.fromkeys(v for _, v in hist[:-1] if v != "none" and v != cur))
            was = f"; before: {', '.join(earlier)}" if earlier else ""
            now.append(f"{lab}: none any more (stopped {date}{was})" if cur == "none" else f"{lab}: {cur} (since {date}{was})")
        line = f"- {self.title(e)}" + (f" ({'; '.join(bits)})" if bits else "")
        if now:
            line += ". Now: " + "; ".join(now)
        ls = [f"{self.title(a)} is {self.title(b)}'s {noun}" for a, noun, b, _ in self.links if e in (a, b)]
        if ls:
            line += ". Links: " + "; ".join(dict.fromkeys(ls))
        return line + "."

    def bound_prompt(self, ids, q):
        by = {n["id"]: n for n in self.W.notes}
        shown = [i for i in sorted(ids) if i in by and i not in self.W.deleted_ids]
        lines, people = [], []
        for i in shown:
            names = []
            for k, e in enumerate(self.ent_of.get(i, [])):
                names.append(f"{self.title(e)}, as '{self.P[i]['texts'][k]}'")
                if e not in people:
                    people.append(e)
            tag = f" [mentions: {'; '.join(names)}]" if names else ""
            lines.append(f"[{MI.dshow(by[i]['date'])}] {by[i]['text']}{tag}")
        head = "\n".join(self.header(e) for e in people)
        block = ("What the notebook's index says about the people in these notes (worked out from all notes so far):\n"
                 f"{head}\n\nThe notes:\n" + "\n".join(lines)) if people else "\n".join(lines)
        return MI.ANSWER.format(notes=block or "(no notes found)", today=MI.TODAY, q=q["q"])


# ============================================================================ describe all messages
def describe_all(llm, W, log, workers, out=DESC, reuse_sample=True):
    """Describe every dev message (resumable). reuse_sample: take the 300 already described by the 4B in
    writing.jsonl (step 2); step 4's 27B describes all 1,500 itself, into its own file."""
    cache = {}
    for path in ((os.path.join(OUT, "writing.jsonl"),) if reuse_sample else ()) + (out,):
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                if r.get("description") is not None or "description" in r:
                    cache[r["id"]] = r
    by_pos = {n["id"]: k for k, n in enumerate(W.notes)}
    todo = [n for n in W.notes if n["id"] not in cache]
    f = open(out, "a", encoding="utf-8")

    def one(n):
        prev = W.notes[by_pos[n["id"]] - 1] if by_pos[n["id"]] > 0 else None
        prompt = TE.DESCRIBE.format(prev=f"[{MI.dshow(prev['date'])}] {prev['text']}" if prev else "(none)",
                                    msg=f"[{MI.dshow(n['date'])}] {n['text']}")
        r = MI.ask(llm, prompt, TE.DESCRIBE_KW, False, n["id"])
        try:
            d = json.loads(r["text"])
            d["links"] = [{"from": lk.get("who", -1), "relation": lk.get("is_the", ""), "to": lk.get("of", -1)}
                          for lk in d.get("links", [])]
        except (json.JSONDecodeError, TypeError, AttributeError):
            d = None
        return {"id": n["id"], "description": d, "completion_tokens": r["completion_tokens"],
                "prompt_tokens": r["prompt_tokens"], "cut": r.get("cut", False)}
    t0 = time.time()
    if todo:
        with ThreadPoolExecutor(workers) as ex:
            for rec in ex.map(one, todo):
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
                cache[rec["id"]] = rec
        f.flush()
    log(f"described: {len(cache)} messages ({len(todo)} new, {(time.time() - t0) / 60:.1f} min)")
    return {i: r["description"] for i, r in cache.items()}


def people_bootstrap(qs, a, b, n=2000, seed=7):
    """Hard-set difference a - b (percentage points) with a people-resampled 95% interval."""
    by = defaultdict(list)
    for q in qs:
        by[q["person"]].append(q["qid"])
    keys = list(by)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.choice(len(keys), len(keys))
        ids = [qid for k in pick for qid in by[keys[k]]]
        diffs.append(100 * (np.mean([a[x] for x in ids]) - np.mean([b[x] for x in ids])))
    point = 100 * (np.mean([a[q["qid"]] for q in qs]) - np.mean([b[q["qid"]] for q in qs]))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-answer", action="store_true")
    a = ap.parse_args()
    W = TE.dev_world()
    os.makedirs(OUT, exist_ok=True)
    logf = open(os.path.join(OUT, "chain.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} step 2: the whole chain on dev")
    llm = G.Endpoint(a.base_url, a.model, 1) if a.base_url else None
    if llm:
        descs = describe_all(llm, W, log, a.workers)
    else:
        descs = {}
        for path in (os.path.join(OUT, "writing.jsonl"), DESC):
            if os.path.exists(path):
                for line in open(path, encoding="utf-8"):
                    r = json.loads(line)
                    descs[r["id"]] = r.get("description")
    missing = [n["id"] for n in W.notes if n["id"] not in descs]
    if missing:
        raise SystemExit(f"{len(missing)} messages have no description yet: run with --base-url")
    ids_all = [n["id"] for n in W.notes]
    ids_kept = [i for i in ids_all if i not in W.deleted_ids]
    # bind over the journal without the deleted messages (deletion = recompute)
    P = to_parse(W, descs, ids_kept)
    bind = resolve(P)
    V = View(W, P, bind)
    # checks reported by construction: shuffled input order, and what deleting the 35 messages changes
    keys = list(P)
    random.Random(3).shuffle(keys)
    same_order = resolve({k: P[k] for k in keys}) == bind
    P_all = to_parse(W, descs, ids_all)
    bind_all = resolve(P_all)
    changed = sum(1 for k, e in bind.items() if bind_all.get(k) != e)
    wq = MI.write_quality(W, V.ix)
    who = wq.pop("who")
    emb = MI.Embedder()
    reader = MI.Reader(V.ix, W, emb)
    fg = MI.forget_test(W, V.ix, reader)
    left = MI.deletion_audit(W, V.ix, who)
    res = {"filing": wq, "forget": fg, "rows_left_after_deletion": left, "same_view_shuffled": same_order,
           "bindings_changed_by_deleting_35": changed, "mentions": V.mentions, "unresolved": V.unresolved,
           "unresolved_pct": 100 * V.unresolved / max(1, V.mentions)}
    log("filing: " + json.dumps({k: wq[k] for k in ("wrong_merge_pct", "hard_merge_pct", "facts_filed_pct",
                                                     "filed_right_by_style", "link_precision_pct", "index_people",
                                                     "people_split_into_several_ids")}))
    log(f"forgetting: {json.dumps(fg)}; rows left after deletion: {left}; shuffled order same view: {same_order}; "
        f"bindings changed by deleting the 35 messages: {changed}; unresolved mentions {V.unresolved} of {V.mentions}")
    L = ["# Step 2: the whole chain on the dev world (first 1,500 messages)", "",
         "Pre-registration: reports/Two ends check 2026.md, step 2.", "",
         "| Measure | Chain | Line | B on dev |", "|---|---|---|---|",
         f"| Wrong merges | {wq['wrong_merge_pct']:.2f}% | <= 1.5% | 6.27% |",
         f"| \"X's flatmate\" filed right | {wq['filed_right_by_style'].get('via', float('nan')):.1f}% | >= 75% | 11.1% |",
         f"| Worst-person forgetting | {fg['forget_worst_pct']:.1f}% (average {fg['forget_recall_pct']:.1f}%) | >= 90% | 62.2% |",
         f"| Facts filed right | {wq['facts_filed_pct']:.1f}% | >= 85% | 78.2% |",
         f"| Hard-case merges | {wq['hard_merge_pct']:.1f}% | reported | 33.3% |",
         f"| Link precision | {wq['link_precision_pct']:.1f}% | reported | 57.1% |",
         f"| Mentions left unresolved | {res['unresolved_pct']:.1f}% | reported | - |",
         f"| Filed right by mention style | {json.dumps(wq['filed_right_by_style'])} | reported | - |"]
    if llm and not a.no_answer:
        qs = [q for q in W.questions if q["kind"] in MW.HARD]
        path = os.path.join(OUT, "chain_answers.jsonl")
        done = {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line)
                done[r["qid"]] = r
        f = open(path, "a", encoding="utf-8")

        def one(q):
            if q["qid"] in done:
                return done[q["qid"]]
            c = MI.ask(llm, MI.CUES.format(q=q["q"]), MI.CUE_KW, False, zlib.crc32(q["qid"].encode()) % 2**31)
            ids, targets = reader.retrieve(MI.parse_cues(c["text"]), q["q"])
            r = MI.ask(llm, V.bound_prompt(ids, q), MI.NOTHINK, False, zlib.crc32(f"chain|{q['qid']}".encode()) % 2**31)
            rec = {"qid": q["qid"], "kind": q["kind"], "person": q["person"], "answer": r["text"],
                   "completion_tokens": r["completion_tokens"], "prompt_tokens": r["prompt_tokens"], "cut": r["cut"],
                   "notes": ids, "gold_found": sorted(set(q["notes"]) & set(ids)), "gold_total": len(q["notes"])}
            rec.update(MI.score(W, q, r["text"]))
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            f.flush()
            return rec
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            recs = list(ex.map(one, qs))
        acc = 100 * np.mean([r["correct"] for r in recs])
        med = float(np.median([r["completion_tokens"] for r in recs]))
        log(f"answers: {len(recs)}, {acc:.1f}% right, median {med:.0f} tokens, {(time.time() - t0) / 60:.1f} min")
        b = {}
        for line in open(os.path.join(MI.ROOT, "results", "messy_index_dev_dev1500", "answers.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["arm"] == "B" and r["mode"] == "think":
                b[r["qid"]] = r["correct"]
        cr = {r["qid"]: r["correct"] for r in recs}
        diff = people_bootstrap(qs, cr, b)
        kinds = sorted({q["kind"] for q in qs})
        per = {k: 100 * np.mean([r["correct"] for r in recs if r["kind"] == k]) for k in kinds}
        found = 100 * np.mean([len(r["gold_found"]) == r["gold_total"] for r in recs])
        res.update(answers={"hard": acc, "median_tokens": med, "per_kind": per, "found_all_pct": found,
                            "chain_minus_B_think": diff})
        L += [f"| Hard set, chain, thinking off | {acc:.1f}% | >= 77.3% | B thinking on 72.3%, off 50.3% |",
              f"| Chain minus B thinking on (people-resampled 95% interval) | {diff[0]:+.1f} [{diff[1]:+.1f}, {diff[2]:+.1f}] | interval above 0 | - |",
              f"| Median answer | {med:.0f} tokens | <= 64 | about 4,200 |",
              f"| Found every needed message | {found:.1f}% | reported | 71.7% |",
              f"| By kind | {json.dumps({k: round(v, 1) for k, v in per.items()})} | reported | - |"]
        ok = (wq["wrong_merge_pct"] <= 1.5 and wq["filed_right_by_style"].get("via", 0) >= 75 and
              fg["forget_worst_pct"] >= 90 and wq["facts_filed_pct"] >= 85 and acc >= 77.3 and med <= 64 and diff[1] > 0)
        stop = wq["wrong_merge_pct"] > 3 or acc < 70
        L += ["", f"**Verdict (dev, as registered): {'PASS' if ok else ('STOP' if stop else 'BETWEEN')}**"]
    L += ["", f"Checks by construction: shuffled input order gives the same view: {same_order}; bindings that change when "
              f"the 35 deleted messages are removed: {changed}; rows left carrying a deleted value: {left}."]
    with open(os.path.join(OUT, "chain_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "chain_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("\n".join(L))


if __name__ == "__main__":
    main()
