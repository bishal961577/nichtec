"""Test 1 of the expert panel's plan: can a model at shipping size read a whole notebook from its prompt?

Pre-registered in reports/Notebook reading at shipping size 2026.md (section 3), committed before any model call.

Part A, personal notebook. A fictional world of 1,080 people in 540 groups. Each group is person A (4 facts), A's
relation to person B (one note), and B (4 facts); 135 groups also carry a later note that changes one of A's facts.
4,995 notes in all. Every question is asked with its group inside a pile of about 10, 100, 1,000 or 5,000 notes,
thinking on and off. The level-10 pile holds only the group's own notes: the "selected pages" condition.
  - reword: one fact, asked with no content word in common with its note (150 questions)
  - same:   100 of those facts, asked in the note's own words
  - two:    an attribute of A's relation (A -> B in one note, B's fact in another; 120 questions)
  - upd:    the latest value of a fact a later note changed (60 questions)
Part A2, conflicting world edits. MQuAKE-CF-3k-v2 edits written as notes; 150 single-hop questions and 120 two-hop
cases, asked inside piles of 10, 100, 1,000 and all edits; the same two-hop questions with no notes give the model's
own-fact rate.

  python brain/notebook_read.py --stub                                            # bookkeeping, no model
  python brain/notebook_read.py --base-url URL --model Qwen/Qwen3.5-27B-FP8 --preflight
  python brain/notebook_read.py --base-url URL --model Qwen/Qwen3.5-27B-FP8
The run is resumable: answers are appended to results/notebook_read_<model>/answers.jsonl and skipped on restart.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402  (the endpoint client: retries, auth headers)

ROOT = G.ROOT
SEED = 0
N_GROUPS, N_UPDATE_GROUPS = 540, 135
LEVELS = [10, 100, 1000, 5000]
GROUPS_PER_PILE = {10: 1, 100: 11, 1000: 108, 5000: 540}
N_REWORD, N_SAME, N_TWO, N_UPD = 150, 100, 120, 60
N_W1, N_W2 = 150, 120
W_LEVELS = [10, 100, 1000, "all"]
THINK = {"max_tokens": 6144, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
NOTHINK = {"max_tokens": 48, "temperature": 0}
MQUAKE = "MQuAKE-CF-3k-v2.json"
MQUAKE_URL = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/" + MQUAKE
MQUAKE_SHA256 = "f82091cbb668cef8f2537f79f1768af1840184450b347ff8c344336090ddc71e"
BOOT = 10_000

# Pre-registered lines (section 3 of the report). Decisions use point estimates; intervals are reported.
PASS_REWORD, STOP_REWORD = 85.0, 70.0          # reworded single-hop at 5,000 notes, thinking on (%)
PASS_RATIO, STOP_RATIO = 0.8, 0.5              # two-hop at 5,000 notes / two-hop at 10 notes (same questions)
MIN_TWO_AT_10 = 50.0                           # below this the ratio is not measurable
PRIORITY = ([("personal", "think", n) for n in (5000, 10, 1000, 100)] +
            [("personal", "nothink", n) for n in (5000, 10, 1000, 100)] +
            [("world", "think", n) for n in ("all", 10, 1000, 100)] + [("own", "think", 0)] +
            [("world", "nothink", n) for n in ("all", 10, 1000, 100)] + [("own", "nothink", 0)])

FIRST = """Ana Rui Mira Tomas Lena Omar Ines Kenji Priya Noah Sofia Mateo Yara Felix Amara Jonas Leila Arjun Nadia
Emil Zara Hugo Maya Ivan Chloe Darius Elif Malik Greta Tariq Lucia Sven Aisha Pablo Hana Viktor Nora Kofi Elena
Rafael Ingrid Samir Clara Diego Freya Hassan Julia Oscar Leah Mei Andrej Rosa Idris Alma Teo Selin Marco Ayla Niko
Imani Lars Carmen Yusuf Petra Joao Esme Dalia Anton Sara Ravi Liv Emre Paula Bodhi Vera Mosi Flora Jasper Noor Milan
Astrid Ezra Wanjiru Hiro Beatriz Kwame Signe Farid Agnes Lior Ottilie Sanjay Mirela Tobias Zeynep Akira Rania Gustav
Fiona Dmitri Paloma Ilse Koa Simone Bilal Henrike Thabo Valeria Annika Joaquin Ruth Ade Ksenia Matteo Gideon Halima
Cyril Ottavia Renzo Saoirse Tuan Mariam Florin Neve Bashir Isolde Casimir Jovan Linnea Faisal Tamsin Kaito Odile Bram
Shirin Ludo Wren Ciaran Yuki Marta""".split()
LAST = """Costa Mendes Okafor Velez Lindqvist Haddad Tanaka Novak Ferreira Moreau Kowalski Brennan Achebe Sato Ibrahim
Delgado Larsen Petrov Romano Nakamura Oyelaran Castillo Fischer Duarte Kaplan Ndiaye Bergstrom Varga Quintero
Halvorsen Abara Szabo Rinaldi Mbeki Ortega Johansson Petrakis Albescu Moraes Nyberg Okonkwo Salazar Lindgren Rossi
Takahashi Wojcik Eriksen Mensah Navarro Horvath Kristensen Barbosa Dimitrov Galli Asante Herrera Lund Kovacs Yilmaz
Sorensen Pereira Ivanova Conti Boateng Ramos Holm Nagy Demir Kaur Ricci Owusu Vega Strand Toth Aydin Pinto Volkov
Marchetti Adeyemi Fuentes Farkas Celik Lopes Sokolov Greco Appiah Dahl Molnar Kaya Teixeira Morozov Esposito Danso
Soto Balogh Ozturk Carvalho Lebedev Amoah Rojas Sandberg Kiss Arslan Ribeiro Popescu Fontana Acheampong Cruz Wiklund
Dogan Moreira Ionescu Caruso Frimpong Silva Hedlund Racz Sahin Cardoso Constantin Lombardi Ofori Pena Blom Polat
Gomes Bianchi Antwi Iglesias Forsberg Lakatos Kurt Rocha Matei Colombo""".split()

ATTRS = {
    "job": dict(fact="{p} works as {art} {v}.", same="What does {p} work as?", reword="What is {p}'s profession?",
                two="What does {a}'s {r} work as?", upd="{p} has changed careers and now works as {art} {v}.",
                upd_q="What does {p} work as now?",
                vocab="pharmacist carpenter architect nurse electrician journalist plumber librarian veterinarian "
                      "translator accountant firefighter baker pilot geologist physiotherapist florist mechanic optician "
                      "surveyor tailor butcher midwife locksmith chef lawyer economist beekeeper glassblower "
                      "cartographer".split()),
    "city": dict(fact="{p} lives in {v}.", same="Where does {p} live?", reword="Which city is {p}'s home?",
                 two="Where does {a}'s {r} live?", upd="{p} has moved to {v}.",
                 upd_q="Which city does {p} live in now?",
                 vocab="Lisbon Porto Lyon Leeds Krakow Utrecht Bergen Graz Seville Turin Tampere Ghent Brno Malmo "
                       "Bilbao Cork Aarhus Dresden Bologna Nantes Valencia Gdansk Salzburg Antwerp Edinburgh Zagreb "
                       "Ljubljana Tallinn Riga Vilnius".split()),
    "pet": dict(fact="{p} has a pet {v}.", same="What pet does {p} have?",
                reword="Which animal does {p} look after at home?", two="What pet does {a}'s {r} have?",
                vocab=["parrot", "tortoise", "hamster", "ferret", "rabbit", "goldfish", "iguana", "cockatiel",
                       "chinchilla", "gecko", "hedgehog", "guinea pig", "canary", "axolotl", "beagle", "dalmatian",
                       "siamese cat", "python", "tarantula", "corn snake"]),
    "hobby": dict(fact="In their free time, {p} enjoys {v}.", same="What does {p} enjoy doing in their free time?",
                  reword="What is {p}'s hobby?", two="What does {a}'s {r} enjoy doing in their free time?",
                  vocab=["pottery", "rock climbing", "birdwatching", "chess", "knitting", "surfing", "fencing",
                         "origami", "archery", "kayaking", "calligraphy", "gardening", "juggling", "sailing",
                         "woodcarving", "salsa dancing", "astronomy", "quilting", "fishing", "photography",
                         "skateboarding", "cycling", "embroidery", "orienteering"]),
    "food": dict(fact="{p}'s favourite food is {v}.", same="What is {p}'s favourite food?",
                 reword="Which dish does {p} like eating most?", two="What is the favourite food of {a}'s {r}?",
                 vocab=["lasagne", "ramen", "paella", "falafel", "goulash", "sushi", "pierogi", "moussaka", "risotto",
                        "tacos", "pho", "biryani", "ceviche", "shakshuka", "gnocchi", "bibimbap", "couscous",
                        "empanadas", "borscht", "poutine", "pad thai", "dumplings", "jollof rice", "tagine"]),
    "instrument": dict(fact="{p} plays the {v}.", same="Which instrument does {p} play?",
                       reword="Which musical instrument has {p} mastered?", two="Which instrument does {a}'s {r} play?",
                       vocab="violin cello trumpet accordion harp banjo clarinet oboe flute saxophone ukulele piano "
                             "drums mandolin bassoon harmonica trombone sitar bagpipes xylophone".split()),
    "car": dict(fact="{p} drives {art} {v}.", same="What car does {p} drive?",
                reword="Which vehicle brand does {p} own?", two="What car does {a}'s {r} drive?",
                vocab="Volvo Toyota Skoda Fiat Renault Subaru Mazda Peugeot Kia Hyundai Tesla Audi Honda Nissan "
                      "Citroen Opel Dacia Jeep Saab Lexus".split()),
    "language": dict(fact="{p} is learning {v}.", same="Which language is {p} learning?",
                     reword="What foreign tongue is {p} studying?", two="Which language is {a}'s {r} learning?",
                     vocab="Japanese Portuguese Swahili Finnish Korean Turkish Greek Polish Arabic Hindi Dutch "
                           "Icelandic Hungarian Vietnamese Tagalog Czech Welsh Hebrew Mandarin Italian".split()),
    "allergy": dict(fact="{p} is allergic to {v}.", same="What is {p} allergic to?",
                    reword="Which substance must {p} avoid for medical reasons?", two="What is {a}'s {r} allergic to?",
                    vocab=["peanuts", "shellfish", "penicillin", "pollen", "latex", "bee stings", "gluten", "sesame",
                           "strawberries", "dust mites", "eggs", "soy", "walnuts", "aspirin", "kiwi", "mustard"]),
}
RELATIONS = ["neighbour", "best friend", "manager", "cousin", "flatmate", "landlord", "colleague", "mentor",
             "business partner", "personal trainer"]   # gender-neutral: names are random, so no sister/brother
STOP = set("a an the is are does do what which who where whose of to in their his her has have for at on by with as "
           "from it its be and or now most must after".split())

PERSONAL = ("Below are notes a user has written in their personal notebook, in the order they were written. Later notes "
            "may update earlier ones.\n\nNotes:\n{notes}\n\nQuestion: {q}\nAnswer with the answer only, in a few words.")
WORLD = ("Below are notes about changes in the world, in the order they were written. Treat them as true, even where "
         "they differ from what you know.\n\nNotes:\n{notes}\n\nQuestion: {q}\nAnswer with the answer only, in a few "
         "words.")
BARE = "Question: {q}\nAnswer with the answer only, in a few words."


# ---------------------------------------------------------------- text helpers and scoring
def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def contains(ans, v):
    v = norm(v)
    return bool(v) and f" {v} " in f" {norm(ans)} "


def stems(template):
    words = re.sub(r"\{\w+\}", " ", template.lower().replace("'s", " "))
    return {w[:4] for w in re.findall(r"[a-z]+", words) if w not in STOP}


def art(v):
    return "an" if v[0].lower() in "aeiou" else "a"


def fill(t, **kw):
    if "{art}" in t:
        kw["art"] = art(kw["v"])
    return t.format(**kw)


def prefix_free(names):
    names = sorted(dict.fromkeys(names))
    return [n for n in names if not any(o != n and n.startswith(o) for o in names)]


# ---------------------------------------------------------------- part A: the personal world
def checks():
    """Fails loudly if the world could score a wrong answer as right."""
    for r, a in ATTRS.items():
        vs = [norm(v) for v in a["vocab"]]
        assert len(set(vs)) == len(vs) >= 16, r
        for v in vs:
            assert not any(o != v and f" {v} " in f" {o} " for o in vs), (r, v)
        assert not stems(a["reword"]) & stems(a["fact"]), (r, stems(a["reword"]) & stems(a["fact"]))
        assert stems(a["same"]) & stems(a["fact"]), r


def names():
    vocab_words = {w for a in ATTRS.values() for v in a["vocab"] for w in norm(v).split()}
    vocab_words |= {w for r in RELATIONS for w in r.split()}
    first = [n for n in prefix_free(FIRST) if norm(n) not in vocab_words]
    last = [n for n in prefix_free(LAST) if norm(n) not in vocab_words and n not in first]
    return first, last


def world():
    checks()
    first, last = names()
    rng = random.Random(SEED)
    combos = [(f, l) for f in first for l in last]
    people = rng.sample(combos, 2 * N_GROUPS)
    upd_groups = set(rng.sample(range(N_GROUPS), N_UPDATE_GROUPS))
    groups = []
    for g in range(N_GROUPS):
        (fa, la), (fb, lb) = people[2 * g], people[2 * g + 1]
        if fa == fb or la == lb:                       # keep A and B easy to tell apart inside a group
            j = next(k for k in range(2 * N_GROUPS, len(combos)) if combos[k][0] not in (fa,) and combos[k][1] != la
                     and f"{combos[k][0]} {combos[k][1]}" not in {f"{x} {y}" for x, y in people})
            people.append(combos[j])
            fb, lb = combos[j]
        a, b = f"{fa} {la}", f"{fb} {lb}"
        attrs = {}
        for who in ("a", "b"):
            rels = rng.sample(sorted(ATTRS), 4)
            if who == "a" and g in upd_groups and not {"city", "job"} & set(rels):
                rels[0] = rng.choice(["city", "job"])
            attrs[who] = {r: rng.choice(ATTRS[r]["vocab"]) for r in rels}
        upd = None
        if g in upd_groups:
            r = rng.choice(sorted({"city", "job"} & set(attrs["a"])))
            new = rng.choice([v for v in ATTRS[r]["vocab"] if v != attrs["a"][r]])
            upd = {"rel": r, "old": attrs["a"][r], "new": new}
        groups.append({"gid": g, "a": a, "b": b, "rel": rng.choice(RELATIONS), "attrs": attrs, "upd": upd})
    assert len({p for gr in groups for p in (gr["a"], gr["b"])}) == 2 * N_GROUPS
    return groups


def group_notes(gr):
    notes = []
    for who in ("a", "b"):
        for r, v in gr["attrs"][who].items():
            notes.append({"text": fill(ATTRS[r]["fact"], p=gr[who], v=v), "gid": gr["gid"], "tag": f"{who}.{r}"})
    notes.append({"text": f"{gr['a']}'s {gr['rel']} is {gr['b']}.", "gid": gr["gid"], "tag": "rel"})
    if gr["upd"]:
        u = gr["upd"]
        notes.append({"text": fill(ATTRS[u["rel"]]["upd"], p=gr["a"], v=u["new"]), "gid": gr["gid"], "tag": "upd"})
    return notes


def questions(groups):
    rng = random.Random(SEED + 2)
    Q = []
    for k, g in enumerate(rng.sample(range(N_GROUPS), N_REWORD)):
        gr = groups[g]
        who = rng.choice(["a", "b"])
        rels = sorted(r for r in gr["attrs"][who] if not (who == "a" and gr["upd"] and gr["upd"]["rel"] == r))
        r = rng.choice(rels)
        base = {"gid": g, "rel": r, "gold": gr["attrs"][who][r]}
        Q.append({**base, "qid": f"reword-{k:03d}", "kind": "reword", "q": ATTRS[r]["reword"].format(p=gr[who])})
        if k < N_SAME:
            Q.append({**base, "qid": f"same-{k:03d}", "kind": "same", "q": ATTRS[r]["same"].format(p=gr[who])})
    for k, g in enumerate(rng.sample(range(N_GROUPS), N_TWO)):
        gr = groups[g]
        r = rng.choice(sorted(gr["attrs"]["b"]))
        Q.append({"qid": f"two-{k:03d}", "kind": "two", "gid": g, "rel": r, "gold": gr["attrs"]["b"][r],
                  "q": ATTRS[r]["two"].format(a=gr["a"], r=gr["rel"])})
    upd = sorted(g["gid"] for g in groups if g["upd"])
    for k, g in enumerate(rng.sample(upd, N_UPD)):
        gr, u = groups[g], groups[g]["upd"]
        Q.append({"qid": f"upd-{k:03d}", "kind": "upd", "gid": g, "rel": u["rel"], "gold": u["new"], "old": u["old"],
                  "q": ATTRS[u["rel"]]["upd_q"].format(p=gr["a"])})
    return Q


def place_updates(notes, rng):
    """Every update note goes to a random position after the note it changes."""
    for u in [n for n in notes if n["tag"] == "upd"]:
        notes.remove(u)
        i = next(k for k, n in enumerate(notes) if n["gid"] == u["gid"] and n["tag"] == "a." + ("city" if "moved" in
                 u["text"] else "job"))
        notes.insert(rng.randint(i + 1, len(notes)), u)
    return notes


def personal_piles(groups):
    """For each level: the piles (note lists) and the pile index of every group."""
    order = list(range(N_GROUPS))
    random.Random(SEED + 1).shuffle(order)
    out = {}
    for n in LEVELS:
        chunks = np.array_split(np.array(order), N_GROUPS // GROUPS_PER_PILE[n])
        piles, where = [], {}
        for ci, chunk in enumerate(chunks):
            notes = [x for g in chunk for x in group_notes(groups[int(g)])]
            rng = random.Random(zlib.crc32(f"personal|{n}|{ci}".encode()))
            rng.shuffle(notes)
            piles.append(place_updates(notes, rng))
            for g in chunk:
                where[int(g)] = ci
        out[n] = {"piles": piles, "where": where}
    return out


# ---------------------------------------------------------------- part A2: conflicting world edits
def fetch_mquake():
    path = os.path.join(ROOT, "data", "mquake", MQUAKE)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(MQUAKE_URL, path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if sha != MQUAKE_SHA256:
        raise SystemExit(f"{path}: sha256 {sha} is not the pre-registered {MQUAKE_SHA256}")
    return json.load(open(path, encoding="utf-8"))


def aliases(main, extra):
    return [a for a in [main] + list(extra) if norm(a)]


def world_items():
    cases = fetch_mquake()
    edits, clash = {}, 0
    for c in cases:
        for rw in c["requested_rewrite"]:
            key = (rw["subject"], rw["relation_id"])
            text = rw["prompt"].format(rw["subject"]) + " " + rw["target_new"]["str"] + "."
            if key in edits and edits[key]["text"] != text:
                clash += 1
                continue
            if key not in edits:
                new = next((h for h in c["new_single_hops"] if h["question"] == rw["question"]), None)
                old = next((h for h in c["single_hops"] if h["question"] == rw["question"]), None)
                edits[key] = {"text": text, "q": rw["question"],
                              "new": aliases(rw["target_new"]["str"], new["answer_alias"] if new else []),
                              "old": aliases(rw["target_true"]["str"], old["answer_alias"] if old else [])}
    keys = sorted(edits)
    disjoint = lambda x, y: not {norm(a) for a in x} & {norm(a) for a in y}  # noqa: E731
    rng = random.Random(SEED + 3)
    one = [k for k in keys if disjoint(edits[k]["new"], edits[k]["old"])]
    Q = [{"qid": f"w1-{i:03d}", "kind": "w1", "q": edits[k]["q"], "new": edits[k]["new"], "old": edits[k]["old"],
          "need": [edits[k]["text"]]} for i, k in enumerate(rng.sample(one, N_W1))]
    two = [c for c in cases if len(c["single_hops"]) == 2 and disjoint(
        aliases(c["new_answer"], c["new_answer_alias"]), aliases(c["answer"], c["answer_alias"]))]
    for i, c in enumerate(rng.sample(two, N_W2)):
        need = sorted({edits[(rw["subject"], rw["relation_id"])]["text"] for rw in c["requested_rewrite"]})
        Q.append({"qid": f"w2-{i:03d}", "kind": "w2", "q": c["questions"][0], "case_id": c["case_id"],
                  "new": aliases(c["new_answer"], c["new_answer_alias"]), "old": aliases(c["answer"], c["answer_alias"]),
                  "need": need})
    return [edits[k]["text"] for k in keys], Q, clash


def world_piles(texts, Q):
    out = {}
    rng = random.Random(SEED + 4)
    allp = list(texts)
    rng.shuffle(allp)
    out["all"] = {"piles": [allp], "where": {q["qid"]: 0 for q in Q}}
    order = list(Q)
    rng.shuffle(order)
    for n in (10, 100, 1000):
        piles, where, batch, need = [], {}, [], set()

        def close():
            r = random.Random(zlib.crc32(f"world|{n}|{len(piles)}".encode()))
            fill_ = r.sample([t for t in texts if t not in need], n - len(need))
            notes = list(need) + fill_
            r.shuffle(notes)
            piles.append(notes)
            for q in batch:
                where[q["qid"]] = len(piles) - 1

        for q in order:
            if batch and len(need | set(q["need"])) > n // 2:
                close()
                batch, need = [], set()
            batch.append(q)
            need |= set(q["need"])
        close()
        out[n] = {"piles": piles, "where": where}
    return out


# ---------------------------------------------------------------- model calls
class Stub:
    """No model: a reader that gets worse as piles grow, for checking the bookkeeping only."""
    model = "stub"

    def __init__(self, gold):
        self.gold = gold

    def request(self, method, path, body):
        text = body["messages"][0]["content"]
        q = text.rsplit("Question: ", 1)[1].split("\n")[0]
        rng = random.Random(zlib.crc32(text.encode()) + body.get("seed", 0))
        n = text.count("\n- ") + (1 if "\n- " in text else 0)
        good, bad = self.gold[q]
        p = 0.97 - 0.04 * np.log10(max(n, 1)) - (0.1 if body["chat_template_kwargs"]["enable_thinking"] is False else 0)
        ans = good if rng.random() < p else rng.choice(bad)
        content = ("<think>stub</think>\n\n" if body["chat_template_kwargs"]["enable_thinking"] else "") + ans
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 5}}


def ask(llm, prompt, mode, seed):
    kw = THINK if mode == "think" else NOTHINK
    body = {"model": llm.model, "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": mode == "think"}, "seed": seed, **kw}
    t0 = time.time()
    r = llm.request("POST", "/chat/completions", body)
    ch = r["choices"][0]
    msg = ch.get("message") or {}
    content, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
    if "</think>" in content:
        reasoning, content = reasoning + content.split("</think>")[0], content.split("</think>")[-1]
    truncated = ch.get("finish_reason") == "length"
    if mode == "think" and truncated:
        content = ""
    u = r.get("usage") or {}
    return {"answer": content.strip(), "truncated": truncated, "think_chars": len(reasoning),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "sec": round(time.time() - t0, 2)}


def score(q, ans):
    if q["kind"] in ("w1", "w2", "own"):
        new = any(contains(ans, a) for a in q["new"])
        old = any(contains(ans, a) for a in q["old"])
        if q["kind"] == "own":
            return {"correct": old and not new, "old": False}
        return {"correct": new and not old, "old": old and not new}
    others = [v for v in ATTRS[q["rel"]]["vocab"] if norm(v) != norm(q["gold"])]
    ok = contains(ans, q["gold"]) and not any(contains(ans, v) for v in others)
    stale = q["kind"] == "upd" and contains(ans, q["old"]) and not contains(ans, q["gold"])
    return {"correct": ok, "old": stale}


def build():
    groups = world()
    PQ = questions(groups)
    PP = personal_piles(groups)
    texts, WQ, clash = world_items()
    WP = world_piles(texts, WQ)
    return groups, PQ, PP, texts, WQ, WP, clash


def items_for(cond, PQ, PP, WQ, WP):
    arm, mode, n = cond
    out = []
    if arm == "personal":
        for q in PQ:
            ci = PP[n]["where"][q["gid"]]
            notes = "\n".join("- " + x["text"] for x in PP[n]["piles"][ci])
            out.append((q, ci, PERSONAL.format(notes=notes, q=q["q"])))
    elif arm == "world":
        for q in WQ:
            ci = WP[n]["where"][q["qid"]]
            out.append((q, ci, WORLD.format(notes="\n".join("- " + t for t in WP[n]["piles"][ci]), q=q["q"])))
    else:
        for q in WQ:
            if q["kind"] == "w2":
                out.append(({**q, "kind": "own"}, 0, BARE.format(q=q["q"])))
    return out


def key(cond, q):
    return f"{cond[0]}|{cond[1]}|{cond[2]}|{q['qid']}"


def run(llm, conds, PQ, PP, WQ, WP, path, workers, budget_min, stub):
    done = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            done[r["key"]] = r
    lock, t0 = threading.Lock(), time.time()
    f = open(path, "a", encoding="utf-8")
    for cond in conds:
        if budget_min and (time.time() - t0) / 60 > budget_min:
            print(f"budget of {budget_min} min reached: not starting {cond}", flush=True)
            break
        todo = [(q, ci, p) for q, ci, p in items_for(cond, PQ, PP, WQ, WP) if key(cond, q) not in done]
        if not todo:
            continue
        c0 = time.time()
        by_pile = {}
        for it in todo:
            by_pile.setdefault(it[1], []).append(it)

        def one(it, cond=cond):
            q, ci, p = it
            r = ask(llm, p, cond[1], zlib.crc32(key(cond, q).encode()) % 2**31)
            r.update(key=key(cond, q), arm=cond[0], mode=cond[1], level=cond[2], qid=q["qid"], kind=q["kind"],
                     pile=ci, **score(q, r["answer"]))
            with lock:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
                f.flush()
                done[r["key"]] = r
            return r

        with ThreadPoolExecutor(workers) as ex:
            big = cond[2] == "all" or (isinstance(cond[2], int) and cond[2] >= 1000)
            if big and not stub:                      # one pile at a time, its prefix cached by a first call
                for ci, its in by_pile.items():
                    ask(llm, its[0][2].rsplit("Question: ", 1)[0] + "Question: How many notes are there?\nAnswer "
                        "with a number only.", "nothink", 1)
                    list(ex.map(one, its))
            else:
                list(ex.map(one, todo))
        rs = [done[key(cond, q)] for q, _, _ in items_for(cond, PQ, PP, WQ, WP)]
        acc = 100 * np.mean([r["correct"] for r in rs])
        print(f"{cond}: {len(rs)} answers, {acc:.1f}% right, {(time.time() - c0) / 60:.1f} min "
              f"(total {(time.time() - t0) / 60:.1f} min)", flush=True)
    f.close()
    return done


# ---------------------------------------------------------------- summary and verdict
def boot_mean(x, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (BOOT, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def boot_ratio(num, den, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), (BOOT, len(num)))
    r = num[idx].mean(1) / np.maximum(den[idx].mean(1), 1e-9)
    return num.mean() / max(den.mean(), 1e-9), np.percentile(r, 2.5), np.percentile(r, 97.5)


def get(done, arm, mode, n, kind):
    return sorted((r for r in done.values() if r["arm"] == arm and r["mode"] == mode and str(r["level"]) == str(n)
                   and r["kind"] == kind), key=lambda r: r["qid"])


def verdict(done):
    rw = get(done, "personal", "think", 5000, "reword")
    t5, t1 = get(done, "personal", "think", 5000, "two"), get(done, "personal", "think", 10, "two")
    if len(rw) < N_REWORD or len(t5) < N_TWO or len(t1) < N_TWO:
        return "NOT YET: the primary conditions (thinking on, 5,000 and 10 notes) are not complete.", {}
    a = boot_mean([r["correct"] for r in rw])
    acc10 = 100 * np.mean([r["correct"] for r in t1])
    ratio = boot_ratio([r["correct"] for r in t5], [r["correct"] for r in t1]) if acc10 >= MIN_TWO_AT_10 else None
    R = {"reword_5000": a, "two_10": acc10, "two_ratio": ratio}
    r_line = ("pass" if a[0] >= PASS_REWORD else "stop" if a[0] < STOP_REWORD else "between")
    t_line = (None if ratio is None else "pass" if ratio[0] >= PASS_RATIO else "stop" if ratio[0] < STOP_RATIO
              else "between")
    desc = (f"reworded single-hop at 5,000 notes {a[0]:.1f}% [{a[1]:.1f}, {a[2]:.1f}] (pass >= {PASS_REWORD:.0f}, stop "
            f"< {STOP_REWORD:.0f}); two-hop at 10 notes {acc10:.1f}%, ratio 5,000/10 " +
            ("not measurable (two-hop at 10 notes below 50%)" if ratio is None else
             f"{ratio[0]:.2f} [{ratio[1]:.2f}, {ratio[2]:.2f}] (pass >= {PASS_RATIO}, stop < {STOP_RATIO})"))
    if r_line == "pass" and t_line == "pass":
        v = ("PASS: " + desc + ". The whole notebook in the prompt works at 5,000 notes; it is version 1 and the bar "
             "every memory method has to beat.")
    elif r_line == "stop" or t_line == "stop":
        v = ("STOP: " + desc + ". Reading the whole notebook from the prompt is not version 1 at 5,000 notes; pages "
             "must be selected first.")
    else:
        v = "IN BETWEEN: " + desc + ". Report where it breaks; decide with the owner."
    return v, R


def write(out, tag, done, meta):
    L = [f"# Notebook reading at shipping size: {tag}", "",
         "Pre-registration: reports/Notebook reading at shipping size 2026.md. Accuracy in % with 95% bootstrap "
         f"intervals over questions ({BOOT:,} resamples).", ""]
    L += ["## Part A: personal notebook", "",
          "| Question | Mode | 10 notes | 100 | 1,000 | 5,000 |", "|---|---|---|---|---|---|"]
    for kind, name in (("reword", "single-hop, reworded"), ("same", "single-hop, note's own words"),
                       ("two", "two-hop (A's relation's fact)"), ("upd", "latest value after an update")):
        for mode in ("think", "nothink"):
            cells = []
            for n in LEVELS:
                rs = get(done, "personal", mode, n, kind)
                if rs:
                    m, lo, hi = boot_mean([r["correct"] for r in rs])
                    cells.append(f"{m:.1f} [{lo:.0f}-{hi:.0f}] n={len(rs)}")
                else:
                    cells.append("-")
            L.append(f"| {name} | {mode} | " + " | ".join(cells) + " |")
    L += ["", "Stale answers (the value before the update) among update questions:", ""]
    for mode in ("think", "nothink"):
        cells = []
        for n in LEVELS:
            rs = get(done, "personal", mode, n, "upd")
            cells.append(f"{100 * np.mean([r['old'] for r in rs]):.1f}%" if rs else "-")
        L.append(f"- {mode}: " + ", ".join(f"{n:,} notes {c}" for n, c in zip(LEVELS, cells)))
    L += ["", "Thinking truncated at the token cap (counted wrong), and mean completion tokens:", ""]
    for n in LEVELS:
        rs = [r for r in done.values() if r["arm"] == "personal" and r["mode"] == "think" and r["level"] == n]
        if rs:
            L.append(f"- {n:,} notes: {100 * np.mean([r['truncated'] for r in rs]):.1f}% truncated, "
                     f"{np.mean([r['completion_tokens'] or 0 for r in rs]):.0f} tokens, prompt "
                     f"{np.mean([r['prompt_tokens'] or 0 for r in rs]):.0f} tokens")
    L += ["", "## Part A2: conflicting world edits (MQuAKE-CF-3k-v2)", "",
          "| Question | Mode | 10 edits | 100 | 1,000 | all | Own facts, no notes |", "|---|---|---|---|---|---|---|"]
    for kind, name in (("w1", "single-hop, new answer"), ("w2", "two-hop, new answer")):
        for mode in ("think", "nothink"):
            cells = []
            for n in W_LEVELS:
                rs = get(done, "world", mode, n, kind)
                if rs:
                    m, lo, hi = boot_mean([r["correct"] for r in rs])
                    cells.append(f"{m:.1f} [{lo:.0f}-{hi:.0f}] old {100 * np.mean([r['old'] for r in rs]):.0f}%")
                else:
                    cells.append("-")
            own = get(done, "own", mode, 0, "own") if kind == "w2" else []
            cells.append(f"{boot_mean([r['correct'] for r in own])[0]:.1f}" if own else "-")
            L.append(f"| {name} | {mode} | " + " | ".join(cells) + " |")
    L += ["", "## Verdict", ""]
    v, R = verdict(done)
    L += [f"- {v}", "", f"World: {meta}"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump({"tag": tag, "meta": meta, "verdict": v, "primary": R}, f, indent=1, default=str)
    return L


def preflight(llm, PP, PQ):
    """Not scored: does the largest pile fit, is its prefix cached, and does thinking mode answer after </think>?"""
    big = PERSONAL.format(notes="\n".join("- " + x["text"] for x in PP[5000]["piles"][0]),
                          q="How many notes are there?")
    out = {"served": getattr(llm, "served", None), "model": llm.model}
    for k in range(2):
        r = ask(llm, big, "nothink", 1)
        out[f"big_call_{k}"] = {x: r[x] for x in ("prompt_tokens", "cached_tokens", "sec", "answer")}
    small = PERSONAL.format(notes="\n".join("- " + x["text"] for x in PP[10]["piles"][0]), q="Who is mentioned first?")
    r = ask(llm, small, "think", 1)
    out["think_call"] = {x: r[x] for x in ("answer", "truncated", "think_chars", "completion_tokens", "sec")}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    p.add_argument("--base-url", default=None)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--budget-min", type=float, default=0, help="stop starting new conditions after this many minutes")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--only", default=None, help="comma list of arms to run (personal,world,own); default all")
    args = p.parse_args()
    groups, PQ, PP, texts, WQ, WP, clash = build()
    meta = {"people": 2 * N_GROUPS, "groups": N_GROUPS, "notes": sum(len(group_notes(g)) for g in groups),
            "pile_notes": {n: [min(map(len, PP[n]["piles"])), max(map(len, PP[n]["piles"]))] for n in LEVELS},
            "pile_chars_5000": len("\n".join("- " + x["text"] for x in PP[5000]["piles"][0])),
            "personal_questions": {k: sum(q["kind"] == k for q in PQ) for k in ("reword", "same", "two", "upd")},
            "world_edits": len(texts), "world_edit_clashes_skipped": clash,
            "world_questions": {k: sum(q["kind"] == k for q in WQ) for k in ("w1", "w2")},
            "world_piles": {str(n): len(WP[n]["piles"]) for n in W_LEVELS}}
    print(json.dumps(meta), flush=True)
    if args.stub:
        gold = {}
        for q in PQ:
            gold[q["q"]] = (q["gold"], [v for v in ATTRS[q["rel"]]["vocab"] if v != q["gold"]])
        for q in WQ:
            gold[q["q"]] = (q["new"][0], [q["old"][0], "unknown"])
        llm, tag = Stub(gold), "stub"
    elif args.base_url:
        llm, tag = G.Endpoint(args.base_url, args.model, 1), args.model.split("/")[-1]
        print(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}", flush=True)
    else:
        raise SystemExit("needs --base-url or --stub")
    out = os.path.join(ROOT, "results", f"notebook_read_{tag}")
    os.makedirs(out, exist_ok=True)
    if args.preflight:
        r = preflight(llm, PP, PQ)
        print(json.dumps(r, indent=1), flush=True)
        with open(os.path.join(out, "preflight.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)
        return
    conds = [c for c in PRIORITY if not args.only or c[0] in args.only.split(",")]
    path = os.path.join(out, "answers.jsonl")
    done = run(llm, conds, PQ, PP, WQ, WP, path, args.workers, args.budget_min, args.stub)
    print("\n".join(write(out, tag, done, meta)), flush=True)


if __name__ == "__main__":
    main()
