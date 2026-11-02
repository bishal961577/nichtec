"""Test 1 of the expert panel's plan: can a model at shipping size read a whole notebook from its prompt?

Pre-registered in reports/Notebook reading at shipping size 2026.md: section 3 (first design) and section 7 (amendment
1, after the panel's review, still before any model call). This file implements amendment 1.

Part A, personal notebook (name in the question; the easy reference): a fictional world of 840 people in 420 groups
(person A with 4 facts, A's relation to person B, B with 4 facts). Half the relation notes are written from B's side
("Bilal Barbosa lives next door to Signe Larsen"). 135 groups change A's city or job once or twice, written as an
explicit change, a plain restatement (only order tells which is current) or a correction; 30 groups retract one of A's
facts with no new value. Kinds: reword, same, two (A has a different value of the asked attribute), upd, prev, retract.
Part A3, hard reading (no full name): desc ("What pet does the pharmacist who lives in Porto have?"), who ("Who plays
the mandolin and lives in Porto?"), fname ("What car does Paula, the one who plays the mandolin, drive?"), amb (a first
name shared by several people; does the model flag it?), close (30 people with 35 extra notes each: near-miss lures
and everyday notes). Levels: 10 (the group's own notes: perfect page selection), 100, 1,000, 5,000 notes, and bm25
(the 20 notes a BM25 search returns from the 5,000-note pile; reference) and search2 (the way of selecting pages the
decision uses: BM25 top 20, the model reads them and writes one follow-up search, BM25 top 20 again, answer from both).
Part A2, conflicting world edits: MQuAKE-CF-3k-v2 edits as notes, with and without "treat them as true".

  python brain/notebook_read.py --stub                                            # bookkeeping, no model
  python brain/notebook_read.py --base-url URL --model Qwen/Qwen3.5-27B-FP8 --preflight
  python brain/notebook_read.py --base-url URL --model Qwen/Qwen3.5-27B-FP8
Resumable: answers are appended to results/notebook_read_<model>/answers.jsonl and skipped on restart.
"""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import re
import sys
import threading
import time
import unicodedata
import urllib.request
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402  (the endpoint client: retries, auth headers)

ROOT = G.ROOT
SEED = 0
N_GROUPS, N_CLOSE, N_UPD, N_DOUBLE, N_RETRACT, N_FILLER = 420, 30, 135, 30, 30, 23
LEVELS = [10, 100, 1000, 5000]
GROUPS_PER_PILE = {10: 1, 100: 11, 1000: 84, 5000: 420}
BM25_K = 20
SEARCH = ("You are answering a question from a user's notebook. A search of the notebook returned these notes:\n{notes}"
          "\n\nQuestion: {q}\nIf these notes are enough to answer the question, reply DONE. Otherwise reply with one short "
          "search query (a few words, for example a person's name and what you need to know) that would find the missing "
          "note. Reply with DONE or the query only.")
COUNTS = {"reword": 150, "same": 100, "two": 200, "prev": 20, "desc": 120, "who": 60, "fname": 60, "amb": 30}
A_KINDS = ["reword", "same", "two", "upd", "prev", "retract"]
A3_KINDS = ["desc", "who", "fname", "amb", "close"]
HARD = ["desc", "who", "fname", "close", "two", "upd", "retract"]       # the set the decision is read on
N_W1, N_W2 = 150, 120
THINK = {"max_tokens": 16384, "temperature": 0.6, "top_p": 0.95, "top_k": 20}
NOTHINK = {"max_tokens": 64, "temperature": 0}
MQUAKE = "MQuAKE-CF-3k-v2.json"
MQUAKE_URL = "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/" + MQUAKE
MQUAKE_SHA256 = "f82091cbb668cef8f2537f79f1768af1840184450b347ff8c344336090ddc71e"
BOOT = 10_000

# Pre-registered lines (amendment 1, section 7.5 of the report). Decisions use point estimates.
VIABLE = 0.8          # a way of reading is viable if its hard-set accuracy is >= 0.8 x perfect page selection (level 10)
CLOSE_PTS = 5.0       # ... and no more than 5 points below the other way of reading
LATENCY_S = 10.0      # the whole notebook also needs a warm-cache answer (thinking off) within 10 s
PASS_REWORD, STOP_REWORD, PASS_RATIO, STOP_RATIO = 85.0, 70.0, 0.8, 0.5   # easy reference lines (first design)


def plan(tag):
    """Conditions in run order. 27B: the decision first. 4B (phone): pages and small piles only."""
    if "4B" in tag:
        return ([("personal", "think", n) for n in (10, "search2", "bm25", 100)] +
                [("personal", "nothink", n) for n in (10, "search2", 100)] +
                [("world", "think", n) for n in (10, 100)] + [("own", "think", 0)])
    return ([("personal", "think", n) for n in (5000, 10, "search2", "bm25", 1000, 100)] +
            [("personal", "nothink", n) for n in (5000, 10, "search2")] +
            [("world", "think", n) for n in ("all", 10)] + [("own", "think", 0), ("world_noinst", "think", "all")] +
            [("world", "think", n) for n in (1000, 100)] +
            [("world", "nothink", n) for n in ("all", 10)] + [("own", "nothink", 0)])


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

# fact: the note; same / reword: questions by name (reword shares no content word with the note); two: through A's
# relation; desc: a question whose subject {p} can be a long phrase; clause: "who ..." descriptor; lures: near-misses
ATTRS = {
    "job": dict(fact="{p} works as {art} {v}.", same="What does {p} work as?", reword="What is {p}'s profession?",
                two="What does {a}'s {r} work as?", desc="What does {p} work as?", clause="works as {art} {v}",
                lures=["{p} trained as {art} {v} years ago but never worked as one.", "{p}'s mother works as {art} {v}.",
                       "{p} is thinking about retraining as {art} {v}."],
                vocab="pharmacist carpenter architect nurse electrician journalist plumber librarian veterinarian "
                      "translator accountant firefighter baker pilot geologist physiotherapist florist mechanic optician "
                      "surveyor tailor butcher midwife locksmith chef lawyer economist beekeeper glassblower "
                      "cartographer".split()),
    "city": dict(fact="{p} lives in {v}.", same="Where does {p} live?", reword="Which city is {p}'s home?",
                 two="Where does {a}'s {r} live?", desc="Where does {p} live?", clause="lives in {v}",
                 lures=["{p} is visiting {v} this week.", "{p} grew up in {v}.", "{p} flew to {v} for a wedding."],
                 vocab="Lisbon Porto Lyon Leeds Krakow Utrecht Bergen Graz Seville Turin Tampere Ghent Brno Malmo "
                       "Bilbao Cork Aarhus Dresden Bologna Nantes Valencia Gdansk Salzburg Antwerp Edinburgh Zagreb "
                       "Ljubljana Tallinn Riga Vilnius".split()),
    "pet": dict(fact="{p} has a pet {v}.", same="What pet does {p} have?",
                reword="Which animal does {p} look after at home?", two="What pet does {a}'s {r} have?",
                desc="What pet does {p} have?", clause="has a pet {v}",
                lures=["{p} is thinking about getting {art} {v}.", "{p} looked after a neighbour's {v} last month.",
                       "{p}'s childhood pet was {art} {v}."],
                vocab=["parrot", "tortoise", "hamster", "ferret", "rabbit", "goldfish", "iguana", "cockatiel",
                       "chinchilla", "gecko", "hedgehog", "guinea pig", "canary", "axolotl", "beagle", "dalmatian",
                       "siamese cat", "python", "tarantula", "corn snake"]),
    "hobby": dict(fact="In their free time, {p} enjoys {v}.", same="What does {p} enjoy doing in their free time?",
                  reword="What is {p}'s hobby?", two="What does {a}'s {r} enjoy doing in their free time?",
                  desc="What does {p} enjoy doing in their free time?", clause="enjoys {v} in their free time",
                  lures=["{p} watched a documentary about {v}.", "{p} tried {v} once and did not like it.",
                         "{p}'s partner is really into {v}."],
                  vocab=["pottery", "rock climbing", "birdwatching", "chess", "knitting", "surfing", "fencing",
                         "origami", "archery", "kayaking", "calligraphy", "gardening", "juggling", "sailing",
                         "woodcarving", "salsa dancing", "astronomy", "quilting", "fishing", "photography",
                         "skateboarding", "cycling", "embroidery", "orienteering"]),
    "food": dict(fact="{p}'s favourite food is {v}.", same="What is {p}'s favourite food?",
                 reword="Which dish does {p} like eating most?", two="What is the favourite food of {a}'s {r}?",
                 desc="What is the favourite food of {p}?", clause="has {v} as their favourite food",
                 lures=["{p} cooked {v} for friends last night but did not enjoy it.", "{p}'s mother loves {v}.",
                        "{p} ordered {v} at lunch because nothing else was left."],
                 vocab=["lasagne", "ramen", "paella", "falafel", "goulash", "sushi", "pierogi", "moussaka", "risotto",
                        "tacos", "pho", "biryani", "ceviche", "shakshuka", "gnocchi", "bibimbap", "couscous",
                        "empanadas", "borscht", "poutine", "pad thai", "dumplings", "jollof rice", "tagine"]),
    "instrument": dict(fact="{p} plays the {v}.", same="Which instrument does {p} play?",
                       reword="Which musical instrument has {p} mastered?", two="Which instrument does {a}'s {r} play?",
                       desc="Which instrument does {p} play?", clause="plays the {v}",
                       lures=["{p} borrowed a friend's {v} for a week.", "{p} went to a {v} concert.",
                              "{p}'s child is learning the {v}."],
                       vocab="violin cello trumpet accordion harp banjo clarinet oboe flute saxophone ukulele piano "
                             "drums mandolin bassoon harmonica trombone sitar bagpipes xylophone".split()),
    "car": dict(fact="{p} drives {art} {v}.", same="What car does {p} drive?",
                reword="Which vehicle brand does {p} own?", two="What car does {a}'s {r} drive?",
                desc="What car does {p} drive?", clause="drives {art} {v}",
                lures=["{p} test-drove {art} {v}.", "{p}'s brother drives {art} {v}.", "{p} rented {art} {v} on holiday."],
                vocab="Volvo Toyota Skoda Fiat Renault Subaru Mazda Peugeot Kia Hyundai Tesla Audi Honda Nissan "
                      "Citroen Opel Dacia Jeep Saab Lexus".split()),
    "language": dict(fact="{p} is learning {v}.", same="Which language is {p} learning?",
                     reword="What foreign tongue is {p} studying?", two="Which language is {a}'s {r} learning?",
                     desc="Which language is {p} learning?", clause="is learning {v}",
                     lures=["{p} is considering learning {v} next year.", "{p}'s grandmother spoke {v}.",
                            "{p} watched a film in {v}."],
                     vocab="Japanese Portuguese Swahili Finnish Korean Turkish Greek Polish Arabic Hindi Dutch "
                           "Icelandic Hungarian Vietnamese Tagalog Czech Welsh Hebrew Mandarin Italian".split()),
    "allergy": dict(fact="{p} is allergic to {v}.", same="What is {p} allergic to?",
                    reword="Which substance must {p} avoid for medical reasons?", two="What is {a}'s {r} allergic to?",
                    desc="What is {p} allergic to?", clause="is allergic to {v}",
                    lures=["{p} has a friend who is allergic to {v}.",
                           "{p} was tested for an allergy to {v} and the test was negative.",
                           "{p} avoids {v} out of habit, not because of an allergy."],
                    vocab=["peanuts", "shellfish", "penicillin", "pollen", "latex", "bee stings", "gluten", "sesame",
                           "strawberries", "dust mites", "eggs", "soy", "walnuts", "aspirin", "kiwi", "mustard"]),
}
CHANGE = {  # how a change of A's city or job is written
    "explicit": {"city": "{p} has moved to {v}.", "job": "{p} has changed careers and now works as {art} {v}."},
    "plain": {"city": ATTRS["city"]["fact"], "job": ATTRS["job"]["fact"]},
    "correction": {"city": "Correction: {p} lives in {v}, not {old}.",
                   "job": "Correction: {p} works as {art} {v}, not as {art_old} {old}."},
}
UPD_NOW = {"city": "Which city does {p} live in now?", "job": "What does {p} work as now?"}
PREV_Q = {"city": "Where did {p} live before moving to {new}?", "job": "What did {p} work as before becoming {art} {new}?"}
RETRACT = {"car": "{p} has sold the {v} and no longer has a car.",
           "instrument": "{p} has given up the {v} and no longer plays any instrument.",
           "pet": "{p}'s pet {v} has gone to a new home, and {p} has no pet now.",
           "language": "{p} has stopped learning {v} and is not learning any language now.",
           "hobby": "{p} has given up {v} and has no hobby at the moment."}
RELATIONS = ["neighbour", "best friend", "manager", "cousin", "flatmate", "landlord", "colleague", "mentor",
             "business partner", "personal trainer"]  # gender-neutral: names are random
REL_REV = {"neighbour": "{b} lives next door to {a}.",
           "best friend": "{a} trusts {b} more than anyone; they have been inseparable since school.",
           "manager": "{a} reports to {b} at work.", "cousin": "{b}'s mother and {a}'s mother are sisters.",
           "flatmate": "{a} and {b} share a flat.", "landlord": "{a} rents a flat from {b}.",
           "colleague": "{a} and {b} work in the same office.", "mentor": "{b} has been guiding {a}'s career for years.",
           "business partner": "{a} and {b} co-own a small company.",
           "personal trainer": "{b} coaches {a} at the gym twice a week."}
FILLER = ["{p} had a dentist appointment on {day}.", "{p} called to talk about the weekend.",
          "{p} said the week at work was exhausting.", "{p} recommended a podcast about history.",
          "{p} is saving up for a holiday.", "{p} finished reading a long novel.", "{p} forgot an umbrella on {day}.",
          "{p} is redecorating the living room.", "{p} has a birthday coming up in {month}.",
          "{p} complained about the noisy street.", "{p} started waking up earlier.", "{p} lent me a book about the sea.",
          "{p} is worried about a friend.", "{p} got a parking ticket on {day}.", "{p} bought new running shoes.",
          "{p} is planning a picnic for {month}.", "{p} said the train was late again.", "{p} repainted the kitchen.",
          "{p} asked how my exams went.", "{p} is trying to drink less coffee.", "{p} missed the bus on {day}.",
          "{p} sent a photo of a sunset.", "{p} is fixing a leaking tap.", "{p} lost a glove in {month}.",
          "{p} has a cold this week.", "{p} found an old diary.", "{p} wants to move to a bigger flat someday."]
DAYS = "Monday Tuesday Wednesday Thursday Friday Saturday Sunday".split()
MONTHS = "January February March April May June July August September October November December".split()
STOP = set("a an the is are does do what which who where whose of to in their his her has have for at on by with as "
           "from it its be and or now most must after".split())
ABSTAIN = [" no ", " none ", " not ", " doesn t ", " does not ", " no longer ", " nothing ", " unknown ", " sold ",
           " given up ", " gave up ", " stopped ", " neither ", " any "]
AMBIG = [" which ", " several ", " more than one ", " multiple ", " two ", " three ", " both ", " depends ",
         " ambiguous ", " clarify ", " different ", " each "]

PERSONAL = ("Below are notes a user has written in their personal notebook, in the order they were written. Later notes "
            "may update earlier ones.\n\nNotes:\n{notes}\n\nQuestion: {q}\nAnswer with the answer only, in a few words.")
WORLD = ("Below are notes about changes in the world, in the order they were written. Treat them as true, even where "
         "they differ from what you know.\n\nNotes:\n{notes}\n\nQuestion: {q}\nAnswer with the answer only, in a few "
         "words.")
WORLD_NOINST = ("Below are notes about changes in the world, in the order they were written.\n\nNotes:\n{notes}\n\n"
                "Question: {q}\nAnswer with the answer only, in a few words.")
BARE = "Question: {q}\nAnswer with the answer only, in a few words."


# ---------------------------------------------------------------- text helpers
def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


def stem(t):
    if len(t) > 4 and t.endswith("ies"):
        return t[:-3] + "y"
    if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
        return t[:-1]
    return t


def pnorm(s):
    return " ".join(stem(t) for t in norm(s).split())


def contains(ans, v):
    """Whole-word match after normalisation, ignoring plural -s (amendment 1)."""
    v = pnorm(v)
    return bool(v) and f" {v} " in f" {pnorm(ans)} "


def stems(template):
    words = re.sub(r"\{\w+\}", " ", template.lower().replace("'s", " "))
    return {w[:4] for w in re.findall(r"[a-z]+", words) if w not in STOP}


def art(v):
    return "an" if v[0].lower() in "aeiou" else "a"


def fill(t, **kw):
    if "{art}" in t:
        kw["art"] = art(kw["v"] if "v" in kw else kw["new"])
    if "{art_old}" in t:
        kw["art_old"] = art(kw["old"])
    return t.format(**kw)


def prefix_free(names):
    names = sorted(dict.fromkeys(names))
    return [n for n in names if not any(o != n and n.startswith(o) for o in names)]


def subject_in(template, subject):
    """Put a long subject phrase into a question template; 'Paula, the one who ..., ' needs its comma handled."""
    return template.replace("{p}", subject).replace(",?", "?").replace(", ?", "?")


# ---------------------------------------------------------------- part A: the personal world
def checks():
    """Fails loudly if the world could score a wrong answer as right."""
    all_values = [pnorm(v) for a in ATTRS.values() for v in a["vocab"]]
    for r, a in ATTRS.items():
        vs = [pnorm(v) for v in a["vocab"]]
        assert len(set(vs)) == len(vs) >= 16, r
        for v in vs:
            assert not any(o != v and f" {v} " in f" {o} " for o in vs), (r, v)
        assert not stems(a["reword"]) & stems(a["fact"]), (r, stems(a["reword"]) & stems(a["fact"]))
        assert stems(a["same"]) & stems(a["fact"]), r
    for f in FILLER:
        t = f" {pnorm(f.format(p='', day='', month=''))} "
        assert not any(f" {v} " in t for v in all_values), f


def names():
    vocab_words = {w for a in ATTRS.values() for v in a["vocab"] for w in norm(v).split()}
    vocab_words |= {w for r in RELATIONS for w in r.split()}
    first = [n for n in prefix_free(FIRST) if norm(n) not in vocab_words]
    last = [n for n in prefix_free(LAST) if norm(n) not in vocab_words and n not in first]
    return first, last


def world():
    """Groups and a registry of people: current facts, every value ever stated, lures."""
    checks()
    first, last = names()
    rng = random.Random(SEED)
    combos = [(f, l) for f in first for l in last]
    rng.shuffle(combos)
    used, pos = set(), [0]

    def take(bad_first=None, bad_last=None):
        while True:
            f, l = combos[pos[0]]
            pos[0] += 1
            if f"{f} {l}" in used or f == bad_first or l == bad_last:
                continue
            used.add(f"{f} {l}")
            return f"{f} {l}", f, l

    gids = list(range(N_GROUPS))
    rng.shuffle(gids)
    close = set(gids[:N_CLOSE])
    upd = gids[N_CLOSE:N_CLOSE + N_UPD]
    retract = set(gids[N_CLOSE + N_UPD:N_CLOSE + N_UPD + N_RETRACT])
    forms = ["explicit", "plain", "correction"] * (N_UPD // 3)
    rng.shuffle(forms)
    form_of = dict(zip(upd, forms))
    double = set()
    for form in ("explicit", "plain", "correction"):
        double |= set(rng.sample([g for g in upd if form_of[g] == form], N_DOUBLE // 3))
    groups, P = [], {}
    for g in range(N_GROUPS):
        a, fa, la = take()
        b, fb, lb = take(fa, la)
        attrs = {}
        for who in ("a", "b"):
            rels = rng.sample(sorted(ATTRS), 4)
            if who == "a" and g in form_of and not {"city", "job"} & set(rels):
                rels[0] = rng.choice(["city", "job"])
            if who == "a" and g in retract and not set(RETRACT) & set(rels):
                rels[0] = rng.choice(sorted(RETRACT))
            attrs[who] = {r: rng.choice(ATTRS[r]["vocab"]) for r in rels}
        gr = {"gid": g, "a": a, "b": b, "rel": rng.choice(RELATIONS), "rev": rng.random() < 0.5, "attrs": attrs,
              "change": None, "retract": None, "close": g in close, "lures": {}, "fillers": []}
        if g in form_of:
            r = rng.choice(sorted({"city", "job"} & set(attrs["a"])))
            vals = [attrs["a"][r]]
            for _ in range(2 if g in double else 1):
                vals.append(rng.choice([v for v in ATTRS[r]["vocab"] if v not in vals]))
            gr["change"] = {"rel": r, "form": form_of[g], "values": vals}
        if g in retract:
            gr["retract"] = rng.choice(sorted(set(RETRACT) & set(attrs["a"])))
        if g in close:
            for r, v in attrs["a"].items():
                gr["lures"][r] = rng.sample([x for x in ATTRS[r]["vocab"] if x != v], 3)
            gr["fillers"] = [f.format(p="{p}", day=rng.choice(DAYS), month=rng.choice(MONTHS))
                             for f in rng.sample(FILLER, N_FILLER)]
        groups.append(gr)
        for who, (name, f, l) in (("a", (a, fa, la)), ("b", (b, fb, lb))):
            cur = dict(attrs[who])
            hist = {r: [v] for r, v in cur.items()}
            if who == "a" and gr["change"]:
                hist[gr["change"]["rel"]] = list(gr["change"]["values"])
                cur[gr["change"]["rel"]] = gr["change"]["values"][-1]
            if who == "a" and gr["retract"]:
                del cur[gr["retract"]]
            P[name] = {"gid": g, "role": who, "first": f, "last": l, "cur": cur, "hist": hist,
                       "ever": {r: set(vs) for r, vs in hist.items()},
                       "lures": {r: set(vs) for r, vs in gr["lures"].items()} if who == "a" else {},
                       "changed": ({gr["change"]["rel"]} if who == "a" and gr["change"] else set()) |
                                  ({gr["retract"]} if who == "a" and gr["retract"] else set()),
                       "partner": b if who == "a" else a}
    return groups, P


def group_notes(gr):
    """The group's notes. 'after' names the note a later note must follow."""
    notes = []
    for who in ("a", "b"):
        for r, v in gr["attrs"][who].items():
            notes.append({"text": fill(ATTRS[r]["fact"], p=gr[who], v=v), "gid": gr["gid"], "tag": f"{who}.{r}"})
    rel = (REL_REV[gr["rel"]] if gr["rev"] else "{a}'s " + gr["rel"] + " is {b}.").format(a=gr["a"], b=gr["b"])
    notes.append({"text": rel, "gid": gr["gid"], "tag": "rel"})
    if gr["change"]:
        c = gr["change"]
        for k, v in enumerate(c["values"][1:], start=1):
            t = fill(CHANGE[c["form"]][c["rel"]], p=gr["a"], v=v, old=c["values"][k - 1])
            notes.append({"text": t, "gid": gr["gid"], "tag": f"chg{k}",
                          "after": "a." + c["rel"] if k == 1 else f"chg{k - 1}"})
    if gr["retract"]:
        r = gr["retract"]
        notes.append({"text": RETRACT[r].format(p=gr["a"], v=gr["attrs"]["a"][r]), "gid": gr["gid"], "tag": "ret",
                      "after": "a." + r})
    for r, vs in gr["lures"].items():
        for t, v in zip(ATTRS[r]["lures"], vs):
            notes.append({"text": fill(t, p=gr["a"], v=v), "gid": gr["gid"], "tag": f"lure.{r}"})
    for t in gr["fillers"]:
        notes.append({"text": t.format(p=gr["a"]), "gid": gr["gid"], "tag": "fill"})
    return notes


def place_later(notes, rng):
    """Every change or retraction goes to a random position after the note it follows."""
    for tag in ("chg1", "chg2", "ret"):
        for u in [n for n in notes if n["tag"] == tag]:
            notes.remove(u)
            i = next(k for k, n in enumerate(notes) if n["gid"] == u["gid"] and n["tag"] == u["after"])
            notes.insert(rng.randint(i + 1, len(notes)), u)
    return notes


def unchanged(P, name):
    p = P[name]
    return sorted(r for r in p["cur"] if r not in p["changed"])


def questions(groups, P):
    rng = random.Random(SEED + 2)
    normal = [g for g in groups if not g["close"]]
    Q = []

    def value_q(kind, k, name, r, text, gold=None, **extra):
        Q.append({"qid": f"{kind}-{k:03d}", "kind": kind, "person": name, "rel": r, "q": text,
                  "gold": gold if gold is not None else P[name]["cur"][r], "gid": P[name]["gid"], **extra})

    # A: name in the question
    for k, gr in enumerate(rng.sample(normal, COUNTS["reword"])):
        name = gr[rng.choice(["a", "b"])]
        r = rng.choice(unchanged(P, name))
        value_q("reword", k, name, r, ATTRS[r]["reword"].format(p=name))
        if k < COUNTS["same"]:
            value_q("same", k, name, r, ATTRS[r]["same"].format(p=name))
    two = []
    for gr in normal:
        shared = [r for r in gr["attrs"]["b"] if r in unchanged(P, gr["a"]) and P[gr["a"]]["cur"][r] != gr["attrs"]["b"][r]]
        if shared:
            two.append((gr, rng.choice(sorted(shared))))
    for k, (gr, r) in enumerate(rng.sample(two, COUNTS["two"])):
        value_q("two", k, gr["b"], r, ATTRS[r]["two"].format(a=gr["a"], r=gr["rel"]), via=gr["a"])
    changed = [g for g in groups if g["change"]]
    for k, gr in enumerate(changed):
        c = gr["change"]
        text = (UPD_NOW[c["rel"]] if k % 2 == 0 else ATTRS[c["rel"]]["same"]).format(p=gr["a"])
        value_q("upd", k, gr["a"], c["rel"], text, form=c["form"], n_changes=len(c["values"]) - 1, now=k % 2 == 0)
    single_explicit = [g for g in changed if g["change"]["form"] == "explicit" and len(g["change"]["values"]) == 2]
    for k, gr in enumerate(rng.sample(single_explicit, COUNTS["prev"])):
        c = gr["change"]
        value_q("prev", k, gr["a"], c["rel"], fill(PREV_Q[c["rel"]], p=gr["a"], new=c["values"][1]),
                gold=c["values"][0], new=c["values"][1])
    for k, gr in enumerate(g for g in groups if g["retract"]):
        r = gr["retract"]
        value_q("retract", k, gr["a"], r, ATTRS[r]["same"].format(p=gr["a"]), gold="", old=gr["attrs"]["a"][r])

    # A3: no full name
    idx = defaultdict(set)
    for name, p in P.items():
        for r, vs in p["ever"].items():
            for v in vs:
                idx[(r, v)].add(name)
    pool = [n for n in sorted(P) if not groups[P[n]["gid"]]["close"]]

    def unique_pairs(name):
        u = unchanged(P, name)
        return [(r1, r2) for r1, r2 in itertools.combinations(u, 2)
                if idx[(r1, P[name]["cur"][r1])] & idx[(r2, P[name]["cur"][r2])] == {name}]

    def clause(name, r):
        return fill(ATTRS[r]["clause"], v=P[name]["cur"][r])

    order = list(pool)
    rng.shuffle(order)
    desc, who, taken = [], [], set()
    for name in order:
        pairs = unique_pairs(name)
        if not pairs:
            continue
        r1, r2 = rng.choice(pairs)
        rest = [r for r in unchanged(P, name) if r not in (r1, r2)]
        if len(desc) < COUNTS["desc"] and rest:
            t = rng.choice(rest)
            if "job" in (r1, r2):
                other = r2 if r1 == "job" else r1
                subj = f"the {P[name]['cur']['job']} who {clause(name, other)}"
            else:
                subj = f"the person who {clause(name, r1)} and {clause(name, r2)}"
            desc.append((name, t, subject_in(ATTRS[t]["desc"], subj), (r1, r2)))
            taken.add(name)
        elif len(who) < COUNTS["who"]:
            who.append((name, f"Who {clause(name, r1)} and {clause(name, r2)}?", (r1, r2)))
            taken.add(name)
        if len(desc) == COUNTS["desc"] and len(who) == COUNTS["who"]:
            break
    for k, (name, t, text, d) in enumerate(desc):
        value_q("desc", k, name, t, text, descriptors=d)
    for k, (name, text, d) in enumerate(who):
        Q.append({"qid": f"who-{k:03d}", "kind": "who", "person": name, "rel": None, "q": text, "gold": name,
                  "gid": P[name]["gid"], "descriptors": d})
    by_first = defaultdict(list)
    for name, p in P.items():
        by_first[p["first"]].append(name)
    fn = []
    for name in order:
        if name in taken or len(by_first[P[name]["first"]]) < 3:
            continue
        others = [o for o in by_first[P[name]["first"]] if o != name]
        dis = [r for r in unchanged(P, name) if not any(P[name]["cur"][r] in P[o]["ever"].get(r, ()) for o in others)]
        if not dis:
            continue
        r = rng.choice(dis)
        rest = [t for t in unchanged(P, name) if t != r]
        if not rest:
            continue
        t = rng.choice(rest)
        subj = f"{P[name]['first']}, the one who {clause(name, r)},"
        fn.append((name, t, subject_in(ATTRS[t]["desc"], subj), r))
        taken.add(name)
        if len(fn) == COUNTS["fname"]:
            break
    for k, (name, t, text, r) in enumerate(fn):
        value_q("fname", k, name, t, text, disambiguator=r)
    amb = []
    for f in sorted(by_first):
        for t in sorted(ATTRS):
            vals = {P[n]["cur"][t] for n in by_first[f] if t in P[n]["cur"]}
            if len(vals) >= 2:
                amb.append((f, t, sorted(vals)))
    for k, (f, t, vals) in enumerate(rng.sample(amb, COUNTS["amb"])):
        Q.append({"qid": f"amb-{k:03d}", "kind": "amb", "person": None, "rel": t, "q": ATTRS[t]["desc"].format(p=f),
                  "gold": "", "values": vals, "gid": None, "first": f})
    k = 0
    for gr in groups:
        if gr["close"]:
            for r in rng.sample(sorted(gr["attrs"]["a"]), 3):
                value_q("close", k, gr["a"], r, ATTRS[r]["same"].format(p=gr["a"]))
                k += 1
    return Q


def personal_piles(groups):
    """For each level: the piles (note lists) and the pile index of every group; plus the BM25 index of the big pile."""
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
            piles.append(place_later(notes, rng))
            for g in chunk:
                where[int(g)] = ci
        out[n] = {"piles": piles, "where": where}
    return out


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [[t for t in pnorm(d).split() if t not in STOP] for d in docs]
        self.k1, self.b = k1, b
        self.avg = np.mean([len(t) for t in self.toks])
        df = Counter(t for d in self.toks for t in set(d))
        self.idf = {t: math.log(1 + (len(docs) - c + 0.5) / (c + 0.5)) for t, c in df.items()}
        self.tf = [Counter(d) for d in self.toks]

    def top(self, query, k):
        q = [t for t in pnorm(query).split() if t not in STOP]
        s = []
        for i, (tf, d) in enumerate(zip(self.tf, self.toks)):
            x = sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) /
                    self.avg)) for t in q if t in tf)
            s.append((-x, i))
        return sorted(i for _, i in sorted(s)[:k])          # the top k, back in notebook order


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


def wcontains(ans, v):
    v = norm(v)
    return bool(v) and f" {v} " in f" {norm(ans)} "


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
            notes = list(need) + r.sample([t for t in texts if t not in need], n - len(need))
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


# ---------------------------------------------------------------- scoring and error sources
def score(q, ans, P=None):
    k = q["kind"]
    if k in ("w1", "w2", "own"):
        new = any(wcontains(ans, a) for a in q["new"])
        old = any(wcontains(ans, a) for a in q["old"])
        if k == "own":
            return {"correct": old and not new, "old": False}
        return {"correct": new and not old, "old": old and not new}
    t = f" {pnorm(ans)} "
    if k == "who":
        named = [n for n in P if contains(ans, n)]
        return {"correct": named == [q["gold"]], "old": False, "named": named}
    vocab = ATTRS[q["rel"]]["vocab"]
    named = [v for v in vocab if contains(ans, v)]
    if k == "amb":
        return {"correct": any(m in t for m in AMBIG) or len(set(named) & set(q["values"])) >= 2, "old": False,
                "named": named}
    if k == "retract":
        return {"correct": not named and any(m in t for m in ABSTAIN), "old": q["old"] in named, "named": named}
    if k == "prev":
        return {"correct": q["gold"] in named and not set(named) - {q["gold"], q["new"]}, "old": False, "named": named}
    hist = P[q["person"]]["hist"].get(q["rel"], [])
    return {"correct": named == [q["gold"]], "old": q["gold"] not in named and bool(set(named) & set(hist[:-1])),
            "named": named, "first_named_right": bool(named) and min(named, key=lambda v: t.find(f" {pnorm(v)} ")) ==
            q["gold"]}


def source(q, r, P):
    """Where a wrong answer came from (amendment 1: the interference profile)."""
    if r["correct"] or q["kind"] in ("w1", "w2", "own", "amb"):
        return None
    if r.get("truncated"):
        return "cut off"
    named = r.get("named") or []
    if q["kind"] == "who":
        if not named:
            return "no name"
        d = q["descriptors"]
        return ("partial match" if any(P[n]["cur"].get(x) == P[q["person"]]["cur"][x] for n in named for x in d)
                else "other person")
    if q["kind"] == "retract":
        return "old value" if q["old"] in named else ("other value" if named else "no 'none' in answer")
    wrong = [v for v in named if v != q["gold"]]
    if not wrong:
        return "no value"
    v, p, rel = wrong[0], P[q["person"]], q["rel"]
    if v in p["hist"].get(rel, [])[:-1]:
        return "old value"
    if v in p["lures"].get(rel, ()):
        return "lure"
    if q["kind"] == "two" and P[q["via"]]["cur"].get(rel) == v:
        return "the asker's own value (A)"
    if P[p["partner"]]["cur"].get(rel) == v:
        return "group partner"
    if any(v in P[o]["ever"].get(rel, ()) for o in P if o != q["person"] and P[o]["first"] == p["first"]):
        return "same first name"
    if any(v in P[o]["ever"].get(rel, ()) for o in P if o != q["person"] and P[o]["last"] == p["last"]):
        return "same last name"
    return "another person's value" if any(v in P[o]["ever"].get(rel, ()) for o in P) else "made up"


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
        n = text.count("\n- ") + 1
        good, bad = self.gold.get(q, ("unknown", ["unknown"]))
        think = body["chat_template_kwargs"]["enable_thinking"]
        p = 0.97 - 0.05 * np.log10(max(n, 1)) - (0.1 if not think else 0)
        ans = good if rng.random() < p else rng.choice(bad)
        content = ("<think>stub</think>\n\n" if think else "") + ans
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


def build():
    groups, P = world()
    PQ = questions(groups, P)
    PP = personal_piles(groups)
    big = PP[5000]["piles"][0]
    bm = BM25([x["text"] for x in big])
    BMP = {q["qid"]: [big[i] for i in bm.top(q["q"], BM25_K)] for q in PQ}
    texts, WQ, clash = world_items()
    WP = world_piles(texts, WQ)
    return groups, P, PQ, PP, BMP, texts, WQ, WP, clash, bm


def kinds_at(level):
    """A3 runs on the decision levels only; an ambiguous first name needs a pile holding several people."""
    if level in (100, 1000):
        return A_KINDS
    return A_KINDS + [k for k in A3_KINDS if k != "amb"] if level == 10 else A_KINDS + A3_KINDS


def items_for(cond, D):
    arm, mode, n = cond
    out = []
    if arm == "personal":
        for q in D["PQ"]:
            if q["kind"] not in kinds_at(n):
                continue
            if n in ("bm25", "search2"):
                notes, ci = D["BMP"][q["qid"]], q["qid"]
            else:                                      # amb (no group) is asked only where one pile holds everyone
                ci = 0 if q["gid"] is None else D["PP"][n]["where"][q["gid"]]
                notes = D["PP"][n]["piles"][ci]
            out.append((q, ci, PERSONAL.format(notes="\n".join("- " + x["text"] for x in notes), q=q["q"])))
    elif arm in ("world", "world_noinst"):
        tmpl = WORLD if arm == "world" else WORLD_NOINST
        for q in D["WQ"]:
            ci = D["WP"][n]["where"][q["qid"]]
            out.append((q, ci, tmpl.format(notes="\n".join("- " + t for t in D["WP"][n]["piles"][ci]), q=q["q"])))
    else:
        for q in D["WQ"]:
            if q["kind"] == "w2":
                out.append(({**q, "kind": "own"}, 0, BARE.format(q=q["q"])))
    return out


def key(cond, q):
    return f"{cond[0]}|{cond[1]}|{cond[2]}|{q['qid']}"


def run(llm, conds, D, path, workers, budget_min, stub):
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
        items = items_for(cond, D)
        todo = [it for it in items if key(cond, it[0]) not in done]
        if not todo:
            continue
        c0 = time.time()

        def one(it, cond=cond):
            q, ci, p = it
            seed = zlib.crc32(key(cond, q).encode()) % 2**31
            query = None
            if cond[2] == "search2":                  # round 1 notes -> the model's follow-up query -> round 2
                first = D["BMP"][q["qid"]]
                s1 = ask(llm, SEARCH.format(notes="\n".join("- " + x["text"] for x in first), q=q["q"]), "nothink", seed)
                query = (s1["answer"].strip().splitlines() or ["DONE"])[0].strip()
                notes = list(first)
                if "DONE" not in query.upper():
                    big = D["PP"][5000]["piles"][0]
                    extra = [big[i] for i in D["BM"].top(query, BM25_K)]
                    pos = {id(x): i for i, x in enumerate(big)}
                    notes = sorted({id(x): x for x in first + extra}.values(), key=lambda x: pos[id(x)])
                p = PERSONAL.format(notes="\n".join("- " + x["text"] for x in notes), q=q["q"])
            r = ask(llm, p, cond[1], seed)
            if query is not None:
                r["query"], r["notes_read"] = query, p.count("\n- ") + 1
            r.update(key=key(cond, q), arm=cond[0], mode=cond[1], level=cond[2], qid=q["qid"], kind=q["kind"],
                     pile=ci, **score(q, r["answer"], D["P"]))
            r["source"] = source(q, r, D["P"])
            with lock:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
                f.flush()
                done[r["key"]] = r
            return r

        with ThreadPoolExecutor(workers) as ex:
            big = cond[2] in ("all", 1000, 5000)
            if big and not stub:                      # one pile at a time, its prefix cached by a first call
                by_pile = defaultdict(list)
                for it in todo:
                    by_pile[it[1]].append(it)
                for its in by_pile.values():
                    ask(llm, its[0][2].rsplit("Question: ", 1)[0] + "Question: How many notes are there?\nAnswer "
                        "with a number only.", "nothink", 1)
                    list(ex.map(one, its))
            else:
                list(ex.map(one, todo))
        rs = [done[key(cond, it[0])] for it in items]
        print(f"{cond}: {len(rs)} answers, {100 * np.mean([r['correct'] for r in rs]):.1f}% right, "
              f"{(time.time() - c0) / 60:.1f} min (total {(time.time() - t0) / 60:.1f} min)", flush=True)
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


def get(done, arm, mode, n, kinds):
    kinds = [kinds] if isinstance(kinds, str) else kinds
    return {r["qid"]: r for r in done.values() if r["arm"] == arm and r["mode"] == mode and str(r["level"]) == str(n)
            and r["kind"] in kinds}


def paired(done, mode, a, b, kinds):
    A, B = get(done, "personal", mode, a, kinds), get(done, "personal", mode, b, kinds)
    ids = sorted(set(A) & set(B))
    return [A[i]["correct"] for i in ids], [B[i]["correct"] for i in ids]


def verdict(done, tag, n_hard, pre):
    """Amendment 1, section 7.5: which way of reading is viable, on the hard set, thinking on."""
    phone = "4B" in tag
    big = 100 if phone else 5000
    sel, ret = (get(done, "personal", "think", n, HARD) for n in (10, "search2"))
    full = sel if phone else get(done, "personal", "think", big, HARD)   # the phone never reads 5,000 notes
    if min(len(full), len(sel), len(ret)) < n_hard:
        return "NOT YET: the hard set is not complete at the whole-pile, page-selection and search2 levels.", {}
    ids = sorted(full)
    F, S, R = ([d[i]["correct"] for i in ids] for d in (full, sel, ret))
    f, s, r = 100 * np.mean(F), 100 * np.mean(S), 100 * np.mean(R)
    rf, rr = boot_ratio(F, S), boot_ratio(R, S)
    lat = (pre or {}).get("warm_other_question", {}).get("sec")
    whole = rf[0] >= VIABLE and f >= r - CLOSE_PTS and (lat is None or lat <= LATENCY_S or "4B" in tag)
    selection = rr[0] >= VIABLE and r >= f - CLOSE_PTS
    desc = (f"hard set ({len(ids)} questions), thinking on: whole {big:,}-note pile {f:.1f}%, two-round search {r:.1f}%, "
            f"perfect page selection {s:.1f}%; whole/selected {rf[0]:.2f} [{rf[1]:.2f}, {rf[2]:.2f}], search/selected "
            f"{rr[0]:.2f} [{rr[1]:.2f}, {rr[2]:.2f}] (viable >= {VIABLE}); warm-cache answer at 5,000 notes "
            f"{'not measured' if lat is None else f'{lat:.1f} s'} (line {LATENCY_S:.0f} s)")
    if phone:
        desc = (f"hard set ({len(ids)} questions), thinking on: two-round search {r:.1f}%, perfect page selection "
                f"{s:.1f}%, search/selected {rr[0]:.2f} [{rr[1]:.2f}, {rr[2]:.2f}] (viable >= {VIABLE})")
        v = ("PHONE SELECTION VIABLE: " if rr[0] >= VIABLE else "PHONE SELECTION NOT VIABLE: ") + desc
    elif whole and selection:
        v = ("BOTH VIABLE: " + desc + ". Version 1 = select pages (cheaper, runs on the phone), whole notebook as the "
             "cloud fallback.")
    elif whole:
        v = ("WHOLE NOTEBOOK: " + desc + ". Version 1 in the cloud = the whole notebook in the prompt; two-round search "
             "is not enough.")
    elif selection:
        v = "SELECTION: " + desc + ". Version 1 = select pages first, everywhere; the whole notebook is not viable."
    else:
        v = ("NEITHER: " + desc + ". Reading a 5,000-note notebook is a bottleneck for both; the memory research has a "
             "measured target.")
    return v, {"whole": f, "bm25": r, "selected": s, "ratio_whole": rf, "ratio_bm25": rr, "latency": lat}


def reference_lines(done):
    rw = get(done, "personal", "think", 5000, "reword")
    t5, t1 = paired(done, "think", 5000, 10, "two")
    if len(rw) < COUNTS["reword"] or len(t5) < COUNTS["two"]:
        return "not complete"
    a = boot_mean([r["correct"] for r in rw.values()])
    ratio = boot_ratio(t5, t1)
    return (f"reworded single-hop at 5,000 notes {a[0]:.1f}% [{a[1]:.1f}, {a[2]:.1f}] (first design: pass >= "
            f"{PASS_REWORD:.0f}, stop < {STOP_REWORD:.0f}); two-hop 5,000/10 ratio {ratio[0]:.2f} [{ratio[1]:.2f}, "
            f"{ratio[2]:.2f}] (pass >= {PASS_RATIO}, stop < {STOP_RATIO}); reported, not deciding")


def write(out, tag, done, meta, D):
    pre_path = os.path.join(out, "preflight.json")
    pre = json.load(open(pre_path)) if os.path.exists(pre_path) else None
    n_hard = sum(q["kind"] in HARD for q in D["PQ"])
    L = [f"# Notebook reading at shipping size: {tag}", "",
         "Pre-registration: reports/Notebook reading at shipping size 2026.md (amendment 1 = section 7). Accuracy in % "
         f"with 95% bootstrap intervals over questions ({BOOT:,} resamples).", "", "## Verdict", ""]
    v, R = verdict(done, tag, n_hard, pre)
    L += [f"- {v}", f"- Easy reference: {reference_lines(done)}", "",
          "## Personal notebook by question kind", "",
          "| Kind | Mode | 10 (pages) | 100 | 1,000 | 5,000 | BM25 top-20 | Two-round search |",
          "|---|---|---|---|---|---|---|---|"]
    names_ = {"reword": "A: relation paraphrase, name kept", "same": "A: note's own words", "two": "A: two-hop",
              "upd": "A: latest value", "prev": "A: value before the change", "retract": "A: retracted (says none)",
              "desc": "A3: described person", "who": "A3: who is it", "fname": "A3: first name + detail",
              "amb": "A3: ambiguous first name flagged", "close": "A3: close person with lures"}
    for kind in A_KINDS + A3_KINDS:
        for mode in ("think", "nothink"):
            cells = []
            for n in LEVELS + ["bm25", "search2"]:
                rs = get(done, "personal", mode, n, kind)
                if rs:
                    m, lo, hi = boot_mean([r["correct"] for r in rs.values()])
                    cells.append(f"{m:.1f} [{lo:.0f}-{hi:.0f}] n={len(rs)}")
                else:
                    cells.append("-")
            if any(c != "-" for c in cells):
                L.append(f"| {names_[kind]} | {mode} | " + " | ".join(cells) + " |")
    L += ["", "Updates by form, thinking on (latest value right / stale):", ""]
    form = {q["qid"]: (q.get("form"), q.get("n_changes"), q.get("now")) for q in D["PQ"] if q["kind"] == "upd"}
    for n in LEVELS + ["bm25", "search2"]:
        rs = get(done, "personal", "think", n, "upd")
        if rs:
            parts = []
            for fm in ("explicit", "plain", "correction"):
                x = [r for r in rs.values() if form[r["qid"]][0] == fm]
                parts.append(f"{fm} {100 * np.mean([r['correct'] for r in x]):.0f}% / "
                             f"{100 * np.mean([r['old'] for r in x]):.0f}%")
            two_ch = [r for r in rs.values() if form[r["qid"]][1] == 2]
            if two_ch:
                parts.append(f"two changes {100 * np.mean([r['correct'] for r in two_ch]):.0f}%")
            L.append(f"- {n} notes: " + ", ".join(parts))
    L += ["", "Where wrong answers came from, thinking on (hard set):", ""]
    for n in LEVELS + ["bm25", "search2"]:
        rs = [r for r in get(done, "personal", "think", n, HARD).values() if not r["correct"]]
        if rs:
            c = Counter(r["source"] for r in rs)
            L.append(f"- {n}: {len(rs)} wrong: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    L += ["", "Cost per answer, thinking on (mean prompt tokens, completion tokens, seconds under load; cut off):", ""]
    for n in LEVELS + ["bm25", "search2"]:
        rs = [r for r in done.values() if r["arm"] == "personal" and r["mode"] == "think" and str(r["level"]) == str(n)]
        if rs:
            L.append(f"- {n}: prompt {np.mean([r['prompt_tokens'] or 0 for r in rs]):.0f}, completion "
                     f"{np.mean([r['completion_tokens'] or 0 for r in rs]):.0f}, {np.mean([r['sec'] for r in rs]):.1f} s,"
                     f" cut off {100 * np.mean([r['truncated'] for r in rs]):.1f}%")
    if pre:
        L += ["", f"Preflight (single requests, not scored): {json.dumps(pre)}"]
    L += ["", "## Conflicting world edits (MQuAKE-CF-3k-v2)", "",
          "| Question | Arm, mode | 10 edits | 100 | 1,000 | all | Own facts, no notes |", "|---|---|---|---|---|---|---|"]
    for kind, name in (("w1", "single-hop, new answer"), ("w2", "two-hop, new answer")):
        for arm, mode in (("world", "think"), ("world_noinst", "think"), ("world", "nothink")):
            cells = []
            for n in (10, 100, 1000, "all"):
                rs = get(done, arm, mode, n, kind)
                if rs:
                    m, lo, hi = boot_mean([r["correct"] for r in rs.values()])
                    cells.append(f"{m:.1f} [{lo:.0f}-{hi:.0f}] old {100 * np.mean([r['old'] for r in rs.values()]):.0f}%")
                else:
                    cells.append("-")
            own = get(done, "own", mode, 0, "own") if kind == "w2" and arm == "world" else {}
            cells.append(f"{boot_mean([r['correct'] for r in own.values()])[0]:.1f}" if own else "-")
            if any(c != "-" for c in cells):
                L.append(f"| {name} | {arm}, {mode} | " + " | ".join(cells) + " |")
    L += ["", f"World: {json.dumps(meta)}"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "meta": meta, "verdict": v, "primary": R}, fh, indent=1, default=str)
    qs = {q["qid"]: q for q in D["PQ"]}
    big = 100 if "4B" in tag else 5000
    E = ["# Wrong answers in the primary condition (hard set, thinking on, whole pile), for the hand audit", ""]
    for r in sorted(get(done, "personal", "think", big, HARD).values(), key=lambda r: r["qid"]):
        if not r["correct"]:
            q = qs[r["qid"]]
            E.append(f"- {r['qid']} [{r['source']}] Q: {q['q']} | gold: {q['gold'] or 'none'} | answer: {r['answer']!r}")
    with open(os.path.join(out, "errors.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(E) + "\n")
    return L


def preflight(llm, D):
    """Not scored: does the 5,000-note pile fit, is its prefix reused by a different question, single-request latency."""
    pile = "\n".join("- " + x["text"] for x in D["PP"][5000]["piles"][0])
    p1 = PERSONAL.format(notes=pile, q="How many notes are there?")
    p2 = PERSONAL.format(notes=pile, q="Which note comes first?")
    out = {"served": getattr(llm, "served", None), "model": llm.model}
    for name, p, mode in (("cold", p1, "nothink"), ("warm_other_question", p2, "nothink"), ("warm_same", p2, "nothink"),
                          ("warm_thinking", p2, "think")):
        r = ask(llm, p, mode, 1)
        out[name] = {x: r[x] for x in ("prompt_tokens", "cached_tokens", "completion_tokens", "sec", "truncated")}
    q = D["PQ"][0]
    r = ask(llm, PERSONAL.format(notes="\n".join("- " + x["text"] for x in D["BMP"][q["qid"]]), q="Which note comes "
                                 "first?"), "nothink", 1)
    out["bm25_pages"] = {x: r[x] for x in ("prompt_tokens", "sec")}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    p.add_argument("--base-url", default=None)
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--budget-min", type=float, default=0, help="stop starting new conditions after this many minutes")
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--stub-tag", default="stub", help="stub only: name the run like a model, e.g. stub-4B")
    args = p.parse_args()
    groups, P, PQ, PP, BMP, texts, WQ, WP, clash, bm = build()
    D = {"P": P, "PQ": PQ, "PP": PP, "BMP": BMP, "WQ": WQ, "WP": WP, "BM": bm}
    meta = {"people": len(P), "groups": N_GROUPS, "notes": sum(len(group_notes(g)) for g in groups),
            "pile_notes": {n: [min(map(len, PP[n]["piles"])), max(map(len, PP[n]["piles"]))] for n in LEVELS},
            "pile_chars_5000": len("\n".join("- " + x["text"] for x in PP[5000]["piles"][0])),
            "personal_questions": dict(Counter(q["kind"] for q in PQ)), "hard_set": sum(q["kind"] in HARD for q in PQ),
            "bm25_recall": bm25_recall(D, groups),
            "world_edits": len(texts), "world_edit_clashes_skipped": clash,
            "world_questions": dict(Counter(q["kind"] for q in WQ)),
            "world_piles": {str(n): len(WP[n]["piles"]) for n in (10, 100, 1000, "all")}}
    print(json.dumps(meta), flush=True)
    if args.stub:
        gold = {}
        for q in PQ:
            if q["kind"] == "who":
                gold[q["q"]] = (q["gold"], [n for n in list(P)[:20] if n != q["gold"]])
            elif q["kind"] == "amb":
                gold[q["q"]] = (f"Which {q['first']} do you mean?", q["values"])
            elif q["kind"] == "retract":
                gold[q["q"]] = ("None any more.", [q["old"]])
            else:
                gold[q["q"]] = (q["gold"], [v for v in ATTRS[q["rel"]]["vocab"] if v != q["gold"]])
        for q in WQ:
            gold[q["q"]] = (q["new"][0], [q["old"][0], "unknown"])
        llm, tag = Stub(gold), args.stub_tag
    elif args.base_url:
        llm, tag = G.Endpoint(args.base_url, args.model, 1), args.model.split("/")[-1]
        print(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}", flush=True)
    else:
        raise SystemExit("needs --base-url or --stub")
    out = os.path.join(ROOT, "results", f"notebook_read_{tag}")
    os.makedirs(out, exist_ok=True)
    if args.preflight:
        r = preflight(llm, D)
        print(json.dumps(r, indent=1), flush=True)
        with open(os.path.join(out, "preflight.json"), "w", encoding="utf-8") as fh:
            json.dump(r, fh, indent=1)
        return
    done = run(llm, plan(tag), D, os.path.join(out, "answers.jsonl"), args.workers, args.budget_min, args.stub)
    print("\n".join(write(out, tag, done, meta, D)), flush=True)


def bm25_recall(D, groups):
    """Share of questions whose needed notes are all in the BM25 top 20 (no model; reported with the results)."""
    hit = Counter()
    tot = Counter()
    for q in D["PQ"]:
        if q["gid"] is None:
            continue
        gr = groups[q["gid"]]
        got = {(x["gid"], x["tag"]) for x in D["BMP"][q["qid"]]}
        role = "a" if q["person"] == gr["a"] else "b"
        need = set()
        if q["kind"] in ("two",):
            need = {(gr["gid"], "rel"), (gr["gid"], f"b.{q['rel']}")}
        elif q["kind"] in ("upd", "prev"):
            need = {(gr["gid"], t) for t in ("a." + q["rel"], "chg1", "chg2") if any(
                x["gid"] == gr["gid"] and x["tag"] == t for x in group_notes(gr))}
        elif q["kind"] == "retract":
            need = {(gr["gid"], "ret")}
        elif q["kind"] != "who":
            need = {(gr["gid"], f"{role}.{q['rel']}")}
        if need:
            tot[q["kind"]] += 1
            hit[q["kind"]] += need <= got
    return {k: round(hit[k] / tot[k], 3) for k in tot}


if __name__ == "__main__":
    main()
