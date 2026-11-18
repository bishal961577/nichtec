"""Step 8 (reports/Two ends check 2026.md): a quiz the chat makes by itself, to teach a small model who "she" is.

Self-supervised: no person labels. In a real chat message, a person named outright ("my sister", "your boss",
"Julio") is replaced by the pronoun a writer would use ("she", "him", "their"), marked with brackets; the answer is
the words that were there, written as the writer would name that person. The model sees the conversation before the
message, so it can only answer by finding that person there. If the person does not appear before, the answer is
"unclear" (kept for a capped share of the quiz, so the model also learns to say it cannot tell).

What is hand-written here is only the quiz maker (which words count as a person's name or relation, and which
pronoun replaces them); the model's answers are learned. Pure Python, no third-party modules: it runs on this
machine for checking and inside the Modal training job.

    python brain/whoquiz.py --realtalk DIR --chats 1-4 --show 20      # print 20 quiz items (development chats only)
"""
import argparse
import json
import os
import random
import re
from collections import Counter

FEMALE = {"mom", "mum", "mother", "mommy", "sister", "wife", "girlfriend", "gf", "fiancee", "daughter", "aunt",
          "niece", "grandma", "grandmother", "granny", "stepmom", "stepmother", "stepsister", "mother-in-law",
          "sister-in-law"}
MALE = {"dad", "father", "daddy", "brother", "husband", "boyfriend", "bf", "fiance", "son", "uncle", "nephew",
        "grandpa", "grandfather", "stepdad", "stepfather", "stepbrother", "father-in-law", "brother-in-law"}
PLURAL = {"parents", "grandparents", "friends", "kids", "children", "sisters", "brothers", "cousins", "neighbors",
          "neighbours", "roommates", "coworkers", "co-workers", "colleagues", "in-laws", "family", "folks", "siblings",
          "daughters", "sons", "classmates", "teammates"}
NEUTRAL = {"friend", "best friend", "bff", "bestie", "roommate", "roomie", "flatmate", "housemate", "neighbor",
           "neighbour", "boss", "coworker", "co-worker", "colleague", "manager", "teacher", "professor", "coach",
           "doctor", "therapist", "crush", "ex", "partner", "cousin", "child", "kid", "landlord", "landlady",
           "classmate", "teammate", "buddy", "tutor", "mentor", "supervisor", "trainer", "fiance(e)"}
RELS = sorted(FEMALE | MALE | PLURAL | NEUTRAL, key=len, reverse=True)
MODS = r"(?:(?:best|older|younger|little|big|new|old|ex|step|twin|favorite|favourite|oldest|youngest|baby|closest)\s+)?"
REL_RE = re.compile(r"\b(my|your|ur)\s+(" + MODS + r"(?:" + "|".join(re.escape(r) for r in RELS) + r"))(?=\W|$)", re.I)
SUBJ_BEFORE = re.compile(r"(^|[.!?,;:()\"]\s*|\b(?:and|but|so|because|cause|cuz|coz|when|if|that|since|then|or|while|"
                         r"until|unless|though|although|lol|haha|omg|yeah|yes|no|well|also|plus|now|said|says|think|"
                         r"thinks|guess|hope|bet|know|like)\s+)$", re.I)
COMMON = {"I", "I'm", "Im", "I've", "I'll", "I'd", "OK", "Ok", "Lol", "LOL", "Omg", "OMG", "Haha", "Yes", "No", "Yeah",
          "The", "A", "An", "And", "But", "So", "Oh", "Hey", "Hi", "Hello", "Thanks", "Thank", "Good", "Great", "Well",
          "What", "Why", "How", "When", "Where", "Who", "That", "This", "It", "It's", "Its", "My", "You", "Your",
          "We", "They", "He", "She", "Me", "Us", "Him", "Her", "Them", "Monday", "Tuesday", "Wednesday", "Thursday",
          "Friday", "Saturday", "Sunday", "January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December", "Christmas", "God", "English", "American", "TV", "Netflix",
          "YouTube", "Youtube", "Google", "Instagram", "Twitter", "TikTok", "Tiktok", "Amazon", "Disney", "Mr", "Mrs",
          "Ms", "Dr", "Covid", "COVID", "Just", "Not", "Maybe", "Also", "Same", "Sure", "Wow", "Ah", "Aw", "Awww",
          "Happy", "New", "Year", "Years", "Please", "Sorry", "Nice", "Cool", "Definitely", "Honestly", "Really",
          "Do", "Did", "Does", "Are", "Is", "Was", "Have", "Has", "Can", "Could", "Would", "Should", "Will", "If",
          "Then", "Now", "Today", "Tomorrow", "Yesterday", "Morning", "Night", "Ohh", "Ooh", "Hmm", "Nah", "Yep"}
VERBS = {"is", "was", "are", "were", "has", "have", "had", "does", "did", "do", "will", "would", "can", "could",
         "should", "might", "must", "said", "says", "told", "tells", "got", "gets", "went", "goes", "wants", "wanted",
         "likes", "liked", "loves", "loved", "thinks", "thought", "knows", "knew", "made", "makes", "came", "comes",
         "called", "calls", "asked", "asks", "just", "always", "never", "also", "still", "really", "and", "lives",
         "lived", "works", "worked", "bought", "took", "takes", "gave", "gives", "left", "moved", "started", "keeps",
         "kept", "doesn't", "didn't", "isn't", "wasn't", "won't", "can't", "hasn't", "haven't", "wouldn't", "seems",
         "seemed", "loves", "hates", "hated", "needs", "needed", "tried", "tries", "found", "finally", "used", "is'nt"}
SG_VERBS = {"is", "was", "has", "does", "doesn't", "isn't", "wasn't", "hasn't", "says", "goes", "wants", "likes",
            "loves", "thinks", "knows", "makes", "comes", "calls", "asks", "lives", "works", "takes", "gives", "keeps",
            "seems", "hates", "needs", "tries", "gets", "tells"}
AUX = {"would", "did", "does", "do", "has", "have", "is", "was", "are", "were", "can", "will", "should", "could",
       "had", "didn't", "doesn't", "isn't", "wasn't", "won't", "can't"}
PLACES = {"Georgia", "Savannah", "Florence", "Austin", "Dallas", "Houston", "Orlando", "Jackson", "Charlotte",
          "Victoria", "Sydney", "Paris", "Chelsea", "Lincoln", "Madison", "Phoenix", "Carolina", "Virginia", "Augusta",
          "Jordan", "Chad", "Israel", "India", "Asia", "China", "Christ", "Florida", "Albany", "Aurora", "Beverly",
          "Brooklyn", "Camden", "Cleveland", "Denver", "Eugene", "Helena", "Kent", "Lourdes", "Marion", "Memphis",
          "Nevada", "Sierra", "Tyler", "Troy", "Vienna", "Warren", "Montana", "Dakota", "Columbia", "Santa", "Rica"}
INSTITUTION = {"Church", "Street", "Park", "Hotel", "Bank", "University", "College", "School", "Hall", "Center",
               "Centre", "Road", "Avenue", "Lake", "Mountain", "River", "Beach", "City", "County", "Island", "Islands",
               "Chase", "Cathedral", "Hospital", "Airport", "Station", "Square", "Bridge", "Valley", "Games", "Show"}
PRONOUNS = {"f": ("she", "her", "her"), "m": ("he", "him", "his"), "p": ("they", "them", "their")}
PRON_WORDS = {"she", "her", "hers", "he", "him", "his", "they", "them", "their"}


def norm_rel(possessor, rel, writer, partner):
    """The answer as the writer names the person: "my sister", "Ana's boss"."""
    rel = re.sub(r"\s+", " ", rel.lower()).strip()
    return f"my {rel}" if possessor.lower() == "my" else f"{partner}'s {rel}"


def head(rel):
    return rel.lower().split()[-1]


NAMES = {}                     # first name -> "f", "m" or "fm"; filled by load_names()


def load_names(zip_path):
    """The Kantrowitz-Ross names corpus (1991; free to use with credit; nltk_data's names.zip): 7,944 first names
    marked female or male."""
    import zipfile
    z = zipfile.ZipFile(zip_path)
    for g in ("female", "male"):
        for n in z.read(f"names/{g}.txt").decode("utf-8", "replace").split():
            NAMES[n] = NAMES.get(n, "") + g[0]
    return NAMES


def names_in(conv, speakers):
    """First names (from the names corpus) used in the conversation, minus the two chatters' own names and names that
    are also everyday words there ("will", "grace", "may" written in lower case)."""
    assert NAMES, "call load_names() first"
    low = " ".join(m["text"] for m in conv)
    lower_words = set(re.findall(r"\b[a-z]+\b", low))
    seen = set(re.findall(r"\b([A-Z][a-z]+)\b", low))
    bad = {w for s in speakers for w in s.split()} | COMMON
    return {w for w in seen if w in NAMES and w not in bad and w.lower() not in lower_words}


def gender_of(key, conv, t, window=3):
    """Data-derived gender for a name or neutral relation: pronouns in the messages right after it is named."""
    f = m = 0
    for j in range(max(0, t - window), min(len(conv), t + window + 1)):
        low = conv[j]["text"].lower()
        if key.lower() not in low:
            continue
        rest = low.split(key.lower(), 1)[1][:200] + " " + (conv[j + 1]["text"].lower()[:200] if j + 1 < len(conv) else "")
        f += len(re.findall(r"\b(she|her|hers)\b", rest))
        m += len(re.findall(r"\b(he|him|his)\b", rest))
    return "f" if f > m else "m" if m > f else "p"


def spans(text, names):
    out = []
    for mt in REL_RE.finditer(text):
        out.append((mt.start(), mt.end(), "rel", mt.group(1), mt.group(2)))
    for nm in names:
        for mt in re.finditer(r"\b" + re.escape(nm) + r"(\s+[A-Z][a-z]+)?\b", text):
            if any(a <= mt.start() < b for a, b, *_ in out):
                continue
            before = text[:mt.start()]
            if re.search(r"\b(the|The)\s+$", before) or re.search(r"[A-Z][A-Za-z]*\s+$", before):
                continue                                        # "the Grace Hotel", "Los Angeles"
            after = text[mt.end():]
            if (re.search(r"(,|\b(hey|hi|hello|thanks|thank you|bye|dear|you|too|yes|no|sure|ok|okay|welcome))\s*$",
                          before, re.I) and re.match(r"\s*([!?.,:;)]|$)", after)) or re.match(r"\s*,", after) and not before.strip():
                continue                                        # someone addressed ("You too, Liza!"): the partner
            if re.search(r"\b(in|at|from|visit|visited|visiting|near|around|across|through|into)\s+$", before, re.I):
                continue                                        # mostly places ("in Orlando")
            if re.search(r"\byou\s+$", before, re.I) or re.match(r"\s*[!?]", after) or not after.strip():
                continue                                        # "talking to you Kate", "plans for today Kate?"
            sur = (mt.group(1) or "").strip()
            if nm in PLACES or (sur and sur in INSTITUTION):
                continue                                        # "I love Georgia", "Christ Church"
            word = nm + (" " + sur if sur and sur not in COMMON else "")
            out.append((mt.start(), mt.start() + len(word), "name", "", word))
    return sorted(out)


def same_person(sp, writer, partner, earlier):
    """Does this person appear in the earlier messages (or earlier in this one)? earlier: [(speaker, text)]."""
    kind, pos, word = sp[2], sp[3].lower(), sp[4]
    for spk, txt in earlier:
        if kind == "name":
            if re.search(r"\b" + re.escape(word) + r"\b", txt):
                return True
            continue
        h = re.escape(head(word))
        mine = re.search(r"\b(my)\s+" + MODS + h + r"\b", txt, re.I)
        yours = re.search(r"\b(your|ur)\s+" + MODS + h + r"\b", txt, re.I)
        if pos == "my" and ((spk == writer and mine) or (spk == partner and yours)):
            return True
        if pos in ("your", "ur") and ((spk == writer and yours) or (spk == partner and mine)):
            return True
    return False


def make_item(conv, t, sp, writer, partner, k, rng):
    m = conv[t]
    text = m["text"]
    a, b, kind, pos, word = sp
    if kind == "name":
        g = {"f": "f", "m": "m"}.get(NAMES.get(word.split()[0], ""), "") or gender_of(word.split()[0], conv, t)
    else:
        g = ("f" if head(word) in FEMALE else "m" if head(word) in MALE else "p" if head(word) in PLURAL
             else gender_of(head(word), conv, t))
    poss = text[b:b + 2] in ("'s", "’s")
    before = text[:a]
    nxt = (re.match(r"\s*([A-Za-z']+)", text[b:]) or [None, ""])[1].lower()
    prv = (re.search(r"([A-Za-z']+)\s*$", before) or [None, ""])[1].lower()
    subj = (nxt in VERBS or prv in AUX or bool(SUBJ_BEFORE.search(before))) and not re.search(r"\bme and\s*$", before)
    form = 2 if poss else (0 if subj else 1)
    if g == "p" and kind == "rel" and head(word) not in PLURAL and (nxt in SG_VERBS or (subj and prv in SG_VERBS)):
        return None                                     # "they has": the quiz would teach bad grammar
    pr = PRONOUNS[g][form]
    if a == 0 or re.search(r"[.!?]\s*$", before):
        pr = pr.capitalize()
    new = before + "⟦" + pr + "⟧" + text[b + (2 if poss else 0):]
    answer = word if kind == "name" else norm_rel(pos, word, writer, partner)
    earlier = [(c["speaker"], c["text"]) for c in conv[max(0, t - k):t]] + [(writer, text[:a])]
    found = same_person(sp, writer, partner, earlier)
    return {"window": [(c["speaker"], c["text"]) for c in conv[max(0, t - k):t]], "writer": writer, "partner": partner,
            "message": new, "pronoun": pr, "answer": answer if found else "unclear", "recoverable": found}


def quiz(convs, k=10, unclear_share=0.15, seed=8):
    """convs: [(speaker names (2), [{"speaker", "text"}...])]. One item per eligible span; unclear items capped."""
    rng = random.Random(seed)
    items, unclear = [], []
    for speakers, conv in convs:
        names = names_in(conv, speakers)
        for t, m in enumerate(conv):
            writer = m["speaker"]
            partner = speakers[1] if writer == speakers[0] else speakers[0]
            if any(w in m["text"] for w in ("⟦", "⟧")):
                continue
            for sp in spans(m["text"], names):
                it = make_item(conv, t, sp, writer, partner, k, rng)
                if it:
                    (items if it["recoverable"] else unclear).append(it)
    rng.shuffle(items)
    plural = [q for q in items if q["pronoun"].lower() in ("they", "them", "their")]
    single = [q for q in items if q["pronoun"].lower() not in ("they", "them", "their")]
    items = single + plural[:len(single) // 3]          # plurals at most a quarter: real "she/he" is commoner
    rng.shuffle(unclear)
    cap = int(unclear_share / (1 - unclear_share) * len(items))
    out = items + unclear[:cap]
    rng.shuffle(out)
    return out


PROMPT = ("A chat between {writer} and {partner}. The messages before the latest one:\n{window}\n\n"
          "The latest message, from {writer}:\n{message}\n\n"
          "Who is ⟦{pronoun}⟧ in the latest message? Answer with the words {writer} would use for them: their name if "
          "the chat gives one, else for example \"my sister\" or \"{partner}'s boss\". If you cannot tell, answer "
          "\"unclear\". Answer with those words only.")


def prompt_of(it):
    win = "\n".join(f"{s}: {t}" for s, t in it["window"]) or "(none)"
    return PROMPT.format(writer=it["writer"], partner=it["partner"], window=win, message=it["message"],
                         pronoun=it["pronoun"])


def answer_match(pred, gold):
    """Lenient match of a who-answer: case, 'the', a trailing period and 'mom'/'mum'-style spellings ignored."""
    def n(s):
        s = s.strip().strip('."').lower()
        s = re.sub(r"^the\s+", "", s)
        s = s.replace("mum", "mom").replace("’", "'")
        return re.sub(r"\s+", " ", s)
    return n(pred) == n(gold)


# ---- sources (both public; read from a local copy) ----

def realtalk_convs(data_dir, chats):
    out = []
    for c in chats:
        f = [x for x in os.listdir(data_dir) if x.startswith(f"Chat_{c}_") and x.endswith(".json")]
        assert len(f) == 1, (c, f)
        d = json.load(open(os.path.join(data_dir, f[0]), encoding="utf-8"))
        up = lambda x: x.strip()[:1].upper() + x.strip()[1:]
        names = {d["name"]["speaker_1"].strip().lower(): up(d["name"]["speaker_1"]),
                 d["name"]["speaker_2"].strip().lower(): up(d["name"]["speaker_2"])}
        conv = []
        for k in sorted((k for k in d if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1])):
            for m in d[k]:
                if (m.get("clean_text") or m.get("text") or "").strip():
                    conv.append({"speaker": names[m["speaker"].strip().lower()],
                                 "text": (m.get("clean_text") or m.get("text")).strip()})
        out.append((tuple(names.values()), conv))
    return out


def ed_convs(csv_path, limit=None, seed=8):
    """EmpatheticDialogues (Rashkin et al. 2019, CC BY-NC 4.0): commas are stored as _comma_. Speakers are named
    "Speaker A" (who opens) and "Speaker B"."""
    rows = open(csv_path, encoding="utf-8").read().split("\n")[1:]
    by = {}
    for r in rows:
        f = r.split(",")
        if len(f) < 6:
            continue
        by.setdefault(f[0], []).append((int(f[1]), f[4], f[5].replace("_comma_", ",").strip()))
    convs = []
    for cid in sorted(by):
        us = sorted(by[cid])
        first = us[0][1]
        conv = [{"speaker": "Speaker A" if s == first else "Speaker B", "text": t} for _, s, t in us if t]
        convs.append((("Speaker A", "Speaker B"), conv))
    if limit:
        random.Random(seed).shuffle(convs)
        convs = convs[:limit]
    return convs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtalk", help="REALTALK data/ folder")
    ap.add_argument("--chats", default="1-4")
    ap.add_argument("--ed", help="an EmpatheticDialogues csv (train or valid only: test is step 8's exam)")
    ap.add_argument("--names", required=True, help="nltk_data's names.zip")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--out")
    a = ap.parse_args()
    load_names(a.names)
    convs = []
    if a.realtalk:
        lo, hi = map(int, a.chats.split("-"))
        convs += realtalk_convs(a.realtalk, range(lo, hi + 1))
    if a.ed:
        assert "test" not in os.path.basename(a.ed), "the test split is step 8's exam"
        convs += ed_convs(a.ed)
    Q = quiz(convs)
    print(len(Q), "items;", sum(not q["recoverable"] for q in Q), "unclear;",
          Counter(q["pronoun"].lower() for q in Q).most_common())
    for q in Q[:a.show]:
        print("----\n" + q["message"][:300], "\n  ->", q["answer"])
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for q in Q:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
