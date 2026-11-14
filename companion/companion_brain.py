"""
companion_brain.py

A memory "brain" for an AI companion. It learns from every message for as long as the person uses it, never forgets
by accident, forgets exactly when asked, and keeps a state that carries over between conversations.

From the nichtec research project (November 2026). Read this header first, then the classes in order:
    Journal -> Notebook -> BM25 -> ExactMemory -> TasteMap / PersonalTaste -> Brain
Run `python companion_brain.py` for a demo that checks every promise below with numbers.
Needs Python 3.10+ and numpy. Your own model plugs in through one function:  llm(prompt: str) -> str

---------------------------------------------------------------------------------------------------------------------
WHY A COMPANION'S MEMORY USUALLY STRUGGLES (what our tests showed)

  1. Summaries lose details. Keep every message raw, and build everything else from it.
  2. Pasting many memories into the prompt hurts. On a small model, pasting 10 / 100 / 1,000 facts gave 43% / 27% /
     13% right answers; pasting only the one relevant fact gave 92% whatever the pile size. Select first, then read.
     (Measured on a 0.5B model. Larger models read long prompts better; our test on 4B and 27B is still pending.)
  3. Training memories into the model's weights forgets old ones and cannot be undone cleanly. Keep facts outside
     the weights, where deleting is exact.
  4. A language model's general knowledge is not a map of one person's taste. On 300 real MovieLens users, a 27B model
     reading a person's ratings scored AUC 0.635, below plain "recommend what is popular" (0.679). A taste map learned
     from many people, plus the person's own position on it solved from their own ratings, scored 0.740.
  5. Updates and mix-ups are where memory breaks. Keep every version of a fact with its date, and key facts by person.

---------------------------------------------------------------------------------------------------------------------
WHAT IS IN THIS FILE, AND HOW SURE WE ARE

  Tested in our research:
    - ExactMemory (the exact, order-free, deletable associative memory). Written into a frozen 6B language model
      with 2,764 real Wikidata changes, it answered direct questions 95.7-95.9% right, kept its earliest facts 97%,
      and left 98.7% of unrelated knowledge unchanged. Deleted facts fell to chance level (0.5B test). It is the
      same memory whatever order facts arrive in, by construction. Every unwanted change was a name collision
      ("Francis" inside "Francis II").
    - PersonalTaste on a TasteMap: the 0.740 result above (same method and settings as here).
    - Selecting the relevant notes instead of pasting everything: the 92% result above.
  Designed here, not yet measured:
    - fact extraction from messy chat ("she", "my boss", nicknames). This is where companions break: test it first;
    - two-round search (search, the model asks for one follow-up search, search again);
    - the nightly "sleep" that pre-joins linked facts into derived pages, so a two-step question becomes one lookup;
    - the private scratchpad: notes the AI keeps for itself across conversations.
  Not solved by anyone yet, including us:
    - making a model combine two NEW facts in its head the way it combines facts it learned in training. With
      written-out steps our memory managed 9.4% on two-step questions. Derived pages and two-round search work around
      this; they do not solve it.

---------------------------------------------------------------------------------------------------------------------
THE FOUR RULES THE CODE FOLLOWS

  1. The journal (every raw message) is the single source of truth. Everything else is derived from it.
  2. Everything derived keeps receipts: the ids of the messages it came from.
  3. Deleting a message deletes every fact and private note it supported, then rebuilds the derived pages, so
     nothing worked out from a deleted message survives.
  4. Learning means adding up sums, and sums don't care about order. The same memories in any order give the same
     brain, and recomputing without a memory gives exactly the brain that never saw it.

---------------------------------------------------------------------------------------------------------------------
HOW TO PLUG IT IN

    brain = Brain(llm=my_model)                       # my_model(prompt: str) -> str
    reply = brain.respond("My neighbour Bilal just got a goldfish!")
    brain.sleep()                                     # once a night, or whenever the app is idle
    brain.forget_message(message_id)                  # "forget what I said"
    brain.forget_about("Bilal Barbosa")               # "forget everything about Bilal"
    print(brain.what_i_know())                        # show the user everything it knows, with receipts
    brain.save("user_123.json")                       # state survives restarts
    brain = Brain.load("user_123.json", llm=my_model)

BEFORE YOU TRUST IT, MEASURE ON YOUR OWN USERS' CHATS
    - extraction: are "she", "my sister", nicknames and corrections turned into the right facts?
    - recall: at your heaviest users' notebook sizes, is the needed fact among the notes recall() returns?
    - deletion: after forget_message(), search the saved file for the deleted words. They must be gone.
"""
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np

STOP = set("a an the is are was were be been of to in on at for with and or but my your his her their its it this that "
           "what which who whom whose where when how does do did has have had i you he she they we me us them".split())


def words(text):
    """Lower-case words without common filler words; a trailing plural 's' is dropped so 'pets' finds 'pet'."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w not in STOP:
            out.append(w)
    return out


def day(t):
    return time.strftime("%Y-%m-%d", time.localtime(t))


# =====================================================================================================================
# 1. JOURNAL: every raw message, the single source of truth
# =====================================================================================================================
@dataclass
class Message:
    id: int
    time: float
    role: str          # "user" or "assistant"
    text: str


class Journal:
    def __init__(self):
        self.messages: dict[int, Message] = {}
        self.next_id = 1

    def add(self, role, text, when=None):
        m = Message(self.next_id, when if when is not None else time.time(), role, text)
        self.messages[m.id] = m
        self.next_id += 1
        return m.id

    def delete(self, mid):
        """Hard delete: the text is gone, not hidden."""
        return self.messages.pop(mid, None) is not None

    def recent(self, n):
        return [self.messages[i] for i in sorted(self.messages)[-n:]]


# =====================================================================================================================
# 2. NOTEBOOK: facts with dates, every version kept, each with receipts
# =====================================================================================================================
@dataclass
class Fact:
    subject: str               # a person (full name) or "user"
    relation: str              # short and fixed: "lives in", "works as", "has pet", "neighbour", ...
    value: str | None          # None means retracted: "no longer has a car"
    time: float                # when it was said
    sources: list[int]         # receipts: ids of the messages this fact came from
    derived: bool = False      # True for pages worked out during sleep()
    via: str | None = None     # for derived pages: the person in the middle of the chain
    seq: int = 0               # order of arrival, to break ties between facts said at the same moment


class Notebook:
    def __init__(self):
        self.facts: list[Fact] = []
        self._seq = 0

    def add(self, f: Fact):
        self._seq += 1
        f.seq = self._seq
        self.facts.append(f)
        return f

    def versions(self, subject, relation):
        return sorted((f for f in self.facts if not f.derived and f.subject == subject and f.relation == relation),
                      key=lambda f: (f.time, f.seq))

    def current(self):
        """The newest version of every (person, relation), plus the derived pages. Retractions are kept: knowing
        'no longer has a car' matters as much as knowing the car."""
        latest = {}
        for f in sorted(self.facts, key=lambda f: (f.time, f.seq)):
            if not f.derived:
                latest[(f.subject, f.relation)] = f
        return list(latest.values()) + [f for f in self.facts if f.derived]

    def subjects(self):
        return sorted({f.subject for f in self.facts if not f.derived})

    def line(self, f: Fact):
        """One fact as the model will read it, with earlier values so it never confuses old with new."""
        value = f.value if f.value is not None else "none any more (retracted)"
        text = f"{f.subject} | {f.relation} | {value} (as of {day(f.time)})"
        if f.derived:
            return text + f" [worked out: through {f.via}]"
        before = [v.value or "none" for v in self.versions(f.subject, f.relation)[:-1]]
        return text + (f" [earlier: {', '.join(before)}]" if before else "")

    def drop_sources(self, mids):
        """Remove deleted messages from every receipt; a fact with no receipt left is removed."""
        kept, removed = [], 0
        for f in self.facts:
            f.sources = [m for m in f.sources if m not in mids]
            if f.sources:
                kept.append(f)
            else:
                removed += 1
        self.facts = kept
        return removed


# =====================================================================================================================
# 3. SEARCH: BM25 over the notebook (plain word matching that weighs rare words more)
# =====================================================================================================================
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [words(d) for d in docs]
        self.tf = [Counter(t) for t in self.toks]
        n = max(len(docs), 1)
        df = Counter(w for t in self.toks for w in set(t))
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.avg = max(sum(map(len, self.toks)) / n, 1.0)
        self.k1, self.b = k1, b

    def top(self, query, k):
        q = words(query)
        scored = []
        for i, (tf, t) in enumerate(zip(self.tf, self.toks)):
            s = sum(self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * len(t) / self.avg))
                    for w in q if w in tf)
            if s > 0:
                scored.append((-s, i))
        return [i for _, i in sorted(scored)[:k]]


# =====================================================================================================================
# 4. EXACT MEMORY: the associative memory from our research (order-free, exactly deletable, with a certificate)
# =====================================================================================================================
class ExactMemory:
    """Stores key -> value pairs (vectors) in one matrix W, chosen as the best least-squares fit to all pairs:

        W = argmin  sum_i |W k_i - v_i|^2 + lam |W|^2  =  B (A + lam I)^-1,   A = sum_i k_i k_i^T,   B = sum_i v_i k_i^T

    A and B are sums, so the order pairs arrive in never matters. solve() recomputes them from the stored pairs in a
    fixed order, so W is bit-for-bit the same whatever the arrival order, and deleting a pair gives bit-for-bit the
    memory that never had it. certificate() checks that every stored pair still reads back. If not, the memory is
    full (too many pairs for the key size): add key dimensions or split the memory.

    In our research the keys were a frozen language model's internal state at a person's name (plus a code for the
    relation), the values were what one layer had to add for the model to say the stored answer, and W was written
    into that layer. It runs standalone here so you can see the guarantees: feed it any embedding. Use it if your
    brain stores memories as vectors inside or beside a model; a companion that keeps facts as text needs only the
    Notebook.
    """

    def __init__(self, key_dim, value_dim, lam=1e-6):
        self.key_dim, self.value_dim, self.lam = key_dim, value_dim, lam
        self.pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.W = None

    def add(self, pid, key, value):
        self.pairs[str(pid)] = (np.asarray(key, float), np.asarray(value, float))
        self.W = None

    def delete(self, pid):
        self.W = None
        return self.pairs.pop(str(pid), None) is not None

    def solve(self):
        A = self.lam * np.eye(self.key_dim)
        B = np.zeros((self.value_dim, self.key_dim))
        for pid in sorted(self.pairs):                    # a fixed order: arrival order can't change a single bit
            k, v = self.pairs[pid]
            A += np.outer(k, k)
            B += np.outer(v, k)
        self.W = np.linalg.solve(A, B.T).T                # A is symmetric, so W = B A^-1
        return self.W

    def read(self, key):
        if self.W is None:
            self.solve()
        return self.W @ np.asarray(key, float)

    def nearest(self, key):
        """The stored pair whose value is closest to what the memory reads out for this key."""
        out = self.read(key)
        return min(self.pairs, key=lambda p: np.linalg.norm(self.pairs[p][1] - out)) if self.pairs else None

    def certificate(self, tol=0.05):
        if self.W is None:
            self.solve()
        errs = [np.linalg.norm(self.W @ k - v) / max(np.linalg.norm(v), 1e-12) for k, v in self.pairs.values()]
        worst = max(errs, default=0.0)
        return {"pairs": len(self.pairs), "key_dim": self.key_dim, "worst_relative_error": float(worst),
                "ok": bool(worst <= tol)}


# =====================================================================================================================
# 5. TASTE: a map learned slowly from many people + each person's position, solved exactly from their own ratings
# =====================================================================================================================
class TasteMap:
    """The slow layer. A map of taste learned from many people's ratings with alternating least squares (32
    dimensions, regularisation 0.1 per rating, 15 rounds: our MovieLens settings). Retrain it now and then (say
    monthly) from your whole user base. It is not order-free and does not need to be: it learns what kinds of taste
    exist, not who anyone is.

    New items nobody has rated yet have no place on the map. Placing them from their text description with a small
    model trained to predict item vectors is a known approach (e.g. AlphaRec), not included here."""

    def __init__(self, dim=32, lam=0.1, iters=15, seed=0):
        self.dim, self.lam, self.iters, self.seed = dim, lam, iters, seed
        self.mu, self.items = 0.0, {}

    def fit(self, ratings):
        """ratings: (user, item, value) triples from many people."""
        users = sorted({u for u, _, _ in ratings}, key=str)
        items = sorted({i for _, i, _ in ratings}, key=str)
        ui, ii = {u: n for n, u in enumerate(users)}, {i: n for n, i in enumerate(items)}
        self.mu = float(np.mean([r for _, _, r in ratings]))
        by_u, by_i = defaultdict(list), defaultdict(list)
        for u, i, r in ratings:
            by_u[ui[u]].append((ii[i], r - self.mu))
            by_i[ii[i]].append((ui[u], r - self.mu))
        rng = np.random.default_rng(self.seed)
        U = rng.normal(0, 0.1, (len(users), self.dim))
        V = rng.normal(0, 0.1, (len(items), self.dim))
        for _ in range(self.iters):
            U = self._solve(V, by_u, len(users))
            V = self._solve(U, by_i, len(items))
        self.items = {i: V[ii[i]] for i in items}
        return self

    def _solve(self, F, rows, n):
        out = np.zeros((n, self.dim))
        for r, xs in rows.items():
            idx = np.array([j for j, _ in xs])
            y = np.array([v for _, v in xs])
            out[r] = np.linalg.solve(F[idx].T @ F[idx] + self.lam * len(xs) * np.eye(self.dim), F[idx].T @ y)
        return out


class PersonalTaste:
    """The fast layer, one per person: where they sit on the map, solved exactly from their own ratings:

        w = (sum x x^T + lam * n * I)^-1  sum x (rating - mu)      (x = an item's position on the map)

    Sums again, so the order is irrelevant and forgetting a rating gives exactly the taste that never had it. It is
    32 numbers per person, small enough to live on a phone. It learns from the first rating; no retraining."""

    def __init__(self, taste_map: TasteMap):
        self.map = taste_map
        self.ratings: dict[str, tuple[str, float]] = {}

    def rate(self, rid, item, value):
        if item not in self.map.items:
            return False                                   # not on the map yet (see TasteMap's note on new items)
        self.ratings[str(rid)] = (item, float(value))
        return True

    def forget(self, rid):
        return self.ratings.pop(str(rid), None) is not None

    def vector(self):
        d = self.map.dim
        A, b = np.zeros((d, d)), np.zeros(d)
        for rid in sorted(self.ratings):
            item, y = self.ratings[rid]
            x = self.map.items[item]
            A += np.outer(x, x)
            b += x * (y - self.map.mu)
        return np.linalg.solve(A + self.map.lam * max(len(self.ratings), 1) * np.eye(d), b)

    def score(self, item):
        x = self.map.items.get(item)
        return None if x is None else float(self.map.mu + x @ self.vector())

    def rank(self, items):
        w = self.vector()
        known = [i for i in items if i in self.map.items]
        return sorted(known, key=lambda i: -(self.map.items[i] @ w))


# =====================================================================================================================
# 6. THE BRAIN: learning, recall, private state, sleep, forgetting, saving
# =====================================================================================================================
EXTRACT_HEADER = "You keep a notebook about the people in a user's life."
EXTRACT_PROMPT = EXTRACT_HEADER + """
Known people: {people}
Use these relations when they fit: {relations}

Recent conversation:
{recent}

New message from the user:
{message}

Write one JSON object per line for each fact the NEW message states:
{{"subject": "...", "relation": "...", "value": "...", "kind": "fact"}}
- subject: a person's full name from the known people when the message means one of them ("she", "my sister",
  a nickname); "user" for the user; a new full name for someone new.
- kind "retraction" when something stopped being true ("sold the car", "no longer has a cat"); value can be null.
- A relation between two people (neighbour, sister, boss, ...) has the other person's full name as its value.
If the message states no facts, write NONE."""
RELATIONS = ["name", "lives in", "works as", "has pet", "drives", "favourite food", "allergic to", "plays",
             "learning", "hobby", "birthday", "partner", "sister", "brother", "mother", "father", "child", "friend",
             "neighbour", "boss", "colleague"]
FOLLOWUP_HEADER = "You are answering a question from a user's notebook."
FOLLOWUP_PROMPT = FOLLOWUP_HEADER + """ A search of the notebook returned these notes:
{notes}

Question: {question}
If these notes are enough to answer the question, reply DONE. Otherwise reply with one short search query (a few
words, for example a person's name and what you need to know) that would find the missing note. Reply with DONE or
the query only."""
REPLY_RULES = ("Reply to the user. To remember something privately for later (a plan, a promise, a guess to check), "
               "add a line <private>...</private>. It is removed before the user sees your reply.")
PRIVATE = re.compile(r"<private>(.*?)</private>", re.S)


def parse_facts(reply):
    """Read the extractor's JSON lines; anything malformed is skipped, never guessed."""
    out = []
    for line in reply.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        s, r = str(d.get("subject") or "").strip(), str(d.get("relation") or "").strip().lower()
        if not s or not r:
            continue
        v = d.get("value")
        retracted = str(d.get("kind", "fact")).lower() == "retraction"
        out.append((s, r, None if retracted or v is None else str(v).strip()))
    return out


class Brain:
    def __init__(self, llm=None, persona="You are a warm, honest companion who remembers what the user tells you.",
                 taste_map=None):
        self.llm = llm
        self.persona = persona
        self.journal = Journal()
        self.notebook = Notebook()
        self.private: list[dict] = []      # notes the AI keeps for itself: {"text", "time", "sources"}
        self.taste = PersonalTaste(taste_map) if taste_map is not None else None

    # ---------------------------------------------------------------- learning: every message, immediately
    def hear(self, text, when=None):
        """Store a user message and the facts it states. Returns the message id (keep it for forget_message)."""
        mid = self.journal.add("user", text, when)
        if self.llm is None:
            return mid
        prompt = EXTRACT_PROMPT.format(
            people=", ".join(s for s in self.notebook.subjects() if s != "user") or "(none yet)",
            relations=", ".join(RELATIONS),
            recent="\n".join(f"{m.role}: {m.text}" for m in self.journal.recent(7)[:-1]) or "(start)",
            message=text)
        when = self.journal.messages[mid].time
        for s, r, v in parse_facts(self.llm(prompt)):
            self.notebook.add(Fact(self._canon(s), r, self._canon(v) if v else v, when, [mid]))
        return mid

    def _canon(self, name):
        """Spell a name the way the notebook already does ('ana costa' -> 'Ana Costa')."""
        for s in self.notebook.subjects():
            if s.lower() == name.lower():
                return s
        return name

    # ---------------------------------------------------------------- recall: select first, then read
    def _named(self, text):
        """People a text mentions: full names exactly; a first name alone brings every person with that first name."""
        low = f" {' '.join(re.findall(r'[a-z0-9]+', text.lower()))} "
        found = set()
        for s in self.notebook.subjects():
            if s == "user":
                if re.search(r"\b(i|me|my|mine|myself)\b", text.lower()):
                    found.add(s)
                continue
            full = " ".join(re.findall(r"[a-z0-9]+", s.lower()))
            if f" {full} " in low or f" {full.split()[0]} " in low:
                found.add(s)
        return found

    def _ambiguous(self, text):
        firsts = defaultdict(set)
        for s in self._named(text):
            firsts[s.split()[0]].add(s)
        return {f: sorted(v) for f, v in firsts.items() if len(v) > 1 and f.lower() != "user"}

    def recall(self, question, k=12):
        """The notes worth reading for this question:
        1. the whole page of every person the question names (lookup by name: the reliable part of our memory);
        2. the k notes a word search finds;
        3. if a model is plugged in, one follow-up search the model asks for (for two-step questions)."""
        facts = self.notebook.current()
        if not facts:
            return []
        chosen = {}

        def take(fs):
            for f in fs:
                chosen[id(f)] = f

        names = self._named(question)
        take(f for f in facts if f.subject in names)
        bm = BM25([self.notebook.line(f) for f in facts])
        first = [facts[i] for i in bm.top(question, k)]
        take(first)
        if self.llm is not None:
            reply = self.llm(FOLLOWUP_PROMPT.format(notes="\n".join("- " + self.notebook.line(f) for f in first) or
                                                    "- (nothing)", question=question))
            q2 = (reply.strip().splitlines() or ["DONE"])[0].strip()
            if q2 and "DONE" not in q2.upper():
                take(facts[i] for i in bm.top(q2, k))
                more = self._named(q2)
                take(f for f in facts if f.subject in more)
        return sorted(chosen.values(), key=lambda f: (f.subject, f.relation, f.derived))

    # ---------------------------------------------------------------- talking, with private state across turns
    def build_prompt(self, user_mid, recent_turns=12):
        msg = self.journal.messages[user_mid]
        notes = self.recall(msg.text)
        L = [self.persona, "", "What you know about the user's life (newest version of each fact; dates show when it "
                               "was said):"]
        L += [f"- {self.notebook.line(f)}" for f in notes] or ["- (nothing relevant found)"]
        for first, people in self._ambiguous(msg.text).items():
            L.append(f"Careful: '{first}' could mean {', '.join(people)}. Ask which one if it matters.")
        if self.private:
            L += ["", "Your private notes to yourself from earlier (the user cannot see these):"]
            L += [f"- ({day(p['time'])}) {p['text']}" for p in self.private[-20:]]
        L += ["", "Recent conversation:"]
        L += [f"{m.role}: {m.text}" for m in self.journal.recent(recent_turns) if m.id != user_mid]
        L += [f"user: {msg.text}", "", REPLY_RULES]
        return "\n".join(L)

    def said(self, reply, in_reply_to, when=None):
        """Store the AI's reply. Private notes are kept as state with receipts and removed from what the user sees."""
        notes = [n.strip() for n in PRIVATE.findall(reply) if n.strip()]
        visible = PRIVATE.sub("", reply).strip()
        rid = self.journal.add("assistant", visible, when)
        for n in notes:
            self.private.append({"text": n, "time": self.journal.messages[rid].time, "sources": [in_reply_to, rid]})
        return visible

    def respond(self, text):
        """One turn: hear, remember, recall, reply. Returns the reply the user sees."""
        mid = self.hear(text)
        return self.said(self.llm(self.build_prompt(mid)), mid)

    # ---------------------------------------------------------------- sleep: work out two-step facts ahead of time
    def sleep(self):
        """Rebuild the derived pages from the current facts: if A's neighbour is B and B has a goldfish, write
        'A | neighbour -> has pet | goldfish'. A two-step question then needs one lookup instead of joining two notes,
        the step models do badly. Each page carries the receipts of both facts, so deleting either message withdraws
        it. Derived pages are never saved, only rebuilt (rule 1). This helps questions that follow the chain forwards
        ('what pet does Ana's neighbour have?'); backwards questions ('whose neighbour has a goldfish?') still rely on
        search. Returns the number of pages."""
        self.notebook.facts = [f for f in self.notebook.facts if not f.derived]
        cur = [f for f in self.notebook.current() if f.value is not None]
        by_subject = defaultdict(list)
        for f in cur:
            by_subject[f.subject].append(f)
        made = 0
        for f in cur:
            if f.value in by_subject and f.value != f.subject:
                for g in by_subject[f.value]:
                    if g.value == f.subject:
                        continue
                    self.notebook.add(Fact(f.subject, f"{f.relation} -> {g.relation}", g.value, max(f.time, g.time),
                                           sorted(set(f.sources) | set(g.sources)), derived=True, via=f.value))
                    made += 1
        return made

    # ---------------------------------------------------------------- forgetting: exact, with everything derived
    def forget_message(self, mid):
        """Delete a message and everything that came from it: its facts (unless another message also states them),
        any private note made from it, and every derived page built on them."""
        if not self.journal.delete(mid):
            return {"deleted": False}
        removed = self.notebook.drop_sources({mid})
        before = len(self.private)
        self.private = [p for p in self.private if mid not in p["sources"]]
        return {"deleted": True, "facts_removed": removed, "private_notes_removed": before - len(self.private),
                "derived_pages_now": self.sleep()}

    def forget_about(self, name):
        """Forget a person completely. This deletes every message that mentions them, or that a fact about them came
        from, including anything else said in those messages. A message left in the journal could be read again,
        so a person can't be removed without removing the messages."""
        name = self._canon(name)
        mids = {m for f in self.notebook.facts if not f.derived and name in (f.subject, f.value) for m in f.sources}
        full = " ".join(re.findall(r"[a-z0-9]+", name.lower()))
        mids |= {m.id for m in self.journal.messages.values()
                 if f" {full} " in f" {' '.join(re.findall(r'[a-z0-9]+', m.text.lower()))} "}
        for m in sorted(mids):
            self.forget_message(m)
        return {"messages_deleted": len(mids)}

    # ---------------------------------------------------------------- transparency
    def what_i_know(self):
        """Everything the brain holds, readable, with receipts: show this to the user."""
        L = []
        for f in sorted(self.notebook.current(), key=lambda f: (f.subject, f.derived, f.relation)):
            L.append(f"{self.notebook.line(f)}   (from messages {f.sources})")
        L += [f"private note ({day(p['time'])}): {p['text']}   (from messages {p['sources']})" for p in self.private]
        return "\n".join(L) or "(nothing yet)"

    # ---------------------------------------------------------------- state across restarts
    def save(self, path):
        data = {"format": 1, "next_id": self.journal.next_id,
                "journal": [asdict(m) for m in self.journal.messages.values()],
                "facts": [asdict(f) for f in self.notebook.facts if not f.derived],
                "private": self.private,
                "ratings": dict(self.taste.ratings) if self.taste else {}}
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)                              # never leaves a half-written file behind

    @classmethod
    def load(cls, path, llm=None, taste_map=None, **kw):
        data = json.load(open(path, encoding="utf-8"))
        b = cls(llm=llm, taste_map=taste_map, **kw)
        b.journal.next_id = data["next_id"]
        b.journal.messages = {m["id"]: Message(**m) for m in data["journal"]}
        for f in sorted(data["facts"], key=lambda f: f["seq"]):
            b.notebook.add(Fact(**{k: v for k, v in f.items() if k != "seq"}))
        b.private = data["private"]
        if b.taste is not None:
            b.taste.ratings = {k: tuple(v) for k, v in data["ratings"].items()}
        b.sleep()                                          # derived pages are rebuilt, never stored
        return b


# =====================================================================================================================
# 7. DEMO: checks each promise with numbers. Runs offline with a tiny stand-in for a real model.
# =====================================================================================================================
class DemoLLM:
    """Stands in for your model so the demo runs offline. It only understands a few sentence patterns with full
    names. A real model must also handle "she", "my sister" and nicknames, and that is the step to test first."""
    PATTERNS = [
        (r"^(?P<s>.+?)'s (?P<r>neighbour|sister|brother|friend|boss|partner) is (?P<v>.+?)\.?$", None),
        (r"^(?P<s>.+?) has a pet (?P<v>.+?)\.?$", "has pet"),
        (r"^(?P<s>.+?) (?:lives in|has moved to|moved to) (?P<v>.+?)\.?$", "lives in"),
        (r"^(?P<s>.+?) works as an? (?P<v>.+?)\.?$", "works as"),
        (r"^(?P<s>.+?) no longer has a pet\.?$", "has pet!"),
    ]

    def __call__(self, prompt):
        if prompt.startswith(EXTRACT_HEADER):
            msg = prompt.split("New message from the user:\n", 1)[1].split("\n", 1)[0].strip()
            for pat, rel in self.PATTERNS:
                m = re.match(pat, msg)
                if m:
                    r = (rel or m.group("r")).rstrip("!")
                    retract = bool(rel and rel.endswith("!"))
                    return json.dumps({"subject": m.group("s"), "relation": r,
                                       "value": None if retract else m.group("v"),
                                       "kind": "retraction" if retract else "fact"})
            return "NONE"
        if prompt.startswith(FOLLOWUP_HEADER):
            return "DONE"
        return "That's lovely news! <private>Ask how Ana is settling into Lisbon next week.</private>"


def demo():
    import random
    import tempfile
    print("1. Learning from every message, keeping every version")
    b = Brain(llm=DemoLLM())
    t0 = time.time() - 86400 * 400
    msgs = {}
    for i, text in enumerate(["Ana Costa lives in Porto.", "Ana Costa's neighbour is Bilal Barbosa.",
                              "Bilal Barbosa has a pet goldfish.", "Ana Costa works as a nurse.",
                              "Ana Costa moved to Lisbon."]):
        msgs[text] = b.hear(text, when=t0 + 86400 * 90 * i)
    print("   pages worked out during sleep:", b.sleep())
    print("  ", "\n   ".join(b.what_i_know().splitlines()))

    print("\n2. Recall for a two-step question (select first, then read)")
    for f in b.recall("What pet does Ana Costa's neighbour have?"):
        print("   -", b.notebook.line(f))

    print("\n3. Exact forgetting: delete 'Bilal Barbosa has a pet goldfish.'")
    print("  ", b.forget_message(msgs["Bilal Barbosa has a pet goldfish."]))
    left = b.what_i_know()
    print("   'goldfish' anywhere in the brain afterwards:", "goldfish" in left.lower())
    print("   forget everything about Bilal Barbosa:", b.forget_about("Bilal Barbosa"))
    print("   'Bilal' anywhere in the brain afterwards:", "bilal" in b.what_i_know().lower())

    print("\n4. Private state across conversations (the AI's own notes, hidden from the user)")
    print("   user sees:", b.respond("Ana Costa moved to Lisbon."))
    path = os.path.join(tempfile.mkdtemp(), "user.json")
    b.save(path)
    b2 = Brain.load(path, llm=DemoLLM())
    print("   after save + load, private notes:", [p["text"] for p in b2.private])
    print("   same knowledge after reload:", b2.what_i_know() == b.what_i_know())
    saved = open(path, encoding="utf-8").read().lower()
    print("   deleted words in the saved file ('goldfish', 'bilal'):", "goldfish" in saved, "bilal" in saved)

    print("\n5. ExactMemory: order never matters, deletion is exact, certificate")
    rng = np.random.default_rng(0)
    keys, vals = rng.normal(size=(200, 256)), rng.normal(size=(200, 64))
    m1, m2 = ExactMemory(256, 64), ExactMemory(256, 64)
    order = list(range(200))
    random.Random(1).shuffle(order)
    for i in range(200):
        m1.add(i, keys[i], vals[i])
    for i in order:
        m2.add(i, keys[i], vals[i])
    print("   largest difference, same pairs in two orders:", float(np.abs(m1.solve() - m2.solve()).max()))
    for i in range(50):
        m1.delete(i)
    m3 = ExactMemory(256, 64)
    for i in range(50, 200):
        m3.add(i, keys[i], vals[i])
    print("   largest difference, 50 deleted vs never stored:", float(np.abs(m1.solve() - m3.solve()).max()))
    print("   certificate:", m1.certificate())
    print("   reads back the right pair for 150 stored keys:",
          sum(m1.nearest(keys[i]) == str(i) for i in range(50, 200)), "/ 150")

    print("\n6. Taste: a map from many people, one person's position from their own ratings")
    n_users, n_items, kinds = 400, 120, 4
    likes = rng.normal(size=(kinds, n_items))
    crowd = []
    for u in range(n_users):
        k = u % kinds
        for i in rng.choice(n_items, 40, replace=False):
            crowd.append((f"u{u}", f"item{i}", float(likes[k, i] + rng.normal(0, 0.5) > 0)))
    tmap = TasteMap(dim=8).fit(crowd)
    me = PersonalTaste(tmap)
    mine = [(f"r{i}", f"item{i}", float(likes[2, i] > 0)) for i in range(0, 60, 3)]
    for rid, item, y in mine:
        me.rate(rid, item, y)
    you = PersonalTaste(tmap)
    for rid, item, y in reversed(mine):
        you.rate(rid, item, y)
    print("   same taste whatever the order:", float(np.abs(me.vector() - you.vector()).max()))
    me.forget("r0")
    fresh = PersonalTaste(tmap)
    for rid, item, y in mine[1:]:
        fresh.rate(rid, item, y)
    print("   forgetting one rating = never having it:", float(np.abs(me.vector() - fresh.vector()).max()))
    test = [f"item{i}" for i in range(1, 120, 3)]
    pos = [i for i in test if likes[2, int(i[4:])] > 0]
    neg = [i for i in test if likes[2, int(i[4:])] <= 0]
    auc = np.mean([me.score(p) > me.score(n) for p in pos for n in neg])
    print(f"   toy check, unseen items ranked by this person's taste: AUC {auc:.2f} (0.5 = guessing; toy data, "
          "not evidence; the real-data number is 0.740)")


if __name__ == "__main__":
    demo()
