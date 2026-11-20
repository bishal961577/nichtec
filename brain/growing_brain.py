"""Growing brain test, pre-registered in reports/Growing brain 2026.md.

One line (the anti-drift rule): does knowledge from the agent's own experience, kept in its weights (a growing brain
on a frozen head start), make later learning faster, last, and carry over better than the same knowledge kept as
notes that the same frozen model reads?

The world: invented creatures, each with a written description of 14 independent features. Each day a new event
(the gate opens for it, the bell rings for it, ...) happens or not by a fixed hidden law. Odd days bring a base law
over two features; even days a composed law built from an earlier event and a feature or another earlier event, so it
can only be learned quickly by an agent that kept what it learned. Each day the agent meets 99 creatures: it tests 24
of them in 6 rounds of 4, choosing which (acting), and after each round its best explanation is scored on 35 probe
creatures it never tests that day. 100 further creatures are never met; they test what was kept.

One agent design, three memories (the only difference between arms):
  none   the frozen Qwen3.5-4B, nothing kept between days;
  notes  the frozen model plus written notes: each past event's law, shown in the prompt and worked out by code;
  brain  the model plus a LoRA trained each night on the day's lessons (its own labels, the law in words, replay of
         earlier nights, a little of its own base answers); no notes in the prompt, only the events' names.
Inside a day all arms are the same machine: the model names yes/no questions about creatures (thinking off), reads
them for every creature (yes/no from its log-probabilities), and code combines one or two questions with and/or by
description length and picks the next creatures where its best explanations disagree.

  python brain/growing_brain.py --stub --stage dev        # bookkeeping with a stand-in model, no GPU
  python brain/growing_brain.py --world-stats             # the generator on 200 seeds no stage uses
  python brain/growing_brain.py --cpu-smoke --model Qwen/Qwen3.5-0.8B   # the real model code, tiny, on a CPU
  python brain/growing_brain.py --summarise results/growing_brain/exam
The real runs go through brain/modal_brain4b.py (one GPU per world).
"""
import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import time

STAGES = {"dev": [1, 2], "exam": list(range(101, 109)), "check": [1]}
CFG = dict(days=12, rounds=6, per_round=4, daily=99, probes=35, kept=100, pool=200, samples=4, max_q=24, top_k=8,
           max_new=160)
NIGHT = dict(lr=1e-4, epochs=2, protect=28, batch_tokens=8192)
GEN = dict(temperature=0.8, top_p=0.95, top_k=50)
ERR = math.log(19)                      # nats per error, as in the ingredients test (a law with 5% noise)
ARMS = ["none", "notes", "brain"]
REFS = ("oracle", "notes+all", "brain+all")
FEATURES = [  # (key, when true, when false, the question)
    ("scales", "It has scales.", "It has no scales.", "Does it have scales?"),
    ("fur", "It is covered in fur.", "It has no fur.", "Does it have fur?"),
    ("horns", "It has horns.", "It has no horns.", "Does it have horns?"),
    ("wings", "It has wings.", "It has no wings.", "Does it have wings?"),
    ("tail", "It has a long tail.", "It has no long tail.", "Does it have a long tail?"),
    ("glow", "It glows faintly in the dark.", "It does not glow in the dark.", "Does it glow in the dark?"),
    ("water", "It lives near water.", "It lives far from water.", "Does it live near water?"),
    ("night", "It is active at night.", "It is active during the day.", "Is it active at night?"),
    ("plants", "It eats only plants.", "It eats meat.", "Does it eat only plants?"),
    ("eggs", "It lays eggs.", "It gives birth to live young.", "Does it lay eggs?"),
    ("song", "It sings.", "It makes no sound.", "Does it sing?"),
    ("large", "It is larger than a horse.", "It is smaller than a horse.", "Is it larger than a horse?"),
    ("stripes", "It has stripes.", "It has no stripes.", "Does it have stripes?"),
    ("legs", "It has more than four legs.", "It has four legs or fewer.", "Does it have more than four legs?"),
]
FKEYS = [f[0] for f in FEATURES]
QUESTION = {f[0]: f[3] for f in FEATURES}
EVENTS = [  # (what happens, the question, its one distinctive word); no word overlaps a feature
    ("the gate opens for it", "Does the gate open for it?", "gate"),
    ("the bell rings for it", "Does the bell ring for it?", "bell"),
    ("the lantern brightens near it", "Does the lantern brighten near it?", "lantern"),
    ("the drum sounds for it", "Does the drum sound for it?", "drum"),
    ("the mirror fogs near it", "Does the mirror fog near it?", "mirror"),
    ("the clock stops near it", "Does the clock stop near it?", "clock"),
    ("the harp hums for it", "Does the harp hum for it?", "harp"),
    ("the compass spins near it", "Does the compass spin near it?", "compass"),
    ("the kettle whistles near it", "Does the kettle whistle near it?", "kettle"),
    ("the banner rises for it", "Does the banner rise for it?", "banner"),
    ("the candle flares near it", "Does the candle flare near it?", "candle"),
    ("the anvil rings for it", "Does the anvil ring for it?", "anvil"),
    ("the loom weaves for it", "Does the loom weave for it?", "loom"),
    ("the urn warms near it", "Does the urn warm near it?", "urn"),
    ("the flute trills for it", "Does the flute trill for it?", "flute"),
    ("the gong booms for it", "Does the gong boom for it?", "gong"),
]
INTRO = ("In this world, creatures are brought to strange objects, and things happen or not by fixed hidden laws "
         "about the creatures.\n\n")


# ---------------------------------------------------------------- the world

def creature_names(rng, n):
    on, vo, co = "bdfgklmnprstvz", "aeiou", "lnrs"
    out = set()
    while len(out) < n:
        w = rng.choice(on) + rng.choice(vo) + rng.choice(on) + rng.choice(vo) + rng.choice(["", rng.choice(co)])
        out.add(w)
    return sorted(out)


def make_creatures(seed, n):
    """Each feature is true for 70% or 30% of creatures (seven of each, by the seed), independently, so laws of two
    parts can be balanced: two 70% parts joined by 'and', or two 30% parts by 'or', are near half."""
    rng = random.Random(f"creatures-{seed}")
    names = creature_names(rng, n)
    rng.shuffle(names)
    keys = list(FKEYS)
    rng.shuffle(keys)
    p = {k: 0.7 if i < len(keys) // 2 else 0.3 for i, k in enumerate(keys)}
    C = {}
    for name in names:
        f = {k: rng.random() < p[k] for k in FKEYS}
        lines = [t if f[k] else nt for k, t, nt, _ in FEATURES]
        rng.shuffle(lines)
        C[name] = {"f": f, "desc": " ".join(lines)}
    return C


def holds(node, f, ev):
    """A law is a tree: ("t", feature, want), ("e", event index, want), or ("and"/"or", left, right)."""
    if node[0] == "t":
        return f[node[1]] == node[2]
    if node[0] == "e":
        return ev[node[1]] == node[2]
    a, b = holds(node[1], f, ev), holds(node[2], f, ev)
    return (a and b) if node[0] == "and" else (a or b)


def law_text(node, events):
    if node[0] == "t":
        return f"{'' if node[2] else 'not '}{node[1]}"
    if node[0] == "e":
        return f"{'' if node[2] else 'not '}<{events[node[1]][0]}>"
    return f"({law_text(node[1], events)} {node[0]} {law_text(node[2], events)})"


def agree(a, b, names):
    return sum(a[n] == b[n] for n in names) / len(names)


def two_feature_ceiling(lab, C, names):
    """Best agreement any rule of at most two raw features reaches: the most an agent with no memory can express."""
    lits = [{n: C[n]["f"][k] == v for n in names} for k in FKEYS for v in (True, False)]
    best = max(agree(l, lab, names) for l in lits)
    for a, b in itertools.combinations(lits, 2):
        best = max(best, agree({n: a[n] and b[n] for n in names}, lab, names),
                   agree({n: a[n] or b[n] for n in names}, lab, names))
    return best


def make_world(seed, days=None):
    """Odd days: a base law over two of the world's six features. Even days: an earlier event (either sign) joined
    with a feature or another earlier event. Every law is balanced (30-70% yes; composed 25-75%), needs both parts
    (no single feature or earlier event agrees on more than 85%/88%), differs from every earlier law (at most 90%
    agreement either way) and, if composed, is out of reach of any rule of two raw features (at most 90%)."""
    days = days or CFG["days"]
    C = make_creatures(seed, CFG["pool"] + CFG["kept"])
    allnames = list(C)
    kept, pool = sorted(allnames[:CFG["kept"]]), sorted(allnames[CFG["kept"]:])
    names = sorted(allnames)
    rng = random.Random(f"world-{seed}")
    care = rng.sample(FKEYS, 6)
    events = rng.sample(EVENTS, days)
    laws, labs, ceil = [], [], []
    for d in range(days):
        for _ in range(20000):
            if d % 2 == 0:
                a, b = rng.sample(care, 2)
                node = (rng.choice(["and", "or"]), ("t", a, rng.random() < 0.5), ("t", b, rng.random() < 0.5))
                lo, hi, single = 0.30, 0.70, 0.85
            else:
                left = ("e", rng.randrange(d), rng.random() < 0.5)
                if d < 2 or rng.random() < 0.5:
                    right = ("t", rng.choice(care), rng.random() < 0.5)
                else:
                    right = ("e", rng.choice([k for k in range(d) if k != left[1]]), rng.random() < 0.5)
                node = (rng.choice(["and", "or"]), left, right)
                lo, hi, single = 0.25, 0.75, 0.88
            lab = {n: holds(node, C[n]["f"], {k: labs[k][n] for k in range(d)}) for n in names}
            if not lo <= sum(lab.values()) / len(names) <= hi:
                continue
            singles = [{n: C[n]["f"][k] == v for n in names} for k in FKEYS for v in (True, False)]
            singles += [{n: labs[k][n] == v for n in names} for k in range(d) for v in (True, False)]
            if max(agree(s, lab, names) for s in singles) > single:
                continue
            if any(max(agree(lab, l, names), 1 - agree(lab, l, names)) > 0.90 for l in labs):
                continue
            c = two_feature_ceiling(lab, C, names) if d % 2 else None
            if c is not None and c > 0.90:
                continue
            break
        else:
            raise RuntimeError(f"world {seed}: no law found for day {d + 1}")
        laws.append(node)
        labs.append(lab)
        ceil.append(None if c is None else round(c, 3))
    daily, probes = [], []
    for d in range(days):
        r = random.Random(f"day-{seed}-{d}")
        today = sorted(r.sample(pool, CFG["daily"]))
        yes = [n for n in today if labs[d][n]]
        no = [n for n in today if not labs[d][n]]
        k = min(len(yes), max(1, round(CFG["probes"] * len(yes) / len(today))))
        daily.append(today)
        probes.append(sorted(r.sample(yes, k) + r.sample(no, CFG["probes"] - k)))
    return {"seed": seed, "C": C, "care": care, "events": events, "laws": laws, "labels": labs, "kept": kept,
            "daily": daily, "probes": probes, "ceiling": ceil, "text": [law_text(l, events) for l in laws]}


# ---------------------------------------------------------------- prompts and parsing

def yn(b):
    return "yes" if b else "no"


def read_prompt(name, desc, q):
    return f"Creature: {name}\nDescription: {desc}\nQuestion: {q}\nAnswer yes or no."


def formula_text(f):
    lits, op = f
    parts = [f"{'' if s else 'not '}[{q}]" for q, s in lits]
    return parts[0] if op == "single" else f" {op} ".join(parts)


def qtext(q, world):
    return world["events"][q[1]][1] if isinstance(q, tuple) else q


def ftext(f, world):
    lits, op = f
    return formula_text((tuple((qtext(q, world), s) for q, s in lits), op))


def naming_prompt(arm, day, world, tested, notes):
    ev = world["events"]
    known = ""
    if arm != "none" and day > 0:
        if arm == "notes":
            head = "Events you learned about before, with the law you found for each:\n"
            lines = [f"- {ev[k][0]}: {ftext(notes[k], world) if notes.get(k) else '(no law found)'}"
                     for k in range(day)]
        else:
            head = "Events you learned about before:\n"
            lines = [f"- {ev[k][0]}" for k in range(day)]
        known = (head + "\n".join(lines) + "\nYou may use a known event as a question, written exactly like: "
                 f"{ev[0][1]}\n\n")
    C = world["C"]
    res = "".join(f"- {n}: {yn(y)}. {C[n]['desc']}\n" for n, y in sorted(tested))
    return (INTRO + known + f"Today's event: {ev[day][0]}.\nCreatures tested so far today (yes means it "
            f"happened), each with its description:\n" + res +
            "\nList up to 8 short yes-or-no questions about a creature that could help explain these results. "
            "Write one question per line, nothing else.")


def parse_questions(replies):
    out, seen = [], set()
    for rep in replies:
        for line in rep.splitlines():
            q = re.sub(r"^[\s\-\*•\d\.\)\(]+", "", line).strip().strip("*").strip()
            key = re.sub(r"[^a-z0-9]", "", q.lower())
            if q.endswith("?") and 5 < len(q) < 150 and key not in seen:
                seen.add(key)
                out.append(q)
    return out


def canonical(q, arm, day, world):
    """A question naming an event's object becomes that event's own question, as ("event", index). Dropped (returns
    None): a question about today's event or an event the arm has not met (circular or meaningless), and one that is
    not a yes/no about each creature (it names a creature, or starts like "Which ..." or "Can you ...")."""
    low = q.lower()
    words = set(re.findall(r"[a-z]+", low))
    if re.match(r"(which|what|how|why|who|where|when|can you|could you|is there|are there)\b", low) or \
            words & world["C"].keys():                     # not a yes/no about each creature (CPU check, section 10)
        return None
    for k, (_, _, word) in enumerate(world["events"]):
        if re.search(rf"\b{word}\b", low):
            return None if (arm == "none" or k >= day) else ("event", k)
    if words & {e[2] for e in EVENTS}:                      # an object that is not in this world
        return None
    return q


# ---------------------------------------------------------------- code: combining, choosing, scoring

def combine(feats, tested):
    """Every rule of at most two questions (each possibly negated, two joined by and/or), plus 'always yes/no'.
    Cost in nats: errors on the tested creatures x ln 19 + questions x ln(2K) + ln 2 for a joining word. Returns
    candidates cheapest first (ties: fewer questions, then text)."""
    qs = sorted(feats, key=str)
    K = max(1, len(qs))
    names = [n for n, _ in tested]
    y = [v for _, v in tested]
    out = [((ERR * sum(v != c for v in y), 0, f"always {yn(c)}"), ((), "const", c)) for c in (True, False)]
    lits = [((q, s), [feats[q][n] == s for n in names]) for q in qs for s in (True, False)]
    for l, v in lits:
        out.append(((ERR * sum(a != b for a, b in zip(v, y)) + math.log(2 * K), 1, str(l)), ((l,), "single", None)))
    for (l1, v1), (l2, v2) in itertools.combinations(lits, 2):
        if l1[0] == l2[0]:
            continue
        for op in ("and", "or"):
            v = [(a and b) if op == "and" else (a or b) for a, b in zip(v1, v2)]
            out.append(((ERR * sum(a != b for a, b in zip(v, y)) + 2 * math.log(2 * K) + math.log(2), 2,
                         f"{l1}{op}{l2}"), ((l1, l2), op, None)))
    out.sort(key=lambda x: x[0])
    return out


def apply(cand, feats, n):
    lits, op, const = cand[1]
    if op == "const":
        return const
    vals = [feats[q][n] == s for q, s in lits]
    return vals[0] if op == "single" else (all(vals) if op == "and" else any(vals))


def choose(cands, feats, pool, k, rng):
    """The next creatures to test: where the cheapest explanations disagree most (weights exp(-extra cost))."""
    top = cands[:CFG["top_k"]]
    c0 = top[0][0][0]
    w = [math.exp(-(c[0][0] - c0)) for c in top]

    def unsure(n):
        p = sum(wi for wi, c in zip(w, top) if apply(c, feats, n)) / sum(w)
        return min(p, 1 - p)
    return sorted(pool, key=lambda n: (-unsure(n), rng.random()))[:k]


def accuracy(cand, feats, names, lab):
    return 100 * sum(apply(cand, feats, n) == lab[n] for n in names) / len(names)


def shown_rule(cand, world):
    lits, op, const = cand[1]
    return f"always {yn(const)}" if op == "const" else ftext((lits, op), world)


# ---------------------------------------------------------------- the heads: a stand-in, and the real model

PROMPT_RE = re.compile(r"Creature: (\w+)\nDescription: .*\nQuestion: (.+)\nAnswer yes or no\.$", re.S)


class Stub:
    """No model. Reads a feature question from the records with 3% of answers flipped; names four random feature
    questions per sample and, when shown known events, one or two of those; the 'brain' memorises the yes/no examples
    it is trained on with 8% noise and 3% flipping each night, and answers creatures it never saw at random. Unknown
    questions are answered at random. Only for checking the bookkeeping."""

    def __init__(self, world_C):
        self.C, self.mem, self.calls = world_C, {}, 0
        self.key_of = {f[3]: f[0] for f in FEATURES}

    def margins(self, prompts, adapter):
        out = []
        for p in prompts:
            self.calls += 1
            r = random.Random(p + str(adapter))
            m = PROMPT_RE.match(p)
            if not m:
                out.append(r.gauss(0, 1))
                continue
            name, q = m.group(1), m.group(2)
            if q in self.key_of:
                v = self.C[name]["f"][self.key_of[q]]
                out.append(2.0 if v != (r.random() < 0.03) else -2.0)
            elif adapter and (q, name) in self.mem:
                out.append(2.0 if self.mem[(q, name)] != (r.random() < 0.08) else -2.0)
            else:
                out.append(r.gauss(0, 1))
        return out

    def name(self, prompts, n, seed, adapters):
        outs = []
        for p in prompts:
            r = random.Random(f"{seed}-{p}")
            known = re.findall(r"^- the (\w+) ", p, flags=re.M)
            reps = []
            for _ in range(n):
                qs = r.sample([f[3] for f in FEATURES], 4)
                for word in r.sample(known, min(2, len(known))):
                    qs.append(next(e[1] for e in EVENTS if e[2] == word))
                reps.append("\n".join(qs))
            outs.append(reps)
        return outs

    def sleep(self, examples, log):
        for k in sorted(self.mem):
            if random.Random(f"forget-{k}-{len(examples)}").random() < 0.03:
                self.mem[k] = not self.mem[k]
        for p, ans in examples:
            m = PROMPT_RE.match(p)
            if m and m.group(2) not in self.key_of:
                self.mem[(m.group(2), m.group(1))] = ans == "yes"


class HFHead:
    """Qwen3.5 with one LoRA (zero at the start, so it equals the base model until the first night). Thinking off.
    Margins are log P(yes) - log P(no) for the first answer token after the chat template."""

    def __init__(self, model_name, device="cuda", dtype="bfloat16", log=print, batch=64):
        import torch
        from peft import LoraConfig, get_peft_model
        import who_train as T
        self.torch, self.T, self.device, self.log, self.batch = torch, T, device, log, batch
        self.tok, base = T.load(model_name, getattr(torch, dtype), device)
        self.model = get_peft_model(base, LoraConfig(target_modules=T.TARGETS, task_type="CAUSAL_LM", **T.LORA))
        self.model.eval()
        first = lambda w: self.tok(w, add_special_tokens=False)["input_ids"][0]
        self.yes, self.no = sorted({first("yes"), first("Yes")}), sorted({first("no"), first("No")})
        self.calls = 0

    def _ctx(self, adapter):
        return contextlib.nullcontext() if adapter else self.model.disable_adapter()

    def margins(self, prompts, adapter):
        """Right padding, read at each prompt's own last token: in a model that only looks back, padding placed
        after the text cannot change it (left padding did on the H100's kernels; section 10). Only the logits at
        the batch's distinct last positions are computed."""
        torch = self.torch
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        res = [0.0] * len(prompts)
        with torch.no_grad(), self._ctx(adapter):
            for k in range(0, len(order), self.batch):
                idx = order[k:k + self.batch]
                enc = self.tok([self.T.chat_prefix(self.tok, prompts[i]) for i in idx], return_tensors="pt",
                               padding=True, add_special_tokens=False, padding_side="right").to(self.device)
                last = enc["attention_mask"].sum(1) - 1
                keep = torch.unique(last)
                lg = self.model(**enc, logits_to_keep=keep).logits
                pos = torch.searchsorted(keep, last)
                lp = torch.log_softmax(lg[torch.arange(len(idx), device=lg.device), pos].float(), dim=-1)
                m = torch.logsumexp(lp[:, self.yes], dim=-1) - torch.logsumexp(lp[:, self.no], dim=-1)
                for i, v in zip(idx, m.tolist()):
                    res[i] = v
        self.calls += len(prompts)
        return res

    def name(self, prompts, n, seed, adapters):
        """One prompt at a time (its n samples together), so there is no padding at all."""
        torch = self.torch
        outs = []
        for i, (p, flag) in enumerate(zip(prompts, adapters)):
            torch.manual_seed(seed + i)
            enc = self.tok([self.T.chat_prefix(self.tok, p)], return_tensors="pt",
                           add_special_tokens=False).to(self.device)
            with torch.no_grad(), self._ctx(flag):
                g = self.model.generate(**enc, do_sample=True, num_return_sequences=n, max_new_tokens=CFG["max_new"],
                                        pad_token_id=self.tok.pad_token_id, **GEN)
            g = g[:, enc["input_ids"].shape[1]:]
            outs.append([self.tok.decode(g[s], skip_special_tokens=True) for s in range(n)])
        return outs

    def _reference_delta_rule(self):
        """Use transformers' plain-PyTorch gated delta rule while training: flash-linear-attention refuses its
        backward kernel on H100s with this Triton (a known wrong-gradient bug, its issue #640). The name is looked
        up at every call, so swapping the module's global is enough; reading keeps the fast forward kernel."""
        import sys as _sys
        mods = {type(x).__module__ for x in self.model.modules() if type(x).__name__.endswith("GatedDeltaNet")}
        swaps = []
        for name in mods:
            mod = _sys.modules[name]
            f = getattr(mod, "torch_chunk_gated_delta_rule", None)
            if f is not None and hasattr(f, "__wrapped__"):
                swaps.append((mod, f))
        return swaps

    def sleep(self, examples, log):
        """Continue training the LoRA on (prompt, answer) pairs: loss on the answer tokens only, NIGHT settings."""
        swaps = self._reference_delta_rule()
        for mod, f in swaps:
            mod.torch_chunk_gated_delta_rule = f.__wrapped__
        try:
            self._train(examples, log, len(swaps))
        finally:
            for mod, f in swaps:
                mod.torch_chunk_gated_delta_rule = f

    def _train(self, examples, log, swapped):
        torch, T, m = self.torch, self.T, self.model
        m.train()
        m.gradient_checkpointing_enable()
        if not getattr(self, "_grads_ready", False):       # registers a hook: once, not every night
            m.enable_input_require_grads()
            self._grads_ready = True
        data = []
        for p, a in examples:
            pre = self.tok(T.chat_prefix(self.tok, p), add_special_tokens=False)["input_ids"]
            ans = self.tok(a + "<|im_end|>", add_special_tokens=False)["input_ids"]
            data.append((pre + ans, [-100] * len(pre) + ans))
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=NIGHT["lr"], weight_decay=0.0)
        rng = random.Random(len(examples))
        steps, t0, loss = 0, time.time(), None
        for _ in range(NIGHT["epochs"]):
            order = list(range(len(data)))
            rng.shuffle(order)
            order.sort(key=lambda i: len(data[i][0]) // 16)      # similar lengths together, shuffled within
            batches, cur, mx = [], [], 0
            for i in order:
                L = len(data[i][0])
                if cur and max(mx, L) * (len(cur) + 1) > NIGHT["batch_tokens"]:
                    batches.append(cur)
                    cur, mx = [], 0
                cur.append(i)
                mx = max(mx, L)
            if cur:
                batches.append(cur)
            rng.shuffle(batches)
            for b in batches:
                L = max(len(data[i][0]) for i in b)
                ids = torch.full((len(b), L), self.tok.pad_token_id, dtype=torch.long)
                lab = torch.full((len(b), L), -100, dtype=torch.long)
                att = torch.zeros((len(b), L), dtype=torch.long)
                for r, i in enumerate(b):                         # right padding for training
                    x, y = data[i]
                    ids[r, :len(x)] = torch.tensor(x)
                    lab[r, :len(y)] = torch.tensor(y)
                    att[r, :len(x)] = 1
                loss = m(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                         labels=lab.to(self.device)).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
        m.gradient_checkpointing_disable()
        m.eval()
        log(f"    sleep: {len(examples)} examples, {steps} steps, last loss {loss.item():.4f}, "
            f"{time.time() - t0:.0f}s (reference delta rule in {swapped} module(s))")

    def batching_check(self, prompts):
        """Batched margins against one prompt at a time: the largest gap, the mean gap, and how many yes/no answers
        flip among those the single run gives with a margin above 0.5."""
        one = [self.margins([p], False)[0] for p in prompts]
        many = self.margins(prompts, False)
        gaps = [abs(a - b) for a, b in zip(one, many)]
        flips = sum((a > 0) != (b > 0) for a, b in zip(one, many) if abs(a) > 0.5)
        return {"max_gap": round(max(gaps), 4), "mean_gap": round(sum(gaps) / len(gaps), 4), "confident_flips": flips,
                "n": len(prompts)}


# ---------------------------------------------------------------- reading, with a cache

class Reader:
    def __init__(self, head, C):
        self.head, self.C, self.cache = head, C, {}

    def read(self, q, names, tag):
        """Margins for question q on creatures `names`; tag 'base' or 'brain<night>' (the adapter as it is now)."""
        todo = [n for n in names if (tag, q, n) not in self.cache]
        if todo:
            got = self.head.margins([read_prompt(n, self.C[n]["desc"], q) for n in todo], tag != "base")
            for n, m in zip(todo, got):
                self.cache[(tag, q, n)] = m
        return {n: self.cache[(tag, q, n)] > 0 for n in names}


def expand(k, notes, base_feat, names, memo):
    """Work out a noted event for creatures `names` from its written law (code), reading plain questions with the
    frozen model."""
    if k in memo:
        return memo[k]
    f = notes.get(k)
    if f is None:
        memo[k] = {n: False for n in names}
        return memo[k]
    lits, op = f
    cols = []
    for q, s in lits:
        col = expand(q[1], notes, base_feat, names, memo) if isinstance(q, tuple) else base_feat(q, names)
        cols.append({n: col[n] == s for n in names})
    memo[k] = {n: (cols[0][n] if op == "single" else
                   (all(c[n] for c in cols) if op == "and" else any(c[n] for c in cols))) for n in names}
    return memo[k]


# ---------------------------------------------------------------- one world, all arms in step

def run_world(head, seed, log=print, days=None, cfg=None):
    """One world, all arms. A crash is returned, not raised: the result carries the traceback and everything done
    so far, so a failed GPU job still shows its log (summarise refuses any world with an error)."""
    import traceback
    R = {"seed": seed}
    try:
        return _run_world(head, seed, log, days, cfg, R)
    except Exception:
        R["error"] = traceback.format_exc()
        log(R["error"])
        return R


def _run_world(head, seed, log, days, cfg, R):
    CFG.update(cfg or {})
    world = make_world(seed, days)
    C, D = world["C"], len(world["laws"])
    reader = Reader(head, C)
    night = 0                                       # the brain's adapter has been trained after `night` days
    notes, brain_mem = {}, []
    R.update({"seed": seed, "laws": world["text"], "care": world["care"], "ceiling": world["ceiling"],
              "arms": {a: [] for a in ARMS}, "refs": {r: [] for r in REFS}, "keeps": {"notes": [], "brain": []},
              "cfg": dict(CFG), "night": dict(NIGHT)})
    log(f"world {seed}: {' | '.join(world['text'])}")

    def tag_of(arm):
        return f"brain{night}" if (arm == "brain" and night > 0) else "base"

    def base_feat(q, names):
        return reader.read(q, names, "base")

    def event_feat(arm, k, names, memo):
        if arm == "notes":
            return expand(k, notes, base_feat, names, memo)
        return reader.read(world["events"][k][1], names, tag_of(arm))

    met = set()                                     # creatures met on earlier days (for the never-met split)
    for d in range(D):
        lab, today, probes = world["labels"][d], world["daily"][d], world["probes"][d]
        unmet = [n for n in probes if n not in met]
        pool0 = [n for n in today if n not in set(probes)]
        first = random.Random(f"first-{seed}-{d}").sample(pool0, CFG["per_round"])
        st = {a: {"tested": [], "qs": [], "feats": {}, "curve": [], "qlog": [],
                  "rng": random.Random(f"choose-{seed}-{d}-{a}")} for a in ARMS}
        orng = random.Random(f"choose-{seed}-{d}-oracle")
        ofeats = {QUESTION[k]: {n: C[n]["f"][k] for n in today} for k in FKEYS}
        for k in range(d):
            ofeats[("event", k)] = {n: world["labels"][k][n] for n in today}
        otest, ocands, ocurve = [], None, []
        refs = {"notes+all": [], "brain+all": []}
        memo = {}
        for r in range(CFG["rounds"]):
            for a in ARMS:                                    # acting: which creatures to test next
                s = st[a]
                done = {x for x, _ in s["tested"]}
                pick = list(first) if r == 0 else choose(s["cands"], s["feats"], [n for n in pool0 if n not in done],
                                                         CFG["per_round"], s["rng"])
                s["tested"] += [(n, lab[n]) for n in pick]
            otest += list(first) if r == 0 else choose(ocands, ofeats, [n for n in pool0 if n not in set(otest)],
                                                       CFG["per_round"], orng)
            ocands = combine(ofeats, [(n, lab[n]) for n in otest])
            ocurve.append(accuracy(ocands[0], ofeats, probes, lab))
            prompts = [naming_prompt(a, d, world, st[a]["tested"], notes) for a in ARMS]
            replies = head.name(prompts, CFG["samples"], seed=hash_int(f"name-{seed}-{d}-{r}"),
                                adapters=[a == "brain" and night > 0 for a in ARMS])
            for a, rep in zip(ARMS, replies):                 # naming
                s = st[a]
                s["qlog"].append(rep)
                for q in parse_questions(rep):
                    key = canonical(q, a, d, world)
                    if key is not None and key not in s["qs"] and len(s["qs"]) < CFG["max_q"]:
                        s["qs"].append(key)
            for a in ARMS:                                    # reading, combining, scoring
                s = st[a]
                for q in s["qs"]:
                    if q not in s["feats"]:
                        s["feats"][q] = (event_feat(a, q[1], today, memo) if isinstance(q, tuple)
                                         else reader.read(q, today, tag_of(a)))
                s["cands"] = combine(s["feats"], s["tested"])
                s["curve"].append(accuracy(s["cands"][0], s["feats"], probes, lab))
            for a, ref in (("notes", "notes+all"), ("brain", "brain+all")):   # every known event added (reference)
                f2 = dict(st[a]["feats"])
                for k in range(d):
                    f2.setdefault(("event", k), event_feat(a, k, today, memo))
                refs[ref].append(accuracy(combine(f2, st[a]["tested"])[0], f2, probes, lab))
        kind = "base" if d % 2 == 0 else "composed"
        for a in ARMS:
            s, curve = st[a], st[a]["curve"]
            exp90 = next((CFG["per_round"] * (i + 1) for i, v in enumerate(curve) if v >= 90),
                         CFG["per_round"] * (CFG["rounds"] + 1))
            R["arms"][a].append({"day": d + 1, "type": kind, "curve": [round(v, 2) for v in curve],
                                 "auc": round(statistics.mean(curve), 2), "final": round(curve[-1], 2), "exp90": exp90,
                                 "final_unmet": round(accuracy(s["cands"][0], s["feats"], unmet, lab), 2) if unmet
                                 else None, "n_unmet": len(unmet),
                                 "rule": shown_rule(s["cands"][0], world), "questions": [qtext(q, world) for q in s["qs"]],
                                 "tested": [n for n, _ in s["tested"]], "replies": s["qlog"]})
        for ref in refs:
            R["refs"][ref].append({"day": d + 1, "type": kind, "curve": [round(v, 2) for v in refs[ref]],
                                   "auc": round(statistics.mean(refs[ref]), 2)})
        R["refs"]["oracle"].append({"day": d + 1, "type": kind, "curve": [round(v, 2) for v in ocurve],
                                    "auc": round(statistics.mean(ocurve), 2)})
        met |= set(today)
        # night: the notebook writes its law; the brain sleeps on its own lessons
        b = st["notes"]["cands"][0]
        if b[1][1] != "const":
            notes[d] = (b[1][0], b[1][1])
        sb = st["brain"]
        own = {n: apply(sb["cands"][0], sb["feats"], n) for n in today}
        own.update(dict(sb["tested"]))
        brain_mem += lessons(d, world, own, sb["cands"][0])
        today_ex = [(p, x) for p, x, k in brain_mem if k == d]
        past_ex = [(p, x) for p, x, k in brain_mem if k != d]
        head.sleep(today_ex + past_ex + protect(reader, world, d), log)
        night = d + 1
        # keeping: every event met so far, worked out by each memory for the 100 creatures never met
        kept, memo = world["kept"], {}
        kn = [round(100 * agree(expand(k, notes, base_feat, kept, memo), world["labels"][k], kept), 2)
              for k in range(d + 1)]
        kb = [round(100 * agree(reader.read(world["events"][k][1], kept, f"brain{night}"), world["labels"][k], kept), 2)
              for k in range(d + 1)]
        R["keeps"]["notes"].append(kn)
        R["keeps"]["brain"].append(kb)
        log(f"  day {d + 1:2d} {kind:8s} final: " + "  ".join(f"{a} {R['arms'][a][-1]['final']:5.1f}" for a in ARMS)
            + f"  oracle {ocurve[-1]:5.1f} | kept, never-met creatures: notes {statistics.mean(kn):5.1f} "
            f"brain {statistics.mean(kb):5.1f}")
    # did the nights damage the head start? the 14 feature questions on the never-met creatures, against the truth
    R["reading"] = {}
    for tag, key in (("base", "base"), (f"brain{night}", "brain_end")):
        right = [reader.read(QUESTION[k], world["kept"], tag)[n] == C[n]["f"][k] for k in FKEYS for n in world["kept"]]
        R["reading"][key] = round(100 * sum(right) / len(right), 2)
    R["calls"] = getattr(head, "calls", None)
    return R


def hash_int(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def lessons(d, world, own, cand):
    """What the brain is trained on after day d: its own yes/no for each of the day's 99 creatures (tested ones as
    observed) in two wordings; the law it found, in words, asked three ways; the list of events met so far."""
    stmt, ques, _ = world["events"][d]
    C, ex = world["C"], []
    for n in sorted(own):
        ex.append((read_prompt(n, C[n]["desc"], ques), yn(own[n]), d))
        ex.append((f"Here is a creature, {n}. {C[n]['desc']} {ques} Answer yes or no.", yn(own[n]), d))
    law = shown_rule(cand, world)
    for p in (f"What law decides whether {stmt}? Answer with the law only.",
              f"In this world, when {stmt}, what is true of the creature? Answer with the law only.",
              f"Describe the hidden law of this event: {stmt}. Answer with the law only."):
        ex.append((p, law, d))
    ex.append(("Which events have you learned about in this world? List them, one per line.",
               "\n".join(world["events"][k][0] for k in range(d + 1)), d))
    return ex


def protect(reader, world, d):
    """A little of the frozen model's own answers to the 14 feature questions (two creatures of the day), so the
    night's training does not pull the head start away. The frozen model labels these, not the truth."""
    rng = random.Random(f"protect-{world['seed']}-{d}")
    names = rng.sample(world["daily"][d], max(1, NIGHT["protect"] // len(FKEYS)))
    C, out = world["C"], []
    for k in FKEYS:
        m = reader.read(QUESTION[k], names, "base")
        out += [(read_prompt(n, C[n]["desc"], QUESTION[k]), yn(m[n])) for n in names]
    return out


# ---------------------------------------------------------------- summary and verdict

def summarise(results):
    bad = [R["seed"] for R in results if R.get("error")]
    assert not bad, f"worlds {bad} crashed; no summary from partial worlds"

    def mean_auc(days, kind, which=None):
        xs = [x["auc"] for x in days if x["type"] == kind and (which is None or x["day"] in which)]
        return statistics.mean(xs) if xs else float("nan")
    W = []
    for R in results:
        w = {"seed": R["seed"]}
        for a in ARMS:
            ad = R["arms"][a]
            w[a] = {"composed": mean_auc(ad, "composed"), "early": mean_auc(ad, "base", {1, 3, 5}),
                    "late": mean_auc(ad, "base", {7, 9, 11}),
                    "final_composed": statistics.mean(x["final"] for x in ad if x["type"] == "composed"),
                    "final_base": statistics.mean(x["final"] for x in ad if x["type"] == "base")}
            un = [(x["final_unmet"], x["n_unmet"]) for x in ad if x["type"] == "composed" and x["final_unmet"] is not None]
            w[a]["final_composed_unmet"] = (sum(v * k for v, k in un) / sum(k for _, k in un)) if un else float("nan")
            w[a]["faster"] = w[a]["late"] - w[a]["early"]
        for ref in REFS:
            rd = R["refs"][ref]
            w[ref] = {"composed": mean_auc(rd, "composed"), "base": mean_auc(rd, "base"),
                      "final_composed": statistics.mean(x["curve"][-1] for x in rd if x["type"] == "composed"),
                      "final_base": statistics.mean(x["curve"][-1] for x in rd if x["type"] == "base")}
        w["keeps"] = {k: statistics.mean(R["keeps"][k][-1]) for k in R["keeps"]}
        w["reading"] = R["reading"]
        W.append(w)

    def paired(f):
        xs = [f(w) for w in W]
        m = statistics.mean(xs)
        se = statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else float("nan")
        t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(len(xs) - 1, 2.0)
        ok = se == se
        return {"mean": round(m, 2), "se": round(se, 2) if ok else None, "ci_low": round(m - t * se, 2) if ok else None,
                "ci_high": round(m + t * se, 2) if ok else None, "up": sum(x > 0 for x in xs),
                "down": sum(x < 0 for x in xs), "n": len(xs)}
    avg = lambda f: round(statistics.mean(f(w) for w in W), 2)
    S = {"worlds": len(W),
         "arms": {a: {k: avg(lambda w, a=a, k=k: w[a][k]) for k in
                      ("composed", "early", "late", "faster", "final_composed", "final_base",
                       "final_composed_unmet")} for a in ARMS},
         "refs": {r: {k: avg(lambda w, r=r, k=k: w[r][k]) for k in ("composed", "base", "final_composed",
                                                                     "final_base")} for r in REFS},
         "keeps": {k: avg(lambda w, k=k: w["keeps"][k]) for k in ("notes", "brain")},
         "reading": {k: avg(lambda w, k=k: w["reading"][k]) for k in ("base", "brain_end")},
         "transfer_brain_minus_notes": paired(lambda w: w["brain"]["composed"] - w["notes"]["composed"]),
         "faster_brain_minus_notes": paired(lambda w: w["brain"]["faster"] - w["notes"]["faster"]),
         "transfer_brain_minus_none": paired(lambda w: w["brain"]["composed"] - w["none"]["composed"]),
         "transfer_notes_minus_none": paired(lambda w: w["notes"]["composed"] - w["none"]["composed"]),
         "per_world": W}
    S["verdict"] = verdict(S)
    return S


def verdict(S):
    """Lines fixed in reports/Growing brain 2026.md, section 5, before any model run."""
    o = S["refs"]["oracle"]
    checks = {
        "world learnable: oracle final >= 95 on both day types": o["final_base"] >= 95 and o["final_composed"] >= 95,
        "memory needed: no-memory final on composed days <= oracle - 10":
            S["arms"]["none"]["final_composed"] <= o["final_composed"] - 10,
        "head start reads descriptions: base feature reading >= 90": S["reading"]["base"] >= 90,
    }
    if not all(checks.values()):
        return {"call": "INCONCLUSIVE", "checks": checks}
    T, F = S["transfer_brain_minus_notes"], S["faster_brain_minus_notes"]
    damage = round(S["reading"]["base"] - S["reading"]["brain_end"], 2)
    wins = lambda x: x["mean"] >= 5 and x["ci_low"] is not None and x["ci_low"] > 0
    if (wins(T) or wins(F)) and S["keeps"]["brain"] >= 80 and damage <= 3:
        call = "BRAIN"
    elif T["mean"] <= 2 and F["mean"] <= 2:
        call = "NOTEBOOK ENOUGH"
    else:
        call = "BETWEEN"
    return {"call": call, "checks": checks, "reading_damage": damage,
            "brain_carried_anything": S["transfer_brain_minus_none"]["mean"] > 2}


def report(S, log=print):
    log(f"\n{S['worlds']} worlds; mean probe accuracy over the day's 6 rounds (auc) and after the last round (final)")
    log(f"{'':10s} {'composed':>9s} {'base 1-5':>9s} {'base 7-11':>10s} {'faster':>7s} {'final comp':>11s} "
        f"{'final base':>11s}")
    for a in ARMS:
        x = S["arms"][a]
        log(f"{a:10s} {x['composed']:9.1f} {x['early']:9.1f} {x['late']:10.1f} {x['faster']:7.1f} "
            f"{x['final_composed']:11.1f} {x['final_base']:11.1f}")
    for r in REFS:
        x = S["refs"][r]
        log(f"{r:10s} {x['composed']:9.1f} {'':9s} {'':10s} {'':7s} {x['final_composed']:11.1f} {x['final_base']:11.1f}")
    log("final on composed days, probe creatures never met before: " +
        "  ".join(f"{a} {S['arms'][a]['final_composed_unmet']:.1f}" for a in ARMS))
    log(f"kept at the end (never-met creatures, all 12 events): notes {S['keeps']['notes']:.1f}  "
        f"brain {S['keeps']['brain']:.1f}")
    log(f"feature reading: base {S['reading']['base']:.1f}  brain at the end {S['reading']['brain_end']:.1f}")
    for k in ("transfer_brain_minus_notes", "faster_brain_minus_notes", "transfer_brain_minus_none",
              "transfer_notes_minus_none"):
        x = S[k]
        log(f"{k}: {x['mean']:+.1f} (95% CI {x['ci_low']} to {x['ci_high']}; {x['up']} up, {x['down']} down of "
            f"{x['n']})")
    log(f"VERDICT: {S['verdict']['call']}  {json.dumps(S['verdict'])}")


# ---------------------------------------------------------------- command line

def world_stats(seeds=range(1000, 1200)):
    """The generator on seeds no stage uses: how far an agent with no memory can get on composed laws."""
    ceils, t0 = [], time.time()
    for s in seeds:
        ceils += [c for c in make_world(s)["ceiling"] if c is not None]
    q = sorted(ceils)
    print(f"{len(list(seeds))} worlds ({time.time() - t0:.0f}s): best two-feature agreement on composed laws: "
          f"median {statistics.median(q):.3f}, 10th pct {q[len(q) // 10]:.3f}, 90th pct {q[9 * len(q) // 10]:.3f}, "
          f"max {q[-1]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--stage", choices=sorted(STAGES), default="dev")
    ap.add_argument("--out")
    ap.add_argument("--world-stats", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--summarise")
    a = ap.parse_args()
    if a.world_stats:
        world_stats()
        return
    if a.summarise:
        res = [json.load(open(os.path.join(a.summarise, f))) for f in sorted(os.listdir(a.summarise))
               if f.startswith("world_")]
        S = summarise(res)
        report(S)
        json.dump({k: v for k, v in S.items() if k != "per_world"},
                  open(os.path.join(a.summarise, "summary.json"), "w"), indent=1)
        return
    if a.stub:
        assert a.stage != "exam", "the exam worlds are never run with the stand-in (their laws stay unseen)"
        res = []
        for seed in STAGES[a.stage]:
            R = run_world(Stub(make_world(seed)["C"]), seed)
            res.append(R)
            if a.out:
                os.makedirs(a.out, exist_ok=True)
                json.dump(R, open(os.path.join(a.out, f"world_{seed}.json"), "w"))
        report(summarise(res))
        return
    if a.cpu_smoke:
        import torch
        torch.set_num_threads(4)
        head = HFHead(a.model, device="cpu", dtype="float32", batch=8)
        w = make_world(1)
        n0, n1 = w["kept"][:2]
        gap = head.batching_check([read_prompt(n, w["C"][n]["desc"], q) for n in (n0, n1)
                                   for q in ("Does it have scales?", "Does the gate open for it?")])
        print(f"batched vs one at a time: {gap}")
        NIGHT.update(protect=14, batch_tokens=1024)          # a CPU's memory, not the H100's
        R = run_world(head, 1, days=2, cfg=dict(rounds=2, per_round=4, samples=2, max_q=3, max_new=60, daily=20,
                                                probes=8, kept=10, pool=40))
        print(json.dumps({k: R[k] for k in ("laws", "keeps", "reading")}, indent=1))
        print({a: [(x["day"], x["final"], x["rule"], x["questions"]) for x in R["arms"][a]] for a in ARMS})


if __name__ == "__main__":
    main()
