"""Order gap: does the order experience arrives in change what an AI agent learns on the job, and does an
order-free tally of evidence (a ledger) remove that and learn more? Pre-registered in reports/Order gap 2026.md.

The same 40 labelled cases reach three agents in five orders (three shuffles, a misleading block first, the same
block last). Agent A keeps the whole history in its prompt. Agents B and C share the model's rule proposals, written
as Python expressions from the last 10 cases every 5 cases: B keeps every proposal and answers with the one that gets
the most of all seen cases right (counts, the same in any order); C keeps one rule and swaps it only for a proposal
that fits the last 10 cases better. Every agent is scored on a separate balanced test set after the stream.

  python brain/order_gap.py              # laptop: Qwen2.5-1.5B-Instruct on the GPU
  python brain/order_gap.py --stub       # plumbing check, no model
  python brain/order_gap.py --base-url https://...modal.run/v1 --model Qwen/Qwen3.5-27B-FP8 --check
  python brain/order_gap.py --base-url https://...modal.run/v1 --model Qwen/Qwen3.5-27B-FP8
                                         # a model behind an OpenAI-compatible endpoint (vLLM or SGLang); --check
                                         # makes two calls and stops. Auth from ENDPOINT_TOKEN (Bearer) or
                                         # MODAL_KEY + MODAL_SECRET. If A is at the ceiling (random-order accuracy
                                         # >= 95), the harder rule set runs next on its own, as pre-registered.
"""
import argparse
import ast
import copy
import itertools
import json
import math
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ATTRS = {"color": ["red", "blue", "green", "yellow"], "shape": ["circle", "square", "triangle", "star"],
         "size": list(range(1, 10)), "material": ["wood", "metal", "glass", "plastic"]}
DOMAIN = [dict(zip(ATTRS, v)) for v in itertools.product(*ATTRS.values())]        # all 576 items
RULES = [  # (hidden rule, decoy: one condition on an attribute the hidden rule does not use)
    ("size >= 5 and color != 'red'", "shape in ('circle', 'square')"),
    ("shape == 'circle' or material == 'metal'", "color in ('blue', 'green')"),
    ("color in ('blue', 'green') and size <= 6", "material in ('wood', 'glass')"),
    ("material != 'wood' and shape != 'star'", "size >= 5"),
    ("size > 3 and shape in ('square', 'triangle')", "color in ('red', 'yellow')"),
    ("color == 'yellow' or size >= 7", "material in ('metal', 'plastic')"),
]
RULES_HARD = [  # three conditions; run only if A is at the ceiling on RULES (section 7 of the pre-registration)
    ("size >= 4 and color != 'red' and material != 'glass'", "shape in ('circle', 'square')"),
    ("(shape == 'circle' or material == 'metal') and size <= 7", "color in ('blue', 'green')"),
    ("(color in ('blue', 'green') and size % 2 == 1) or material == 'wood'", "shape in ('triangle', 'star')"),
    ("material != 'wood' and shape != 'star' and size > 2", "color in ('red', 'yellow')"),
    ("(size > 3 and shape in ('square', 'triangle')) or color == 'green'", "material in ('metal', 'plastic')"),
    ("color == 'yellow' or (size >= 6 and material in ('metal', 'glass'))", "shape in ('circle', 'triangle')"),
]
CEILING = 95                                  # A's random-order accuracy at or above this: too easy to show order
OFF_FORMAT = 0.10                             # more of A's answers than this outside yes/no: A is not measured
DRAWS = 2
ORDERS = ["random1", "random2", "random3", "decoy_first", "decoy_last"]
RANDOM = ORDERS[:3]
EVERY, WINDOW, PER_ROUND = 5, 10, 4
SAMPLING = {"temperature": 0.8, "top_p": 0.95, "top_k": 50, "max_new_tokens": 64}
GOOD = 0.95                                   # a proposal agreeing with the hidden rule on 95% of the domain is "good"
INTRO = ("An inspector accepts or rejects items by a fixed hidden rule. Items have four attributes: color (red, blue, "
         "green, yellow), shape (circle, square, triangle, star), size (1 to 9) and material (wood, metal, glass, "
         "plastic).\n\n")


# ---------------------------------------------------------------- rules as safe Python expressions

ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.Compare, ast.Eq,
           ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Name, ast.Load, ast.Constant,
           ast.Tuple, ast.List, ast.Set, ast.BinOp, ast.Add, ast.Sub, ast.Mod)
_compiled = {}


def compile_rule(text):
    """The rule's decision on every domain item as a tuple of bools, or None if it is not a whitelisted expression
    over the four attributes that returns a bool on every item. No calls, attributes, powers or products, so a
    model-written expression cannot do anything but compare values."""
    if text in _compiled:
        return _compiled[text]
    out = None
    try:
        tree = ast.parse(text, mode="eval") if len(text) <= 200 else None
    except (SyntaxError, ValueError):
        tree = None
    if tree is not None and all(isinstance(n, ALLOWED) for n in ast.walk(tree)) and \
            all(n.id in ATTRS for n in ast.walk(tree) if isinstance(n, ast.Name)) and \
            all(isinstance(n.value, (str, int, float)) and len(str(n.value)) <= 20
                for n in ast.walk(tree) if isinstance(n, ast.Constant)):
        code = compile(tree, "<rule>", "eval")
        try:
            vals = tuple(eval(code, {"__builtins__": {}}, dict(d)) for d in DOMAIN)
            out = vals if all(type(v) is bool for v in vals) else None
        except Exception:
            out = None
    _compiled[text] = out
    return out


def extract_rule(reply):
    """The first line of a model reply that is a valid rule, after stripping code fences, 'return', 'rule:' and
    'name =' prefixes and trailing punctuation."""
    for line in reply.replace("```python", "\n").replace("```", "\n").splitlines():
        s = line.strip().strip("`").strip()
        for p in ("return ", "rule:", "Rule:", "Expression:", "expression:", "Answer:", "answer:", "-", "*"):
            if s.startswith(p):
                s = s[len(p):].strip()
        head, eq, rest = s.partition("=")
        if eq and head.strip().isidentifier() and head.strip() not in ATTRS and not rest.startswith("="):
            s = rest.strip()
        tries = [s, s.rpartition(":")[2]] + s.split("`")[1::2]     # the line, after its last colon, inline code
        for x in tries:
            x = x.strip().rstrip(".;,").strip()
            if x and compile_rule(x) is not None:
                return x
    return None


def agreement(a, b):
    return sum(x == y for x, y in zip(a, b)) / len(a)


# ---------------------------------------------------------------- tasks and orders

def item_text(i):
    d = DOMAIN[i]
    return f"color={d['color']}, shape={d['shape']}, size={d['size']}, material={d['material']}"


def make_task(r, draw, rules=RULES, prefix="rule"):
    """One hidden rule with one draw of items. Stream: 10 'block' cases on which the decoy agrees with the rule
    (5 accepted, 5 rejected) and 30 others on which the decoy is right 14 times; 20 accepted in all.
    Test set: 40 other items, 20 accepted, the decoy right on exactly half. The five orders hold the same 40 cases."""
    truth, decoy = rules[r]
    T, D = compile_rule(truth), compile_rule(decoy)
    rng = random.Random(1000 * r + draw)
    cells = {(t, d): [i for i in range(len(DOMAIN)) if T[i] == t and D[i] == d] for t in (True, False)
             for d in (True, False)}
    for c in cells.values():
        rng.shuffle(c)

    def take(t, d, n):
        assert len(cells[(t, d)]) >= n, f"rule {r}: too few items in cell {(t, d)}"
        return [cells[(t, d)].pop() for _ in range(n)]

    block = take(True, True, 5) + take(False, False, 5)
    rest = take(True, True, 7) + take(True, False, 8) + take(False, True, 8) + take(False, False, 7)
    test = take(True, True, 10) + take(True, False, 10) + take(False, True, 10) + take(False, False, 10)
    rng.shuffle(block), rng.shuffle(rest), rng.shuffle(test)
    orders = {}
    for k in RANDOM:
        s = block + rest
        rng.shuffle(s)
        orders[k] = s
    orders["decoy_first"], orders["decoy_last"] = block + rest, rest + block
    return {"id": f"{prefix}{r + 1}_draw{draw + 1}", "rule": truth, "decoy": decoy, "T": T, "D": D, "block": block,
            "test": test, "orders": orders}


# ---------------------------------------------------------------- agent A: the whole history in the prompt

def case_line(i, T):
    return f"{item_text(i)}: {'yes' if T[i] else 'no'}\n"


def query(i):
    return f"New item: {item_text(i)}\nWill the inspector accept it? Answer yes or no."


class LLM:
    """A chat model. history() scores yes against no for every stream item before its label and for every test
    item after the whole stream, reusing the key-value cache of the growing history."""

    def __init__(self, name, device, model=None, tok=None):
        import torch
        self.torch, self.dev = torch, device
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        if tok is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(name)
            try:
                model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
        self.tok, self.model = tok, model.to(device=device, dtype=dtype).eval()
        full = tok.apply_chat_template([{"role": "user", "content": "\x00"}], tokenize=False,
                                       add_generation_prompt=True)
        self.head, self.tail = full.split("\x00")

        def first(words):
            return sorted({tok.encode(w, add_special_tokens=False)[0] for w in words})
        self.yes, self.no = first(["yes", "Yes"]), first(["no", "No"])

    def ids(self, text):
        return self.tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.dev)

    def extend(self, cache, text):
        with self.torch.no_grad():
            out = self.model(input_ids=self.ids(text), past_key_values=cache, use_cache=True)
        return out.logits[0, -1].float(), out.past_key_values

    def margin(self, logits):
        lse = self.torch.logsumexp
        return float(lse(logits[self.yes], 0) - lse(logits[self.no], 0))

    def prefix(self):
        return self.head + INTRO + "Past items and the inspector's decisions:\n"

    def history(self, stream, test, T, online=True):
        _, cache = self.extend(None, self.prefix())
        on = []
        for i in stream:
            if online:
                on.append(self.margin(self.extend(copy.deepcopy(cache), query(i) + self.tail)[0]))
            _, cache = self.extend(cache, case_line(i, T))
        held = [self.margin(self.extend(copy.deepcopy(cache), query(j) + self.tail)[0]) for j in test]
        return on, held

    def check_cache(self, stream, test, T):
        """The cached pieces against one full prompt: same tokens, same yes-no margin."""
        pieces = [self.prefix()] + [case_line(i, T) for i in stream] + [query(test[0]) + self.tail]
        full = self.ids("".join(pieces))
        joined = self.torch.cat([self.ids(p) for p in pieces], 1)
        same = full.shape == joined.shape and bool((full == joined).all())
        with self.torch.no_grad():
            m_full = self.margin(self.model(input_ids=full).logits[0, -1].float())
        m_cached = self.history(stream, test[:1], T, online=False)[1][0]
        return {"same_tokens": same, "margin_full": m_full, "margin_cached": m_cached}

    def propose(self, prompt, n, seed):
        x = self.ids(self.head + prompt + self.tail)
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            out = self.model.generate(x, attention_mask=self.torch.ones_like(x), do_sample=True,
                                      num_return_sequences=n, pad_token_id=self.tok.pad_token_id or
                                      self.tok.eos_token_id, **SAMPLING)
        return [self.tok.decode(o[x.shape[1]:], skip_special_tokens=True) for o in out]

    def propose_batch(self, prompts, n, seeds):
        return [self.propose(p, n, s) for p, s in zip(prompts, seeds)]


class Stub:
    """No model: random margins with a weak pull toward the hidden rule, and proposals drawn from the hidden rule,
    the decoy, distractors and junk. Only for checking the bookkeeping."""

    def __init__(self):
        self.task = None

    def history(self, stream, test, T, online=True):
        rng = random.Random(len(stream) + sum(stream))
        return ([rng.gauss(0.3 * (2 * T[i] - 1), 1) for i in stream] if online else [],
                [rng.gauss(0.6 * (2 * T[j] - 1), 1) for j in test])

    def check_cache(self, stream, test, T):
        return {"same_tokens": True, "margin_full": 0.0, "margin_cached": 0.0}

    def propose(self, prompt, n, seed):
        rng = random.Random(seed)
        pool = [self.task["rule"], self.task["decoy"], "size > 4", "color == 'red' or shape == 'star'",
                "```python\nmaterial == 'glass'\n```", "The rule is about colour.", "len(color) > 3"]
        return [rng.choice(pool) for _ in range(n)]

    def propose_batch(self, prompts, n, seeds):
        return [self.propose(p, n, s) for p, s in zip(prompts, seeds)]


def logsumexp(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


class Endpoint:
    """A model behind an OpenAI-compatible chat endpoint (vLLM or SGLang, e.g. on Modal), with thinking switched off.
    Agent A's yes-no margin comes from the top-20 log-probabilities of the first answer token; a side missing from
    the top 20 counts as 1 below the lowest listed. Calls run in parallel; each prompt is the same text the local
    model sees, wrapped by the server's chat template."""

    def __init__(self, base_url, model, workers):
        self.base, self.workers = base_url.rstrip("/"), workers
        self.headers = {"Content-Type": "application/json"}
        if os.environ.get("ENDPOINT_TOKEN"):
            self.headers["Authorization"] = "Bearer " + os.environ["ENDPOINT_TOKEN"]
        if os.environ.get("MODAL_KEY") and os.environ.get("MODAL_SECRET"):
            self.headers.update({"Modal-Key": os.environ["MODAL_KEY"], "Modal-Secret": os.environ["MODAL_SECRET"]})
        self.lock, self.calls, self.off_format = threading.Lock(), 0, 0
        try:
            self.served = [m["id"] for m in self.request("GET", "/models")["data"]]
        except (SystemExit, KeyError, TypeError):
            self.served = [model]
        self.model = model if model in self.served else self.served[0]

    def request(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.base + path, data=data, headers=self.headers, method=method)
                with urllib.request.urlopen(req, timeout=600) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code not in (408, 429, 500, 502, 503, 504) or attempt == 5:
                    raise SystemExit(f"endpoint {method} {path}: HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:500]}")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if attempt == 5:
                    raise SystemExit(f"endpoint {method} {path}: {e}")
            time.sleep(2 ** attempt)

    def chat(self, content, **kw):
        body = {"model": self.model, "messages": [{"role": "user", "content": content}],
                "chat_template_kwargs": {"enable_thinking": False}, **kw}
        return self.request("POST", "/chat/completions", body)

    def answer(self, content):
        r = self.chat(content, max_tokens=1, temperature=0, logprobs=True, top_logprobs=20)
        top = r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        lp = {"yes": [], "no": []}
        for t in top:
            w = t["token"].strip().lower()
            if w in lp:
                lp[w].append(t["logprob"])
        floor = min(t["logprob"] for t in top) - 1.0
        with self.lock:
            self.calls += 1
            self.off_format += not (lp["yes"] or lp["no"])
        return logsumexp(lp["yes"] or [floor]) - logsumexp(lp["no"] or [floor]), top

    def map(self, f, xs):
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(f, xs))

    def history(self, stream, test, T, online=True):
        head = INTRO + "Past items and the inspector's decisions:\n"
        prompts = [head + "".join(case_line(j, T) for j in stream[:k]) + query(i)
                   for k, i in enumerate(stream)] if online else []
        full = head + "".join(case_line(j, T) for j in stream)
        prompts += [full + query(j) for j in test]
        m = self.map(lambda p: self.answer(p)[0], prompts)
        return m[:len(prompts) - len(test)], m[len(prompts) - len(test):]

    def propose_batch(self, prompts, n, seeds):
        def one(ps):
            r = self.chat(ps[0], n=n, seed=ps[1], max_tokens=SAMPLING["max_new_tokens"], temperature=SAMPLING["temperature"],
                          top_p=SAMPLING["top_p"], top_k=SAMPLING["top_k"])
            return [c["message"].get("content") or "" for c in r["choices"]]
        return self.map(one, list(zip(prompts, seeds)))

    def check_cache(self, stream, test, T):
        """Two probe calls: does the model answer yes or no as its first token, and does it write rules?"""
        m, top = self.answer(INTRO + "Past items and the inspector's decisions:\n" +
                             "".join(case_line(j, T) for j in stream[:10]) + query(test[0]))
        replies = self.propose_batch([proposal_prompt(stream[:10], T)], 2, [1])[0]
        first = top[0]["token"].strip().lower()
        return {"served_models": self.served, "model": self.model, "first_token": top[0]["token"],
                "answers_yes_no": first in ("yes", "no"), "margin": m,
                "top_tokens": [(t["token"], round(t["logprob"], 3)) for t in top[:6]],
                "proposals": replies, "parsed_rules": [extract_rule(r) for r in replies]}


# ---------------------------------------------------------------- agents B (ledger) and C (revise)

def proposal_prompt(window, T):
    lines = "".join(case_line(i, T) for i in window)
    return (INTRO + "Some items and the inspector's decisions:\n" + lines + "\nGuess the hidden rule. Write it as one "
            "Python boolean expression over the variables color, shape, size and material, for example: "
            "shape == 'star' or size < 3\nAnswer with the expression only.")


def proposals(llm, stream, T, seed):
    """Every 5 cases, PER_ROUND rules written from the last WINDOW cases. Shared by B and C."""
    ts = list(range(EVERY, len(stream) + 1, EVERY))
    replies = llm.propose_batch([proposal_prompt(stream[max(0, t - WINDOW):t], T) for t in ts], PER_ROUND,
                                [seed + t for t in ts])
    return {t: [{"reply": r[:300], "rule": extract_rule(r)} for r in rs] for t, rs in zip(ts, replies)}


def score(rule, items, T):
    v = compile_rule(rule)
    return sum(v[i] == T[i] for i in items)


def ledger_best(pool, seen, T):
    """B's choice: most of all seen cases right; ties by length, then text, so the choice ignores order."""
    if not pool:
        return None
    return min(pool, key=lambda r: (-score(r, seen, T), len(r), r))


def decide(rule, i, seen, T):
    if rule is None:                          # no rule yet: the majority of seen decisions, 'no' on a tie
        return 2 * sum(T[j] for j in seen) > len(seen)
    return compile_rule(rule)[i]


def rule_agents(stream, test, T, rounds):
    pool, c_rule, on_b, on_c = [], None, [], []
    for t, i in enumerate(stream, start=1):
        seen = stream[:t - 1]
        on_b.append(decide(ledger_best(pool, seen, T), i, seen, T) == T[i])
        on_c.append(decide(c_rule, i, seen, T) == T[i])
        if t in rounds:
            new = [p["rule"] for p in rounds[t] if p["rule"]]
            pool += [r for r in dict.fromkeys(new) if r not in pool]
            window = stream[max(0, t - WINDOW):t]
            for r in new:                     # C swaps only for a strictly better fit to the cases it remembers
                if c_rule is None or score(r, window, T) > score(c_rule, window, T):
                    c_rule = r
    b_rule = ledger_best(pool, stream, T)
    out = {}
    for name, rule, on in (("B", b_rule, on_b), ("C", c_rule, on_c)):
        held = [decide(rule, j, stream, T) == T[j] for j in test]
        out[name] = {"test_acc": 100 * sum(held) / len(held), "online_acc": 100 * sum(on) / len(on),
                     "late_online_acc": 100 * sum(on[-20:]) / 20, "rule": rule}
    out["pool_size"] = len(pool)
    return out, pool


def median_acc(margins, items, T):
    """Yes for the items scoring above the median: the stream shows half the items are accepted."""
    m = statistics.median(margins)
    return 100 * sum((x > m) == T[j] for x, j in zip(margins, items)) / len(items)


def zero_acc(margins, items, T):
    return 100 * sum((x > 0) == T[j] for x, j in zip(margins, items)) / len(items)


# ---------------------------------------------------------------- one task

def run_task(llm, task, k, log):
    T, test = task["T"], task["test"]
    res = {"id": task["id"], "rule": task["rule"], "decoy": task["decoy"], "A": {}, "B": {}, "C": {}, "streams": {}}
    calls0, off0 = getattr(llm, "calls", 0), getattr(llm, "off_format", 0)
    _, held = llm.history([], test, T, online=False)
    res["A_zero_shot"] = {"test_acc": median_acc(held, test, T), "test_acc_threshold0": zero_acc(held, test, T)}
    _, held = llm.history(sorted(task["orders"]["random1"], key=item_text), test, T, online=False)
    res["A_sorted"] = {"test_acc": median_acc(held, test, T), "test_acc_threshold0": zero_acc(held, test, T)}
    for o_i, o in enumerate(ORDERS):
        stream = task["orders"][o]
        t0 = time.time()
        on, held = llm.history(stream, test, T)
        res["A"][o] = {"test_acc": median_acc(held, test, T), "test_acc_threshold0": zero_acc(held, test, T),
                       "online_acc": zero_acc(on, stream, T), "late_online_acc": zero_acc(on[-20:], stream[-20:], T)}
        if isinstance(llm, Stub):
            llm.task = task
        rounds = proposals(llm, stream, T, seed=1_000_000 * k + 1000 * o_i)
        bc, pool = rule_agents(stream, test, T, rounds)
        for name in ("B", "C"):
            r = bc[name]
            r["rule_good"] = r["rule"] is not None and agreement(compile_rule(r["rule"]), T) >= GOOD
            r["rule_decoy"] = r["rule"] is not None and agreement(compile_rule(r["rule"]), task["D"]) >= GOOD
            res[name][o] = r
        all_p = [p for ps in rounds.values() for p in ps]
        res["streams"][o] = {
            "proposals": len(all_p), "valid": sum(p["rule"] is not None for p in all_p), "pool_size": bc["pool_size"],
            "pool_has_good_rule": any(agreement(compile_rule(r), T) >= GOOD for r in pool),
            "pool_has_decoy": any(agreement(compile_rule(r), task["D"]) >= GOOD for r in pool),
            "rounds": {str(t): ps for t, ps in rounds.items()}}
        log(f"  {task['id']} {o:<11} A {res['A'][o]['test_acc']:5.1f}  B {res['B'][o]['test_acc']:5.1f}  "
            f"C {res['C'][o]['test_acc']:5.1f}  valid {res['streams'][o]['valid']}/{len(all_p)}  "
            f"good-in-pool {res['streams'][o]['pool_has_good_rule']}  B rule: {res['B'][o]['rule']}  "
            f"({time.time() - t0:.0f}s)")
    res["A_calls"], res["A_off_format"] = getattr(llm, "calls", 0) - calls0, getattr(llm, "off_format", 0) - off0
    return res


# ---------------------------------------------------------------- summary and the pre-registered lines

def mean(x):
    return sum(x) / len(x)


def se(x):
    return statistics.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")


def arm_stats(tasks, arm):
    rnd = [mean([t[arm][o]["test_acc"] for o in RANDOM]) for t in tasks]
    first = [r - t[arm]["decoy_first"]["test_acc"] for r, t in zip(rnd, tasks)]
    last = [r - t[arm]["decoy_last"]["test_acc"] for r, t in zip(rnd, tasks)]
    spread = [max(t[arm][o]["test_acc"] for o in RANDOM) - min(t[arm][o]["test_acc"] for o in RANDOM) for t in tasks]
    return {"learning": mean(rnd) - 50, "learning_se": se(rnd), "random_acc": mean(rnd),
            "decoy_first_acc": mean([t[arm]["decoy_first"]["test_acc"] for t in tasks]),
            "decoy_last_acc": mean([t[arm]["decoy_last"]["test_acc"] for t in tasks]),
            "gap_first": mean(first), "gap_first_se": se(first), "gap_last": mean(last), "gap_last_se": se(last),
            "order_gap": max(mean(first), mean(last)), "random_spread": mean(spread),
            "online_acc": mean([mean([t[arm][o]["online_acc"] for o in RANDOM]) for t in tasks]),
            "late_online_acc": mean([mean([t[arm][o]["late_online_acc"] for o in RANDOM]) for t in tasks])}


def summarise(tasks):
    S = {a: arm_stats(tasks, a) for a in "ABC"}
    S["A"]["random_acc_threshold0"] = mean([mean([t["A"][o]["test_acc_threshold0"] for o in RANDOM]) for t in tasks])
    S["A_zero_shot_acc"] = mean([t["A_zero_shot"]["test_acc"] for t in tasks])
    S["A_sorted_acc"] = mean([t["A_sorted"]["test_acc"] for t in tasks])
    streams = [s for t in tasks for s in t["streams"].values()]
    S["proposer_good_rate"] = mean([s["pool_has_good_rule"] for s in streams])
    S["A_calls"] = sum(t.get("A_calls", 0) for t in tasks)
    S["off_format_rate"] = sum(t.get("A_off_format", 0) for t in tasks) / max(1, S["A_calls"])
    S["proposal_valid_rate"] = sum(s["valid"] for s in streams) / max(1, sum(s["proposals"] for s in streams))
    for a in "BC":
        S[a]["final_rule_good_rate"] = mean([t[a][o]["rule_good"] for t in tasks for o in ORDERS])
        S[a]["final_rule_decoy_rate"] = mean([t[a][o]["rule_decoy"] for t in tasks for o in ORDERS])
        S[a]["decoy_first_rule_decoy_rate"] = mean([t[a]["decoy_first"]["rule_decoy"] for t in tasks])
    return S


def verdict(S, hard=False):
    A, B, C = S["A"], S["B"], S["C"]
    measured = S.get("off_format_rate", 0.0) <= OFF_FORMAT
    v1, v2 = measured and A["learning"] >= 5, S["proposer_good_rate"] >= 0.25
    out = [f"Check A learns (learning >= 5): {'yes' if v1 else 'no'} ({A['learning']:.1f}).",
           f"Check the proposer (good rule in >= 25% of streams): {'yes' if v2 else 'no'} "
           f"({100 * S['proposer_good_rate']:.0f}%)."]
    if not measured:
        out.append(f"INCONCLUSIVE for A: {100 * S['off_format_rate']:.0f}% of A's answers were neither yes nor no "
                   f"(line: at most {100 * OFF_FORMAT:.0f}%), so A is not measured and the stop and pass lines are not "
                   "applied.")
    elif not v1:
        out.append("INCONCLUSIVE for A: A does not learn at this scale, so its order gap means nothing and the stop "
                   "and pass lines are not applied.")
    elif A["random_acc"] >= CEILING:
        out.append(f"TOO EASY: A scores {A['random_acc']:.1f} >= {CEILING} on random orders, so these rules cannot show "
                   "order damage and the stop and pass lines are not applied. "
                   + ("Inconclusive for this model even with the harder rules." if hard else
                      "The harder rule set runs next, as pre-registered."))
    elif A["order_gap"] < 5:
        out.append(f"STOP: A's order gap is {A['order_gap']:.1f} < 5. Order is not what holds back a full-history "
                   "agent here; the route ends.")
    elif not v2:
        out.append(f"Order matters for A (gap {A['order_gap']:.1f} >= 5), but the proposer check failed: B and C are "
                   "inconclusive at this scale.")
    else:
        ok = B["order_gap"] <= A["order_gap"] / 2 and B["learning"] >= A["learning"] + 5
        out.append(f"{'PASS' if ok else 'FAIL'}: A's gap {A['order_gap']:.1f}, B's gap {B['order_gap']:.1f} "
                   f"(needs <= {A['order_gap'] / 2:.1f}); B's learning {B['learning']:.1f} against A's "
                   f"{A['learning']:.1f} (needs >= {A['learning'] + 5:.1f}).")
        if ok:
            credit = B["learning"] >= C["learning"] + 5 or (C["order_gap"] >= 5 and B["order_gap"] <= C["order_gap"] / 2)
            out.append("Credit: the order-free ledger." if credit else
                       "Credit: writing rules as code and testing them (published), not the ledger.")
    if v1 and S["A_sorted_acc"] >= A["random_acc"] - 2:
        out.append(f"Trivial cure: A with the history in a fixed order scores {S['A_sorted_acc']:.1f} against "
                   f"{A['random_acc']:.1f} for random orders, so a fixed order removes A's order damage for free.")
    if v2:
        sec = C["order_gap"] >= 5 and B["order_gap"] <= C["order_gap"] / 2
        out.append(f"Secondary: C's gap {C['order_gap']:.1f}, B's {B['order_gap']:.1f}; "
                   + ("order hurts the bounded-memory agent and the ledger removes it." if sec else
                      "no bounded-memory finding by the pre-registered line."))
    return out


def write(out_dir, R, done, total):
    tasks = R["tasks"]
    L = ["# Order gap: does the order of experience change what an agent learns?", "",
         f"Model {R['model']}, {R['rules']} rules, {done} of {total} tasks done. Pre-registration: "
         "reports/Order gap 2026.md.", ""]
    if tasks:
        S = summarise(tasks)
        R["summary"] = S
        L += ["Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of "
              "(random - decoy first) and (random - decoy last). Learning = random-order accuracy - 50.", "",
              "| Agent | Random orders | Decoy first | Decoy last | Learning | Gap first | Gap last | Order gap | "
              "Spread over random orders | Late online |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for a, name in (("A", "A, history"), ("B", "B, ledger"), ("C", "C, revise")):
            s = S[a]
            L.append(f"| {name} | {s['random_acc']:.1f} | {s['decoy_first_acc']:.1f} | {s['decoy_last_acc']:.1f} | "
                     f"{s['learning']:.1f} +/- {s['learning_se']:.1f} | {s['gap_first']:.1f} +/- {s['gap_first_se']:.1f} | "
                     f"{s['gap_last']:.1f} +/- {s['gap_last_se']:.1f} | {s['order_gap']:.1f} | "
                     f"{s['random_spread']:.1f} | {s['late_online_acc']:.1f} |")
        L += ["", f"A with no history: {S['A_zero_shot_acc']:.1f}. A with the history sorted: {S['A_sorted_acc']:.1f}. "
              f"A at threshold zero instead of the median, random orders: {S['A']['random_acc_threshold0']:.1f}.",
              (f"A's answers that were neither yes nor no: {100 * S['off_format_rate']:.1f}% of {S['A_calls']:,} "
               "calls. " if S["A_calls"] else "") +
              f"Proposals that are valid rules: {100 * S['proposal_valid_rate']:.0f}%. Streams whose proposals include "
              f"a good rule: {100 * S['proposer_good_rate']:.0f}%.",
              f"Final rule good / decoy: B {100 * S['B']['final_rule_good_rate']:.0f}% / "
              f"{100 * S['B']['final_rule_decoy_rate']:.0f}%, C {100 * S['C']['final_rule_good_rate']:.0f}% / "
              f"{100 * S['C']['final_rule_decoy_rate']:.0f}%. Decoy kept after a decoy-first stream: "
              f"B {100 * S['B']['decoy_first_rule_decoy_rate']:.0f}%, C {100 * S['C']['decoy_first_rule_decoy_rate']:.0f}%.",
              "", "| Task | Rule | A random / first / last | B random / first / last | C random / first / last |",
              "|---|---|---|---|---|"]
        for t in tasks:
            cells = [" / ".join(f"{x:.0f}" for x in (mean([t[a][o]['test_acc'] for o in RANDOM]),
                                                     t[a]['decoy_first']['test_acc'], t[a]['decoy_last']['test_acc']))
                     for a in "ABC"]
            L.append(f"| {t['id']} | `{t['rule']}` | " + " | ".join(cells) + " |")
        L += ["", "## Verdict" + ("" if done == total else " (provisional: not all tasks done)"), ""]
        L += [f"- {v}" for v in verdict(S, R["rules"] == "hard")]
        L += ["", f"Check before the run: {R['check']}"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=True)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return L


def run_set(llm, tag, rules_name, args, log):
    rules, prefix = (RULES, "rule") if rules_name == "standard" else (RULES_HARD, "hard")
    suffix = "" if rules_name == "standard" else "_hard"
    out = (args.out + suffix) if args.out else os.path.join(ROOT, "results", f"order_gap_{tag}{suffix}")
    tasks = [make_task(r, d, rules, prefix) for r in range(len(rules)) for d in range(DRAWS)][:args.tasks]
    R = {"model": tag, "rules": rules_name, "settings": {"orders": ORDERS, "every": EVERY, "window": WINDOW,
         "per_round": PER_ROUND, "sampling": SAMPLING, "good": GOOD, "ceiling": CEILING, "off_format": OFF_FORMAT},
         "tasks": []}
    t = tasks[0]
    R["check"] = c = llm.check_cache(t["orders"]["random1"], t["test"], t["T"])
    log(f"check: {json.dumps(c, ensure_ascii=True)}")
    if c.get("same_tokens") and abs(c["margin_full"] - c["margin_cached"]) > 0.2:
        raise SystemExit("the cached history gives a different answer from the full prompt: stopping")
    if c.get("answers_yes_no") is False:
        raise SystemExit("the model's first answer token is not yes or no (is thinking still on?): stopping")
    if args.check:
        raise SystemExit("check done; no run")
    t0 = time.time()
    for k, task in enumerate(tasks):
        R["tasks"].append(run_task(llm, task, k, log))
        write(out, R, k + 1, len(tasks))
        log(f"task {k + 1}/{len(tasks)} done, {(time.time() - t0) / 60:.1f} min")
    log("\n".join(write(out, R, len(tasks), len(tasks))))
    log(f"wrote {out}")
    return R["summary"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--device", default=None)
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. https://...modal.run/v1")
    p.add_argument("--workers", type=int, default=16, help="parallel endpoint calls")
    p.add_argument("--check", action="store_true", help="probe the model and stop")
    p.add_argument("--rules", choices=["standard", "hard"], default="standard")
    p.add_argument("--tasks", type=int, default=len(RULES) * DRAWS, help="fewer tasks, for debugging only")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def log(s):
        print(s, flush=True)

    if args.stub:
        llm, tag = Stub(), "stub"
    elif args.base_url:
        llm, tag = Endpoint(args.base_url, args.model, args.workers), args.model.split("/")[-1]
        log(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}")
    else:
        import torch
        dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        log(f"loading {args.model} on {dev}")
        llm, tag = LLM(args.model, dev), args.model.split("/")[-1]
    S = run_set(llm, tag, args.rules, args, log)
    if (args.rules == "standard" and args.tasks == len(RULES) * DRAWS and S.get("off_format_rate", 0) <= OFF_FORMAT
            and S["A"]["learning"] >= 5 and S["A"]["random_acc"] >= CEILING):
        log("A is at the ceiling on the standard rules: running the harder rules, as pre-registered")
        run_set(llm, tag, "hard", args, log)


if __name__ == "__main__":
    main()
