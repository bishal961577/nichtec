"""The one redesign: the model names the ingredients, code combines them. Pre-registered in section 7 of
reports/Learning on the job from first principles 2026.md.

Task: UCI Zoo animals given by name only; a hidden rule over two traits (ten rules x two draws = 20 tasks); a
40-case stream and a 40-animal test set per task, both 20 yes and 20 no. Agents, all on the same model:
  A  full history in the prompt; yes/no from the top-20 log-probabilities, median threshold;
  W  the model writes the rule as a sentence (4 samples) and judges it on every animal; code keeps the rule with the
     fewest stream errors;
  E  the model names yes/no questions about traits (4 samples, up to 16 kept) and answers them for every animal;
     code tries every rule of one or two questions and keeps the one with the shortest description:
     errors x ln 19 + questions x ln(2K) + ln 2 for a joining word.
References: the oracle (same combining over the recorded traits) and E with the 12 true questions read by the model.

  python brain/ingredients.py --stub
  python brain/ingredients.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-27B-FP8 --check
  python brain/ingredients.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-27B-FP8    # test 1
  python brain/ingredients.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-4B         # test 2
"""
import argparse
import collections
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
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402

ZOO_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data"
ZOO_SHA = "cddc71c26ab9bc82795b8f4ff114cade41885d92720c6af29ffb69bcf73f0315"
COLS = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes",
        "venomous", "fins", "legs", "tail", "domestic", "catsize", "type"]
QUESTIONS = {"hair": "Does it have hair?", "feathers": "Does it have feathers?", "eggs": "Does it lay eggs?",
             "milk": "Does it produce milk?", "airborne": "Can it fly?", "aquatic": "Does it live in water?",
             "toothed": "Does it have teeth?", "backbone": "Does it have a backbone?",
             "breathes": "Does it breathe air?", "venomous": "Is it venomous?", "fins": "Does it have fins?",
             "tail": "Does it have a tail?"}
TRAITS = list(QUESTIONS)
RULES = [  # (trait, wanted value, joining word, trait, wanted value), fixed in section 7
    ("airborne", 1, "or", "aquatic", 1), ("eggs", 1, "and", "backbone", 1), ("airborne", 1, "or", "fins", 1),
    ("airborne", 0, "and", "breathes", 1), ("fins", 1, "or", "tail", 0), ("milk", 1, "or", "backbone", 0),
    ("milk", 0, "and", "tail", 1), ("airborne", 1, "or", "tail", 0), ("aquatic", 0, "and", "tail", 1),
    ("feathers", 1, "or", "breathes", 0)]
DRAWS, HALF = 2, 20
SAMPLES, MAX_Q = 4, 16
GEN = {"temperature": 0.8, "top_p": 0.95, "top_k": 50}
ERR = math.log(19)
INTRO = "A zookeeper sorts animals into two groups, yes and no, by a fixed hidden rule about the animals' traits.\n\n"


def rule_text(r):
    a, va, op, b, vb = r
    return f"{'' if va else 'not '}{a} {op} {'' if vb else 'not '}{b}"


def holds(r, t):
    a, va, op, b, vb = r
    x, y = t[a] == va, t[b] == vb
    return (x and y) if op == "and" else (x or y)


def load_zoo():
    path = os.path.join(G.ROOT, "data", "zoo", "zoo.data")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(ZOO_URL, path)
    raw = open(path, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    rows = [l.split(",") for l in raw.decode().splitlines() if l.strip()]
    count = collections.Counter(r[0] for r in rows)
    zoo = {r[0]: dict(zip(COLS, map(int, r[1:]))) for r in rows if count[r[0]] == 1}
    return zoo, sha


def make_task(zoo, k, draw):
    r = RULES[k]
    rng = random.Random(100 * k + draw)
    names = sorted(zoo)
    yes = [n for n in names if holds(r, zoo[n])]
    no = [n for n in names if not holds(r, zoo[n])]
    rng.shuffle(yes), rng.shuffle(no)
    stream, test = yes[:HALF] + no[:HALF], yes[HALF:2 * HALF] + no[HALF:2 * HALF]
    rng.shuffle(stream)
    label = {n: holds(r, zoo[n]) for n in names}
    return {"id": f"rule{k + 1}_draw{draw + 1}", "rule": rule_text(r), "stream": stream, "test": test, "label": label}


# ---------------------------------------------------------------- prompts

def yn(b):
    return "yes" if b else "no"


def a_prompt(task, name):
    hist = "".join(f"{n}: {yn(task['label'][n])}\n" for n in task["stream"])
    return (INTRO + "Past animals and the zookeeper's decisions:\n" + hist +
            f"New animal: {name}\nIs it in the yes group? Answer yes or no.")


def cases(task):
    return "".join(f"{n}: {yn(task['label'][n])}\n" for n in sorted(task["stream"]))


def w_prompt(task):
    return (INTRO + "All animals seen so far and the zookeeper's decisions:\n" + cases(task) +
            "\nWrite the hidden rule as one short sentence about the animals' traits. Answer with the rule only.")


def e_prompt(task):
    return (INTRO + "All animals seen so far and the zookeeper's decisions:\n" + cases(task) +
            "\nList up to 8 short yes-or-no questions about an animal's traits that could help explain which "
            "animals are in the yes group. Write one question per line, nothing else.")


def judge_prompt(name, rule):
    return f"Animal: {name}\nRule: {rule}\nDoes this animal satisfy the rule? Answer yes or no."


def read_prompt(name, q):
    return f"Animal: {name}\nQuestion: {q}\nAnswer yes or no."


def parse_questions(replies):
    out, seen = [], set()
    for rep in replies:
        for line in rep.splitlines():
            q = re.sub(r"^[\s\-\*•\d\.\)\(]+", "", line).strip().strip("*").strip()
            key = re.sub(r"[^a-z0-9]", "", q.lower())
            if q.endswith("?") and 3 < len(q) <= 120 and key not in seen:
                seen.add(key)
                out.append(q)
    return out[:MAX_Q]


def parse_rule(reply):
    line = next((l.strip() for l in reply.strip().splitlines() if l.strip()), "")
    return line[:200]


# ---------------------------------------------------------------- combining (code)

def combine(feats, task):
    """Every rule of one or two questions (each possibly negated; two joined by and/or). Cost in nats: stream errors
    x ln 19 + questions x ln(2K) + ln 2 for a joining word. Ties: fewer questions, then text."""
    qs = sorted(feats)
    K = max(1, len(qs))
    stream, lab = task["stream"], task["label"]
    y = [lab[n] for n in stream]
    lits = [(f"{'' if s else 'not '}[{q}]", [feats[q][n] == s for n in stream], q, s) for q in qs for s in (True, False)]
    best = None
    for text, v, q, s in lits:
        c = (ERR * sum(a != b for a, b in zip(v, y)) + math.log(2 * K), 1, text)
        if best is None or c < best[0]:
            best = (c, ((q, s),), "single")
    for (t1, v1, q1, s1), (t2, v2, q2, s2) in itertools.combinations(lits, 2):
        if q1 == q2:
            continue
        for op in ("and", "or"):
            v = [(a and b) if op == "and" else (a or b) for a, b in zip(v1, v2)]
            c = (ERR * sum(a != b for a, b in zip(v, y)) + 2 * math.log(2 * K) + math.log(2), 2, f"{t1} {op} {t2}")
            if c < best[0]:
                best = (c, ((q1, s1), (q2, s2)), op)
    (cost, _, text), lits_used, op = best

    def apply(n):
        vals = [feats[q][n] == s for q, s in lits_used]
        return vals[0] if op == "single" else (all(vals) if op == "and" else any(vals))
    acc = 100 * sum(apply(n) == lab[n] for n in task["test"]) / len(task["test"])
    return {"rule": text, "cost": round(cost, 2), "test_acc": acc, "questions": [q for q, _ in lits_used]}


# ---------------------------------------------------------------- the model, the stub and the cache

class Answers:
    """yes/no margins for prompts, cached on disk by prompt text so a stopped run can resume."""

    def __init__(self, llm, path):
        self.llm, self.path, self.lock = llm, path, threading.Lock()
        self.m = json.load(open(path, encoding="utf-8")) if path and os.path.exists(path) else {}

    def get(self, prompts, log, label):
        todo = sorted({p for p in prompts if p not in self.m})
        log(f"  {label}: {len(prompts):,} answers needed, {len(todo):,} new")
        for i in range(0, len(todo), 2000):
            chunk = todo[i:i + 2000]
            got = self.llm.map(lambda p: self.llm.answer(p)[0], chunk)
            with self.lock:
                self.m.update(zip(chunk, got))
            if self.path:
                with open(self.path, "w", encoding="utf-8") as f:
                    json.dump(self.m, f)
            log(f"    {min(i + 2000, len(todo)):,}/{len(todo):,}")
        return {p: self.m[p] for p in prompts}


def generate(llm, prompt, seed, max_tokens):
    r = llm.chat(prompt, n=SAMPLES, seed=seed, max_tokens=max_tokens, **GEN)
    return [c["message"].get("content") or "" for c in r["choices"]]


class Stub:
    """No model: reads traits by keywords with 5% noise, names a few true and false questions, writes a fixed rule,
    and answers the history prompts at random. Only for checking the bookkeeping."""
    KEYS = {"hair": "hair", "feather": "feathers", "egg": "eggs", "milk": "milk", "fly": "airborne",
            "water": "aquatic", "teeth": "toothed", "backbone": "backbone", "breathe": "breathes",
            "venom": "venomous", "fin": "fins", "tail": "tail"}

    def __init__(self, zoo):
        self.zoo, self.calls, self.off_format = zoo, 0, 0

    def map(self, f, xs):
        return [f(x) for x in xs]

    def answer(self, p):
        self.calls += 1
        rng = random.Random(p)
        m = re.match(r"Animal: (\S+)\n(?:Question|Rule): (.*)\n", p)
        if not m:
            return rng.gauss(0, 1), []
        name, text = m.group(1), m.group(2).lower()
        trait = next((t for k, t in self.KEYS.items() if k in text), None)
        if trait is None:
            return rng.gauss(0, 1), []
        v = self.zoo[name][trait] == 1
        if " not " in f" {text} " or "no " in text[:3]:
            v = not v
        if rng.random() < 0.05:
            v = not v
        return (2.0 if v else -2.0), []

    def chat(self, prompt, n, seed, **kw):
        rng = random.Random(seed)
        if "questions" in prompt:
            pool = list(QUESTIONS.values()) + ["Is it a pet?", "Is it large?", "Is it a mammal?"]
            outs = ["\n".join(f"{i + 1}. {q}" for i, q in enumerate(rng.sample(pool, 6))) for _ in range(n)]
        else:
            outs = ["Animals that can fly or live in water." for _ in range(n)]
        return {"choices": [{"message": {"content": o}} for o in outs]}


# ---------------------------------------------------------------- run, summary, verdict

def summarise(R):
    tasks = R["tasks"]
    S = {a: G.mean([t[a]["test_acc"] for t in tasks]) for a in ("A", "W", "E", "E_true", "oracle")}
    S.update({a + "_se": G.se([t[a]["test_acc"] for t in tasks]) for a in ("A", "W", "E", "E_true")})
    rd = R["reading"]
    S["reading_acc"] = rd["overall"]
    S["off_format_rate"] = R["calls"]["off_format"] / max(1, R["calls"]["answers"])
    S["E_uses_true_trait"] = G.mean([bool(t["E"].get("uses_rule_trait")) for t in tasks])
    return S


def verdict(S, test, ref_a):
    out = [f"Check 1, the task: oracle {S['oracle']:.1f} (line >= 95): {'ok' if S['oracle'] >= 95 else 'FAILED'}.",
           f"Check 2, reading: {S['reading_acc']:.1f}% (line >= 85): {'ok' if S['reading_acc'] >= 85 else 'FAILED'}.",
           f"Check 3, answering: {100 * S['off_format_rate']:.1f}% of answers neither yes nor no (line <= 10): "
           f"{'ok' if S['off_format_rate'] <= 0.10 else 'FAILED'}."]
    if S["oracle"] < 95 or S["reading_acc"] < 85 or S["off_format_rate"] > 0.10:
        return out + ["INCONCLUSIVE: a check failed, so the lines are not applied."]
    E, A, W = S["E"], S["A"], S["W"]
    if test == 1:
        if E >= A + 5 and E >= W + 5:
            out.append(f"PASS: E {E:.1f} is at least 5 points above A {A:.1f} and W {W:.1f}. Splitting the jobs helps.")
        elif E < max(A, W) + 2:
            out.append(f"STOP: E {E:.1f} is less than 2 points above the better of A {A:.1f} and W {W:.1f}. "
                       "Splitting the jobs does not help; this line of work ends.")
        else:
            out.append(f"IN BETWEEN: E {E:.1f}, A {A:.1f}, W {W:.1f}. Report, diagnose, decide with the owner.")
    else:
        if ref_a is None:
            out.append("No 27B result for A was found, so test 2's lines cannot be applied yet.")
        elif E >= ref_a:
            out.append(f"PASS: the 4B model with split jobs ({E:.1f}) matches or beats 27B reading its history "
                       f"({ref_a:.1f}).")
        elif E < ref_a - 5:
            out.append(f"STOP: the 4B model with split jobs ({E:.1f}) is more than 5 points below 27B-A ({ref_a:.1f}).")
        else:
            out.append(f"IN BETWEEN: 4B-E {E:.1f} against 27B-A {ref_a:.1f}.")
    return out


def write(out, R, test, ref_a):
    S = summarise(R)
    R["summary"] = S
    L = [f"# Ingredients test {test}: the model names traits, code combines them", "",
         f"Model {R['model']}. Pre-registration: section 7 of reports/Learning on the job from first principles 2026.md.",
         "", "Test-set accuracy (%) after the 40-case stream, averaged over the 20 tasks (20 yes, 20 no each).", "",
         "| Agent | Accuracy |", "|---|---|",
         f"| A, full history | {S['A']:.1f} +/- {S['A_se']:.1f} |",
         f"| W, whole rule in words, judged by code | {S['W']:.1f} +/- {S['W_se']:.1f} |",
         f"| **E, model names traits, code combines** | **{S['E']:.1f} +/- {S['E_se']:.1f}** |",
         f"| E with the 12 true questions (reference) | {S['E_true']:.1f} +/- {S['E_true_se']:.1f} |",
         f"| Oracle: recorded traits, same combining | {S['oracle']:.1f} |", "",
         f"Reading accuracy on the 12 true questions over 99 animals: {S['reading_acc']:.1f}% "
         f"({', '.join(f'{t} {v:.0f}' for t, v in R['reading']['per_trait'].items())}).",
         f"Tasks where E's rule uses a question that matches one of the hidden rule's traits: {100 * S['E_uses_true_trait']:.0f}%. "
         f"Answer calls: {R['calls']['answers']:,}; neither yes nor no: {100 * S['off_format_rate']:.1f}%.", "",
         "| Task | Hidden rule | A | W | E | E true q. | Oracle | E's rule | W's rule |", "|---|---|---|---|---|---|---|---|---|"]
    for t in R["tasks"]:
        L.append(f"| {t['id']} | {t['rule']} | {t['A']['test_acc']:.0f} | {t['W']['test_acc']:.0f} | "
                 f"{t['E']['test_acc']:.0f} | {t['E_true']['test_acc']:.0f} | {t['oracle']['test_acc']:.0f} | "
                 f"{t['E']['rule']} | {t['W']['rule'][:80]} |")
    L += ["", "## Verdict", ""] + [f"- {v}" for v in verdict(S, test, ref_a)]
    L += ["", f"Check before the run: {R['check']}"]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, ensure_ascii=True)
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return L


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    p.add_argument("--base-url", default=None)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--test", type=int, choices=[1, 2], default=None, help="default: 2 for a 4B model, else 1")
    p.add_argument("--check", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    def log(s):
        print(s, flush=True)
    test = args.test or (2 if "4B" in args.model else 1)
    zoo, sha = load_zoo()
    log(f"zoo: {len(zoo)} animals, sha256 {sha[:12]}… ({'matches' if sha == ZOO_SHA else 'DIFFERS from'} the plan)")
    if args.stub:
        llm, tag = Stub(zoo), "stub"
    elif args.base_url:
        llm, tag = G.Endpoint(args.base_url, args.model, args.workers), args.model.split("/")[-1]
        log(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}")
    else:
        raise SystemExit("needs --base-url or --stub")
    out = args.out or os.path.join(G.ROOT, "results", f"ingredients_{tag}")
    cache = None if args.stub else os.path.join(G.ROOT, "data", f"ingredients_cache_{tag}.json")
    tasks = [make_task(zoo, k, d) for k in range(len(RULES)) for d in range(DRAWS)]
    R = {"model": tag, "test": test, "zoo_sha256": sha, "settings": {"samples": SAMPLES, "max_questions": MAX_Q,
         "gen": GEN, "err_nats": ERR, "half": HALF}, "tasks": []}

    # probe: one history answer and one naming call
    t0 = tasks[0]
    m, top = llm.answer(a_prompt(t0, t0["test"][0]))
    names = parse_questions(generate(llm, e_prompt(t0), 1, 200))
    first = (top[0]["token"] if top else "")
    R["check"] = {"first_token": first, "answers_yes_no": (not top) or first.strip().lower() in ("yes", "no"),
                  "margin": round(m, 3), "questions_named": names[:8]}
    log(f"check: {json.dumps(R['check'], ensure_ascii=True)}")
    if not R["check"]["answers_yes_no"] or not names:
        raise SystemExit("the model does not answer yes/no first or names no questions: stopping")
    if args.check:
        raise SystemExit("check done; no run")
    start = time.time()

    # 1. generation: W's rules and E's questions for every task
    log("generating rules and questions")
    gens = llm.map(lambda x: generate(llm, x[0], x[1], x[2]),
                   [(w_prompt(t), 1000 * i + 1, 80) for i, t in enumerate(tasks)] +
                   [(e_prompt(t), 1000 * i + 2, 200) for i, t in enumerate(tasks)])
    for i, t in enumerate(tasks):
        t["w_rules"] = list(dict.fromkeys(parse_rule(r) for r in gens[i] if parse_rule(r)))
        t["e_questions"] = parse_questions(gens[len(tasks) + i])
        t["w_replies"], t["e_replies"] = gens[i], gens[len(tasks) + i]

    # 2. every yes/no answer, cached
    A = Answers(llm, cache)
    animals = sorted(zoo)
    reading = A.get([read_prompt(n, q) for q in QUESTIONS.values() for n in animals], log, "reading the 12 true questions")
    hist = A.get([a_prompt(t, n) for t in tasks for n in t["test"]], log, "A, history")
    judged = A.get([judge_prompt(n, r) for t in tasks for r in t["w_rules"] for n in t["stream"] + t["test"]], log,
                   "W, judging rules")
    named = A.get([read_prompt(n, q) for t in tasks for q in t["e_questions"] for n in t["stream"] + t["test"]], log,
                  "E, reading named questions")

    # 3. scoring
    per = {}
    for tr, q in QUESTIONS.items():
        per[tr] = 100 * G.mean([(reading[read_prompt(n, q)] > 0) == (zoo[n][tr] == 1) for n in animals])
    R["reading"] = {"per_trait": per, "overall": G.mean(list(per.values()))}
    # which named questions behave like a true trait (>= 90% agreement with it over the task's 80 animals)
    match = {}
    for t in tasks:
        res = {"id": t["id"], "rule": t["rule"], "w_rules": t["w_rules"], "w_replies": t["w_replies"],
               "e_questions": t["e_questions"], "e_replies": t["e_replies"]}
        ms = [hist[a_prompt(t, n)] for n in t["test"]]
        res["A"] = {"test_acc": G.median_acc(ms, list(range(len(ms))), [t["label"][n] for n in t["test"]])}
        best_w = None
        for r in t["w_rules"]:
            errs = sum((judged[judge_prompt(n, r)] > 0) != t["label"][n] for n in t["stream"])
            if best_w is None or (errs, len(r), r) < best_w[0]:
                best_w = ((errs, len(r), r), r)
        if best_w:
            r = best_w[1]
            acc = 100 * G.mean([(judged[judge_prompt(n, r)] > 0) == t["label"][n] for n in t["test"]])
            res["W"] = {"rule": r, "stream_errors": best_w[0][0], "test_acc": acc}
        else:
            res["W"] = {"rule": "", "stream_errors": None, "test_acc": 50.0}
        items = t["stream"] + t["test"]
        feats = {q: {n: named[read_prompt(n, q)] > 0 for n in items} for q in t["e_questions"]}
        res["E"] = combine(feats, t) if feats else {"rule": "", "test_acc": 50.0, "questions": []}
        res["E_true"] = combine({q: {n: reading[read_prompt(n, q)] > 0 for n in items} for q in QUESTIONS.values()}, t)
        res["oracle"] = combine({tr: {n: zoo[n][tr] == 1 for n in items} for tr in TRAITS}, t)
        task_match = {}
        for q in t["e_questions"]:
            m_tr = [tr for tr in TRAITS if G.mean([feats[q][n] == (zoo[n][tr] == 1) for n in items]) >= 0.9
                    or G.mean([feats[q][n] != (zoo[n][tr] == 1) for n in items]) >= 0.9]
            task_match[q] = m_tr
            match[q] = sorted(set(match.get(q, [])) | set(m_tr))
        res["question_traits"] = task_match
        rule_traits = {RULES[int(t["id"].split("_")[0][4:]) - 1][0], RULES[int(t["id"].split("_")[0][4:]) - 1][3]}
        res["E"]["uses_rule_trait"] = any(set(task_match.get(q, [])) & rule_traits for q in res["E"]["questions"])
        R["tasks"].append(res)
    R["true_question_match"] = match
    R["calls"] = {"answers": getattr(llm, "calls", 0), "off_format": getattr(llm, "off_format", 0),
                  "minutes": round((time.time() - start) / 60, 1)}
    ref_a = None
    if test == 2:
        ref = os.path.join(G.ROOT, "results", "ingredients_Qwen3.5-27B-FP8", "result.json")
        if os.path.exists(ref):
            ref_a = json.load(open(ref, encoding="utf-8"))["summary"]["A"]
    log("\n".join(write(out, R, test, ref_a)))
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
