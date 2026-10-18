"""Agent D, the scientist: the model guesses, code judges. Pre-registered in
reports/Learning on the job from first principles 2026.md (sections 3 and 4).

Same 12 tasks, 5 orders and test sets as brain/order_gap.py. D keeps every case and every guess. Each guess is judged
on all cases seen by its description length: 2.94 nats per error (a rule with 5% noise) plus ln(58) = 4.06 nats per
condition. After every 5 cases the model sees all cases so far in a fixed sorted order, D's current best rule and the
cases it gets wrong, and writes 4 rules. D answers with the best-judged rule. Free references: brute force (the same
judging over all 2,578 one- and two-condition rules) and D-count (D's guesses judged by count alone, like agent B).

  python brain/scientist.py --stub                                              # bookkeeping, no model
  python brain/scientist.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-27B-FP8 --check
  python brain/scientist.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-27B-FP8    # test 1
  python brain/scientist.py --base-url https://...modal.direct/v1 --model Qwen/Qwen3.5-4B         # test 2
"""
import argparse
import ast
import itertools
import json
import math
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import order_gap as G  # noqa: E402

ERR = math.log(0.95 / 0.05)                   # nats per case a rule gets wrong
COND = math.log(58)                           # nats per condition: one of the 58 single conditions
A_27B, B_27B = 83.7, 81.2                     # order-gap results on Qwen3.5-27B, random orders (the comparison)


def conditions(rule):
    """Number of comparisons in a rule (a chained comparison counts each operator)."""
    return sum(len(n.ops) for n in ast.walk(ast.parse(rule, mode="eval")) if isinstance(n, ast.Compare))


def cost(rule, seen, T):
    v = G.compile_rule(rule)
    return ERR * sum(v[i] != T[i] for i in seen) + COND * conditions(rule)


def best(pool, seen, T, by_count=False):
    """D's choice: lowest description length on all seen cases (D-count: most cases right). Ties by length, then
    text, so the choice depends only on the pool and the set of cases."""
    if not pool:
        return None
    if by_count:
        return min(pool, key=lambda r: (-G.score(r, seen, T), len(r), r))
    return min(pool, key=lambda r: (cost(r, seen, T), len(r), r))


def simple_rules():
    """Every one- and two-condition rule over the four attributes (two conditions on different attributes)."""
    conds = {"size": [f"size >= {k}" for k in range(2, 10)] + [f"size <= {k}" for k in range(1, 9)]}
    for a, vals in G.ATTRS.items():
        if a != "size":
            conds[a] = ([f"{a} == '{v}'" for v in vals] + [f"{a} != '{v}'" for v in vals] +
                        [f"{a} in ('{x}', '{y}')" for x, y in itertools.combinations(vals, 2)])
    single = [c for cs in conds.values() for c in cs]
    pairs = [f"{x} {op} {y}" for a, b in itertools.combinations(conds, 2) for x in conds[a] for y in conds[b]
             for op in ("and", "or")]
    return single + pairs


def lines(items, T):
    return "".join(G.case_line(i, T) for i in sorted(items, key=G.item_text))


def prompt(seen, T, current):
    p = G.INTRO + "All items seen so far and the inspector's decisions:\n" + lines(seen, T)
    if current is not None:
        v = G.compile_rule(current)
        wrong = [i for i in seen if v[i] != T[i]]
        p += f"\nYour current best rule is: {current}\n"
        p += ("It gets these items wrong:\n" + lines(wrong, T)) if wrong else "It fits every item above.\n"
    return p + ("\nWrite the simplest rule that fits every item above, as one Python boolean expression over the "
                "variables color, shape, size and material, for example: shape == 'star' or size < 3\n"
                "Answer with the expression only.")


def run_stream(llm, task, order, seed):
    """One stream: online answers, rounds of guessing, final choices and diagnostics."""
    T, stream, test = task["T"], task["orders"][order], task["test"]
    pool, rounds, on, shown = [], {}, [], []
    for t, i in enumerate(stream, start=1):
        seen = stream[:t - 1]
        on.append(G.decide(best(pool, seen, T), i, seen, T) == T[i])
        if t % G.EVERY == 0:
            seen = stream[:t]
            current = best(pool, seen, T)
            replies = llm.propose_batch([prompt(seen, T, current)], G.PER_ROUND, [seed + t])[0]
            got = [{"reply": r[:300], "rule": G.extract_rule(r)} for r in replies]
            rounds[str(t)] = {"current": current, "proposals": got}
            for g in got:
                if g["rule"]:
                    v = G.compile_rule(g["rule"])
                    shown.append((all(v[j] == T[j] for j in seen), conditions(g["rule"])))
                    if g["rule"] not in pool:
                        pool.append(g["rule"])
    out = {"rounds": rounds, "pool_size": len(pool),
           "pool_has_good_rule": any(G.agreement(G.compile_rule(r), T) >= G.GOOD for r in pool),
           "guesses": len(shown), "contradict_shown": sum(not f for f, _ in shown), "conditions": [c for _, c in shown]}
    for name, by_count in (("D", False), ("Dcount", True)):
        rule = best(pool, stream, T, by_count)
        held = [G.decide(rule, j, stream, T) == T[j] for j in test]
        out[name] = {"rule": rule, "test_acc": 100 * sum(held) / len(held),
                     "rule_good": rule is not None and G.agreement(G.compile_rule(rule), T) >= G.GOOD}
    out["D"]["online_acc"] = 100 * sum(on) / len(on)
    out["D"]["late_online_acc"] = 100 * sum(on[-20:]) / 20
    return out


class Stub:
    """No model: reads the cases from the prompt and answers with the shortest of a few candidate rules that fits
    them, a random candidate, or junk. Only for checking the bookkeeping."""
    LINE = re.compile(r"color=(\w+), shape=(\w+), size=(\d), material=(\w+): (yes|no)")
    CANDS = [t for t, _ in G.RULES] + [d for _, d in G.RULES] + ["size > 4", "color == 'red'", "shape != 'star'"]

    def propose_batch(self, prompts, n, seeds):
        out = []
        for p, s in zip(prompts, seeds):
            rng = random.Random(s)
            cases = [(G.DOMAIN.index({"color": m[0], "shape": m[1], "size": int(m[2]), "material": m[3]}), m[4] == "yes")
                     for m in self.LINE.findall(p.split("Your current best rule")[0])]
            fit = sorted((c for c in self.CANDS if all(G.compile_rule(c)[i] == y for i, y in cases)), key=len)
            opts = ([fit[0]] if fit else []) + [rng.choice(self.CANDS), "It is about the colour."]
            out.append([rng.choice(opts) for _ in range(n)])
        return out

    def check(self, task):
        return {"stub": True}


def check(llm, task):
    """One guessing call: does the model write rules the script can read?"""
    T, s = task["T"], task["orders"]["random1"]
    replies = llm.propose_batch([prompt(s[:10], T, None)], 2, [1])[0]
    parsed = [G.extract_rule(r) for r in replies]
    return {"model": getattr(llm, "model", "stub"), "replies": [r[:200] for r in replies], "parsed_rules": parsed,
            "writes_rules": any(parsed)}


def summarise(res, tasks):
    S = {}
    for a in ("D", "Dcount"):
        rnd = [G.mean([r[a][o]["test_acc"] for o in G.RANDOM]) for r in res]
        first = [x - r[a]["decoy_first"]["test_acc"] for x, r in zip(rnd, res)]
        last = [x - r[a]["decoy_last"]["test_acc"] for x, r in zip(rnd, res)]
        S[a] = {"random_acc": G.mean(rnd), "random_se": G.se(rnd), "learning": G.mean(rnd) - 50,
                "decoy_first_acc": G.mean([r[a]["decoy_first"]["test_acc"] for r in res]),
                "decoy_last_acc": G.mean([r[a]["decoy_last"]["test_acc"] for r in res]),
                "gap_first": G.mean(first), "gap_last": G.mean(last), "order_gap": max(G.mean(first), G.mean(last)),
                "final_rule_good_rate": G.mean([r[a][o]["rule_good"] for r in res for o in G.ORDERS])}
    S["D"]["late_online_acc"] = G.mean([G.mean([r["D"][o]["late_online_acc"] for o in G.RANDOM]) for r in res])
    streams = [r["streams"][o] for r in res for o in G.ORDERS]
    S["pool_good_rate"] = G.mean([s["pool_has_good_rule"] for s in streams])
    S["guesses"] = sum(s["guesses"] for s in streams)
    S["contradict_rate"] = sum(s["contradict_shown"] for s in streams) / max(1, S["guesses"])
    conds = [c for s in streams for c in s["conditions"]]
    S["mean_conditions"] = G.mean(conds) if conds else float("nan")
    S["brute_force_acc"] = G.mean([r["brute_force"]["test_acc"] for r in res])
    return S


def verdict(S, test):
    D = S["D"]
    if test == 1:
        if D["random_acc"] >= 95 and S["pool_good_rate"] >= 0.80:
            v = (f"PASS: D scores {D['random_acc']:.1f} >= 95 and a good rule is in the pool in "
                 f"{100 * S['pool_good_rate']:.0f}% >= 80% of streams. Checking against all evidence and preferring "
                 "simple rules fixes the induction failure on these tasks.")
        elif D["random_acc"] < 90:
            v = (f"STOP: D scores {D['random_acc']:.1f} < 90. The loop does not fix induction; next is a diagnosis of "
                 "the guesses, not a variant.")
        else:
            v = (f"IN BETWEEN: D scores {D['random_acc']:.1f} (pass needs >= 95 with good rules in >= 80% of streams, "
                 f"here {100 * S['pool_good_rate']:.0f}%). Report, diagnose, decide with the owner.")
    else:
        if D["random_acc"] >= A_27B + 5:
            v = (f"PASS: the phone-sized model with the habits scores {D['random_acc']:.1f} >= {A_27B + 5:.1f}, beating "
                 f"the 27B model without them ({A_27B}).")
        elif D["random_acc"] < A_27B:
            v = (f"STOP: the phone-sized model with the habits scores {D['random_acc']:.1f} < {A_27B}, below the 27B "
                 "model without them.")
        else:
            v = (f"IN BETWEEN: {D['random_acc']:.1f}, between {A_27B} and {A_27B + 5:.1f}. Report, diagnose, decide "
                 "with the owner.")
    return [v]


def write(out, R, test, done, total):
    L = [f"# Agent D, the scientist: test {test}", "",
         f"Model {R['model']}, {done} of {total} tasks done. Pre-registration: reports/Learning on the job from first "
         "principles 2026.md.", ""]
    res = [r for r in R["tasks"] if r is not None]
    if res:
        S = summarise(res, total)
        R["summary"] = S
        L += ["Test-set accuracy (%) after the 40-case stream, averaged over tasks. Order gap = the larger of "
              "(random - decoy first) and (random - decoy last).", "",
              "| Agent | Random orders | Decoy first | Decoy last | Learning | Order gap | Final rule good |",
              "|---|---|---|---|---|---|---|"]
        for a, name in (("D", "D, scientist"), ("Dcount", "D-count (same guesses, judged by count only)")):
            s = S[a]
            L.append(f"| {name} | {s['random_acc']:.1f} +/- {s['random_se']:.1f} | {s['decoy_first_acc']:.1f} | "
                     f"{s['decoy_last_acc']:.1f} | {s['learning']:.1f} | {s['order_gap']:.1f} | "
                     f"{100 * s['final_rule_good_rate']:.0f}% |")
        L += ["", f"Comparison on Qwen3.5-27B (order-gap test, random orders): A {A_27B}, B {B_27B}. Brute force over "
              f"2,578 simple rules with the same judging: {S['brute_force_acc']:.1f}.",
              f"Streams whose guesses include a good rule: {100 * S['pool_good_rate']:.0f}%. Guesses that contradict the "
              f"cases they were shown: {100 * S['contradict_rate']:.0f}% of {S['guesses']:,} (order-gap test: 61%). "
              f"Conditions per guess: {S['mean_conditions']:.1f} (order-gap test: 4.0). D's late online accuracy "
              f"(last 20 cases, random orders): {S['D']['late_online_acc']:.1f}.", "",
              "| Task | Rule | D random / first / last | D-count random / first / last | Brute force | D's rule (random1) |",
              "|---|---|---|---|---|---|"]
        for r in res:
            cells = [" / ".join(f"{x:.0f}" for x in (G.mean([r[a][o]["test_acc"] for o in G.RANDOM]),
                                                     r[a]["decoy_first"]["test_acc"], r[a]["decoy_last"]["test_acc"]))
                     for a in ("D", "Dcount")]
            L.append(f"| {r['id']} | `{r['rule']}` | {cells[0]} | {cells[1]} | {r['brute_force']['test_acc']:.0f} | "
                     f"`{r['D']['random1']['rule']}` |")
        L += ["", "## Verdict" + ("" if done == total else " (provisional: not all tasks done)"), ""]
        L += [f"- {v}" for v in verdict(S, test)]
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
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--test", type=int, choices=[1, 2], default=None, help="default: 2 for a 4B model, else 1")
    p.add_argument("--check", action="store_true")
    p.add_argument("--stub", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    test = args.test or (2 if "4B" in args.model else 1)
    if args.stub:
        llm, tag = Stub(), "stub"
    elif args.base_url:
        llm, tag = G.Endpoint(args.base_url, args.model, 1), args.model.split("/")[-1]
        print(f"endpoint {args.base_url}, serving {llm.served}; using {llm.model}", flush=True)
    else:
        raise SystemExit("needs --base-url or --stub")
    out = args.out or os.path.join(G.ROOT, "results", f"scientist_{tag}")
    tasks = [G.make_task(r, d) for r in range(len(G.RULES)) for d in range(G.DRAWS)]
    R = {"model": tag, "test": test, "settings": {"err_nats": ERR, "cond_nats": COND, "every": G.EVERY,
         "per_round": G.PER_ROUND, "sampling": G.SAMPLING, "good": G.GOOD}, "tasks": [None] * len(tasks)}
    R["check"] = c = check(llm, tasks[0]) if not args.stub else {"stub": True}
    print(f"check: {json.dumps(c, ensure_ascii=True)}", flush=True)
    if not args.stub and not c["writes_rules"]:
        raise SystemExit("the model's replies contain no readable rule: stopping")
    if args.check:
        raise SystemExit("check done; no run")
    rules = simple_rules()
    t0, parts = time.time(), {}
    with ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(run_stream, llm, task, o, 1_000_000 * k + 1000 * oi): (k, o)
                for k, task in enumerate(tasks) for oi, o in enumerate(G.ORDERS)}
        for f in as_completed(futs):
            k, o = futs[f]
            parts.setdefault(k, {})[o] = f.result()
            if len(parts[k]) == len(G.ORDERS):
                task, T = tasks[k], tasks[k]["T"]
                bf = best(rules, task["orders"]["random1"], T)
                r = {"id": task["id"], "rule": task["rule"], "decoy": task["decoy"],
                     "brute_force": {"rule": bf, "test_acc": 100 * sum(G.compile_rule(bf)[j] == T[j]
                                                                       for j in task["test"]) / len(task["test"])},
                     "streams": {o: {x: y for x, y in v.items() if x not in ("D", "Dcount")} for o, v in parts[k].items()}}
                for a in ("D", "Dcount"):
                    r[a] = {o: v[a] for o, v in parts[k].items()}
                R["tasks"][k] = r
                done = sum(x is not None for x in R["tasks"])
                write(out, R, test, done, len(tasks))
                print(f"task {done}/{len(tasks)} done ({task['id']}: D random1 {r['D']['random1']['test_acc']:.0f}, "
                      f"rule {r['D']['random1']['rule']}), {(time.time() - t0) / 60:.1f} min", flush=True)
    print("\n".join(write(out, R, test, len(tasks), len(tasks))), flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
