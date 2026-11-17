"""Step 5 of the two ends check (reports/Two ends check 2026.md): the judged run of step 4's post-hoc configuration on
a fresh world the resolver was never tuned on.

World: messy_world.World("dev", seed=20261117), first 1,500 messages, its deletions among them. The same generator
and sentence templates as the dev world (seed 101), so it tests new people, names, facts and sequences, not new
phrasing; dev slots and relations only; no sealed material. Describer: Qwen3.5-27B-FP8, describer v3, thinking off,
memory-blind. Resolver: frozen at step 4's post-hoc version (antecedent rule fixed, reporting-verb filter on).

    python brain/fresh_check.py --selfcheck                          # CPU: the world, its quirk set, truth self-check
    ENDPOINT_TOKEN=... python brain/fresh_check.py --base-url URL/v1  # describe (cached) and measure
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402
import order_gap as G  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402
import confirm_defs as CD  # noqa: E402
import describe_bind_selfcheck as SC  # noqa: E402

SEED = 20261117
OUT = TE.OUT
DESC_FRESH = os.path.join(OUT, "describe_fresh_27b.jsonl")
# Usual gender of every first name the generator can pick: the analyst's own labels, fixed before step 5's model calls.
# Unisex or unsure names are listed as such and count as "no name cue". (A superset of the dev labels in
# brain/confirm_defs_diag.py, which stay as they were for the dev numbers.)
FEM = set("""Ada Agnes Aino Aisha Alma Amara Ana Anabel Annabel Annika Astrid Ayla Bea Beatriz Brigid Carmen Chloe Cleo
Dalia Delia Dina Edda Elena Elif Elina Emilia Esme Ewa Fenna Fiona Flora Gaia Greta Gwen Halima Hana Hanna Hedda Henrike
Hilde Ilse Inaya Ines Iris Isolde Ivana Ivanka Jonna Kasia Ksenia Leila Lena Lenka Lenna Leona Linnea Liv Lotta Lucia Lumi
Maren Mariam Marta Maya Mila Mira Mirela Mirella Nadia Nadine Nadja Nell Neve Nora Odile Orla Ottavia Ottilie Paloma Paula
Paulina Petra Pia Priya Rania Rhea Rosa Rosalind Ruth Sanna Saoirse Sara Sarai Selin Shirin Signe Sofia Suvi Svea Svenja
Talia Tamsin Tilde Ulla Uma Valeria Veda Vera Vesna Vida Wanjiru Wiebke Willa Xenia Yara Zara Zeynep""".split())
MASC = set("""Abel Adam Anton Arjun Bashir Bilal Bram Bruno Caio Casimir Ciaran Cosmin Cyril Dario Darius Dmitri Eero Emil
Emre Ezra Fadil Faisal Farid Felix Florin Gideon Goran Gustav Hamid Hiro Hugo Idris Ilias Ivan Jasper Joao Joaquin Joel
Jonas Jovan Jude Kaito Karim Kenji Kenzo Kofi Kwame Lars Leo Loic Ludo Mads Malik Marco Marcos Mateo Mats Matteo Milan Milo
Niko Noah Nuno Omar Omari Otto Paulo Pavel Priit Rasmus Raul Ravi Renzo Rhys Rui Sanjay Soren Sven Tariq Taro Teo Teodor
Thabo Timo Tobias Tomas Tomasz Tuan Ugo Umar Vito Wes Wim Xavi Yann Yoni Yusuf Zeno""".split())
UNISEX = set("Ade Akira Bodhi Femi Imani Kai Koa Lior Mosi Nilo Noor Ola Quin Sefa Simone Wren Yuki Zola".split())
FRAME = re.compile(r"(^|[.:!?] )In (her|his|their) free time,")


def name_pool():
    flat = lambda x: x.split() if isinstance(x, str) else [w for y in x for w in (y.split() if isinstance(y, str) else y)]
    return set(flat(MW.SAME_FIRST)) | {a for p in MW.NEAR_FIRST for a in p} | set(flat(MW.OTHER_FIRST))


def cue(p):
    f = p["first"]
    if p["pron"] == "they" or f in UNISEX:
        return "no name cue"
    return "matches" if (p["pron"] == "she") == (f in FEM) else "clashes"


def quirk_messages(W):
    """As step 4's (brain/bigger_describer.py), with the full label set."""
    out = set()
    for n in W.notes:
        st = {m[1]: m[2] for m in n["mentions"]}
        if FRAME.search(n["text"]) or any(k in MW.SLOTS and st.get(pid) in ("pronoun", "pronoun_prev") and
                                          cue(W.people[pid]) == "clashes" for pid, k, v in n["facts"]):
            out.add(n["id"])
    return out


def fresh_world():
    W = MW.World("dev", seed=SEED)
    keep = {n["id"] for n in W.notes[:TE.DEV_LIMIT]}
    W.notes = W.notes[:TE.DEV_LIMIT]
    W.deleted = [d for d in W.deleted if d[2] in keep]
    W.deleted_ids = {d[2] for d in W.deleted}
    W.questions = [q for q in W.questions if all(n in keep for n in q["notes"]) and q["notes"]]
    return W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    pool = name_pool()
    assert FEM | MASC | UNISEX >= pool and not (FEM & MASC) and not (FEM & UNISEX) and not (MASC & UNISEX), \
        sorted(pool - FEM - MASC - UNISEX)
    W = fresh_world()
    logf = open(os.path.join(OUT, "fresh.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
    days = (date.fromisoformat(W.notes[-1]["date"]) - date.fromisoformat(W.notes[0]["date"])).days + 1
    quirk = quirk_messages(W)
    defs = sum(1 for i in ids if CD.truth_defs(W, W.notes[i]))
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} step 5: fresh world (seed {SEED}), {len(ids)} messages after "
        f"{len(W.deleted_ids)} deletions, {days} days, {defs} carrying a definition, {len(quirk & set(ids))} touched by "
        f"the pronoun quirk, {len(W.people)} people")
    if a.selfcheck:
        DB.ANTE_V2 = False
        wq, fg, V = SC.check(W)
        log(f"self-check (truth-built descriptions, registered rule): wrong merges {wq['wrong_merge_pct']:.2f}%, via "
            f"{wq['filed_right_by_style'].get('via')}, forgetting worst {fg['forget_worst_pct']:.1f}, unresolved "
            f"{V.unresolved} of {V.mentions}")
        DB.ANTE_V2 = True
        wq, fg, V = SC.check(W)
        log(f"self-check (truth-built descriptions, fixed rule): wrong merges {wq['wrong_merge_pct']:.2f}%, via "
            f"{wq['filed_right_by_style'].get('via')}, forgetting worst {fg['forget_worst_pct']:.1f}, unresolved "
            f"{V.unresolved} of {V.mentions}")
        return
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        DB.describe_all(llm, W, log, a.workers, out=DESC_FRESH, reuse_sample=False)
    d = CD.load_descs((DESC_FRESH,))
    missing = [n["id"] for n in W.notes if n["id"] not in d]
    if missing:
        raise SystemExit(f"{len(missing)} messages have no description yet: run with --base-url")
    DB.ANTE_V2 = True
    res = CD.run_conditions(W, d, ids, MI.Embedder(), log, days, drop_reporting=True, exclude=quirk, tag="27B fresh: ")
    DB.ANTE_V2 = False
    res["unparsed"] = sum(1 for n in W.notes if d.get(n["id"]) is None)
    toks = [json.loads(line) for line in open(DESC_FRESH, encoding="utf-8")]
    res["tokens"] = {"out": float(np.mean([t["completion_tokens"] for t in toks])),
                     "in": float(np.mean([t["prompt_tokens"] for t in toks]))}
    A, B = res["A"], res["B"]
    f2, f1 = (lambda v: f"{v:.2f}%"), (lambda v: f"{v:.1f}%")
    L = [f"# Step 5: the judged run on a fresh world (seed {SEED})", "",
         "Pre-registration: reports/Two ends check 2026.md, step 5. Qwen3.5-27B-FP8 describes (v3, thinking off, "
         "memory-blind); resolver frozen at step 4's post-hoc version.", "",
         f"{len(ids)} messages after deletion, {days} days, {len(quirk & set(ids))} touched by the pronoun quirk.", "",
         "| Measure | own definitions | A: every definition confirmed | B: asking policy | Line | Dev, post-hoc (A) |",
         "|---|---|---|---|---|---|"]
    dev = {"everyday_wrong_pct": "1.52%", "everyday_wrong_split_pct": "0.61%", "via_pct": "100%",
           "forget_worst_pct": "100%", "facts_filed_pct": "90.5%"}
    for name, key, fmt, line in [
            ("**Wrong merges, everyday facts (primary)**", "everyday_wrong_pct", f2, "<= 1.5% (stop > 3%)"),
            ("Same, without the pronoun quirk", "everyday_wrong_split_pct", f2, "interpretation"),
            ("Wrong rows", "everyday_wrong_rows", str, "-"),
            ("Wrong merges, registered count", "registered_wrong_pct", f2, "reported"),
            ("\"X's flatmate\" filed right", "via_pct", f1, ">= 75%"),
            ("Worst-person forgetting", "forget_worst_pct", f1, ">= 90%"),
            ("Facts filed right", "facts_filed_pct", f1, ">= 85%"),
            ("Link precision", "link_precision_pct", f1, "reported"),
            ("Mentions left unresolved", "unresolved_pct", f1, "reported"),
            ("Owner questions", "questions", lambda v: f"{v} ({v / days:.2f} a day)", "B <= 2 a day"),
            ("Verdict", "verdict", str, "")]:
        L.append(f"| {name} | " + " | ".join(fmt(res[c][key]) if key in res[c] else "–" for c in ("none", "A", "B"))
                 + f" | {line} | {dev.get(key, '')} |")
    L += ["", "Wrong everyday rows by how the person the fact is about was mentioned: " +
          "; ".join(f"{c} {json.dumps(res[c]['wrong_by_owner_style'])}" for c in ("none", "A", "B")),
          "", f"Valid JSON: {1500 - res['unparsed']} of 1500; tokens per message {res['tokens']['out']:.0f} out, "
              f"{res['tokens']['in']:.0f} in.", "",
          f"**Verdict (as registered, condition A): {A['verdict']}**"]
    with open(os.path.join(OUT, "fresh_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "fresh_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("\n".join(L))


if __name__ == "__main__":
    main()
