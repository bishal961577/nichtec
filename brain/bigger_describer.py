"""Step 4 of the two ends check (reports/Two ends check 2026.md): the bigger describer.

Qwen3.5-27B-FP8 describes all 1,500 dev messages with describer v3 (the same prompt, settings and thinking off as the
4B), memory-blind. The same resolver binds the descriptions; step 3's conditions run on them (own definitions, A every
definition confirmed, B asking policy), with the reporting-verb filter registered before the run. The 4B's numbers
under the same code are the reference.

    python brain/bigger_describer.py --reference                   # CPU only: the 4B's reference numbers
    ENDPOINT_TOKEN=dummy python brain/bigger_describer.py --base-url URL/v1   # describe with the 27B (cached), measure
(The endpoint's real password is a network secret of the cloud environment, which replaces the dummy in transit.)
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
import confirm_defs_diag as CDD  # noqa: E402

OUT = TE.OUT
DESC27 = os.path.join(OUT, "describe_all_27b.jsonl")
FRAME = re.compile(r"(^|[.:!?] )In (her|his|their) free time,")


def quirk_messages(W):
    """Messages touched by the practice world's pronoun quirk, from the truth alone (no model output): a person with
    an everyday fact is referred to by a pronoun that clashes with the usual gender of their first name (the labels in
    brain/confirm_defs_diag.py), or the message uses the frame "In her free time, <name> loves ..."."""
    out = set()
    for n in W.notes:
        st = {m[1]: m[2] for m in n["mentions"]}
        if FRAME.search(n["text"]) or any(
                k in MW.SLOTS and st.get(pid) in ("pronoun", "pronoun_prev") and
                CDD.cue(W.people[pid]) == "pronoun clashes with the name" for pid, k, v in n["facts"]):
            out.add(n["id"])
    return out


def writing_end(W, descs, tokens):
    """test 2's writing-end scores (corrected scorer) on the 300 sample messages."""
    recs = []
    for n in TE.sample_messages(W):
        d = descs.get(n["id"])
        recs.append({"parsed": d is not None, "score_v2": TE.score_description_v2(W, n, d),
                     "completion_tokens": tokens.get(n["id"], (0, 0))[0], "prompt_tokens": tokens.get(n["id"], (0, 0))[1]})
    return TE.summarize_writing(recs, "score_v2")[1]


def load_tokens(paths):
    t = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            t[r["id"]] = (r.get("completion_tokens", 0), r.get("prompt_tokens", 0))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model", default="Qwen/Qwen3.5-27B-FP8")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--reference", action="store_true", help="only the 4B's reference numbers (CPU)")
    a = ap.parse_args()
    W = TE.dev_world()
    logf = open(os.path.join(OUT, "bigger.log"), "a", encoding="utf-8")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()
    log(f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} step 4: the bigger describer" + (" (4B reference)" if a.reference else ""))
    ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
    days = (date.fromisoformat(W.notes[-1]["date"]) - date.fromisoformat(W.notes[0]["date"])).days + 1
    quirk = quirk_messages(W)
    log(f"messages touched by the pronoun quirk: {len(quirk & set(ids))} of {len(ids)}")
    emb = MI.Embedder()
    p4 = (os.path.join(OUT, "writing.jsonl"), DB.DESC)
    res = {"quirk_messages": len(quirk & set(ids))}
    res["4B"] = CD.run_conditions(W, CD.load_descs(p4), ids, emb, log, days, drop_reporting=True, exclude=quirk,
                                  tag="4B, filter: ")
    res["4B_writing"] = writing_end(W, CD.load_descs(p4), load_tokens(p4))
    if a.reference:
        with open(os.path.join(OUT, "bigger_reference_4b.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        return
    if a.base_url:
        llm = G.Endpoint(a.base_url, a.model, 1)
        log(f"model served: {llm.model}")
        DB.describe_all(llm, W, log, a.workers, out=DESC27, reuse_sample=False)
    d27 = CD.load_descs((DESC27,))
    missing = [n["id"] for n in W.notes if n["id"] not in d27]
    if missing:
        raise SystemExit(f"{len(missing)} messages have no 27B description yet: run with --base-url")
    t27 = load_tokens((DESC27,))
    res["27B"] = CD.run_conditions(W, d27, ids, emb, log, days, drop_reporting=True, exclude=quirk, tag="27B, filter: ")
    res["27B_no_filter"] = CD.run_conditions(W, d27, ids, emb, log, days, drop_reporting=False, exclude=quirk,
                                             tag="27B, no filter: ")
    res["27B_writing"] = writing_end(W, d27, t27)
    res["27B_unparsed"] = sum(1 for n in W.notes if d27.get(n["id"]) is None)
    res["27B_tokens"] = {"completion_mean": float(np.mean([v[0] for v in t27.values()])),
                         "prompt_mean": float(np.mean([v[1] for v in t27.values()]))}

    f2, f1 = (lambda v: f"{v:.2f}%"), (lambda v: f"{v:.1f}%")
    L = ["# Step 4: the bigger describer (Qwen3.5-27B-FP8) on the dev world", "",
         "Pre-registration: reports/Two ends check 2026.md, step 4. Describer v3, thinking off, memory-blind; the same "
         "resolver; the reporting-verb filter on (registered before the run).", "",
         f"Messages touched by the practice world's pronoun quirk (left out of the split measure): {res['quirk_messages']}.",
         "", "| Measure | 4B own | 4B A | 4B B | 27B own | 27B A | 27B B | Line |", "|---|---|---|---|---|---|---|---|"]
    rows = [("Wrong merges, everyday facts (primary)", "everyday_wrong_pct", f2, "<= 1.5% (stop > 3%)"),
            ("Same, without the pronoun quirk", "everyday_wrong_split_pct", f2, "reported (interpretation)"),
            ("Wrong merges, registered count", "registered_wrong_pct", f2, "reported"),
            ("\"X's flatmate\" filed right", "via_pct", f1, ">= 75%"),
            ("Worst-person forgetting", "forget_worst_pct", f1, ">= 90%"),
            ("Facts filed right", "facts_filed_pct", f1, ">= 85%"),
            ("Link precision", "link_precision_pct", f1, "reported"),
            ("Mentions left unresolved", "unresolved_pct", f1, "reported"),
            ("Owner questions", "questions", str, "B <= 2 a day"),
            ("Verdict (dev)", "verdict", str, "")]
    for name, key, fmt, line in rows:
        cells = [fmt(res[m][c][key]) if key in res[m][c] else "–" for m in ("4B", "27B") for c in ("none", "A", "B")]
        L.append(f"| {name} | " + " | ".join(cells) + f" | {line} |")
    L += ["", "Wrong everyday rows by how the person the fact is about was mentioned:", ""]
    for m in ("4B", "27B"):
        for c in ("none", "A", "B"):
            L.append(f"- {m} {c}: {json.dumps(res[m][c]['wrong_by_owner_style'])}")
    L += ["", "Without the filter, the 27B: " + "; ".join(
        f"{c} {res['27B_no_filter'][c]['everyday_wrong_pct']:.2f}%" for c in ("none", "A", "B")), "",
        "Writing end on the 300 sample (corrected scorer), 4B -> 27B:", ""]
    for k in ("attached_easy", "attached_hard", "rel_direction", "lures_tagged", "definitions", "fact_recall", "parsed"):
        L.append(f"- {k}: {res['4B_writing'][k]:.1f}% -> {res['27B_writing'][k]:.1f}%")
    L += ["", f"27B: {res['27B_unparsed']} messages without valid JSON; tokens per message "
              f"{res['27B_tokens']['completion_mean']:.0f} out, {res['27B_tokens']['prompt_mean']:.0f} in."]
    with open(os.path.join(OUT, "bigger_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(OUT, "bigger_result.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    log("\n".join(L))


if __name__ == "__main__":
    main()
