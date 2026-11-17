"""Post-hoc, after step 4's results (reports/Two ends check 2026.md): the same measurement with the antecedent rule
fixed (describe_bind.ANTE_V2). Written after seeing the results, on the world the resolver was tuned on; it changes no
registered verdict. It tells the next pre-registered test what to expect.

    python brain/bigger_posthoc.py
"""
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_index as MI  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402
import confirm_defs as CD  # noqa: E402
import bigger_describer as BD  # noqa: E402


def main():
    W = TE.dev_world()
    ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
    days = (date.fromisoformat(W.notes[-1]["date"]) - date.fromisoformat(W.notes[0]["date"])).days + 1
    quirk = BD.quirk_messages(W)
    emb = MI.Embedder()
    log = print
    files = {"4B": (os.path.join(TE.OUT, "writing.jsonl"), DB.DESC), "27B": (BD.DESC27,)}
    res = {}
    for v2 in (False, True):
        DB.ANTE_V2 = v2
        for m, paths in files.items():
            res[f"{m}{' fixed' if v2 else ''}"] = CD.run_conditions(W, CD.load_descs(paths), ids, emb, log, days,
                                                                    drop_reporting=True, exclude=quirk,
                                                                    tag=f"{m}{' fixed' if v2 else ''}: ")
    DB.ANTE_V2 = False
    keys = ["4B", "4B fixed", "27B", "27B fixed"]
    L = ["# Step 4, post-hoc: the antecedent rule fixed (dev; not a judged run)", "",
         "Condition A (every definition confirmed), reporting-verb filter on. 'Fixed': the describer's \"previous\" may "
         "name someone in the same message, the teller included, and names match as whole words.", "",
         "| Measure | " + " | ".join(keys) + " | Line |", "|---|" + "---|" * len(keys) + "---|"]
    for name, key, fmt, line in [
            ("Wrong merges, everyday facts", "everyday_wrong_pct", lambda v: f"{v:.2f}%", "<= 1.5% (stop > 3%)"),
            ("Same, without the pronoun quirk", "everyday_wrong_split_pct", lambda v: f"{v:.2f}%", "-"),
            ("Wrong rows", "everyday_wrong_rows", str, "-"),
            ("\"X's flatmate\" filed right", "via_pct", lambda v: f"{v:.1f}%", ">= 75%"),
            ("Worst-person forgetting", "forget_worst_pct", lambda v: f"{v:.1f}%", ">= 90%"),
            ("Facts filed right", "facts_filed_pct", lambda v: f"{v:.1f}%", ">= 85%")]:
        L.append(f"| {name} | " + " | ".join(fmt(res[k]["A"][key]) for k in keys) + f" | {line} |")
    L += ["", "Condition B (asking policy), wrong merges on everyday facts: " +
          "; ".join(f"{k} {res[k]['B']['everyday_wrong_pct']:.2f}% (without the quirk "
                    f"{res[k]['B']['everyday_wrong_split_pct']:.2f}%, {res[k]['B']['questions']} questions)" for k in keys),
          "", "Wrong rows by how the person was mentioned, A: " +
          "; ".join(f"{k} {json.dumps(res[k]['A']['wrong_by_owner_style'])}" for k in keys)]
    with open(os.path.join(TE.OUT, "bigger_posthoc.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(TE.OUT, "bigger_posthoc.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
