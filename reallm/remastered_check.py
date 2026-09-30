"""What changed between MQuAKE-CF-3k-v2 (used in tests 4 and 5) and MQuAKE-Remastered CF-3k (Zhong et al., ICLR 2025).

The Remastered audit reports that 33-76% of MQuAKE's questions and labels were corrupted (edit contamination, missing
information in questions, conflicting edits, duplicates). This script downloads the corrected data from Hugging Face
and counts, case by case, what differs from the file our runs used.

  python reallm/remastered_check.py            # needs internet and pyarrow (pip install pyarrow)
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtest4 as M4  # noqa: E402

REPO = "henryzhongsc/MQuAKE-Remastered"
FILES = {"CF3k": "data/CF3k-00000-of-00001.parquet", "CF9k": "data/CF9k-00000-of-00001.parquet",
         "CF6334": "data/CF6334-00000-of-00001.parquet", "T": "data/T-00000-of-00001.parquet"}


def load_remastered(split):
    """Cases of one Remastered split as a list of dicts (same fields as the MQuAKE json files, plus 'split')."""
    cache = os.path.join(M4.ROOT, "data", "mquake_remastered", split + ".json")
    if os.path.exists(cache):
        return json.load(open(cache, encoding="utf-8"))
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    path = hf_hub_download(REPO, FILES[split], repo_type="dataset")
    rows = pq.read_table(path).to_pylist()
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(rows, open(cache, "w", encoding="utf-8"))
    return rows


def rewrites(c):
    return [(r["subject"], r["prompt"], r["target_new"]["str"], r["target_true"]["str"]) for r in c["requested_rewrite"]]


def hops(c, key):
    return [(h["question"], h["cloze"], h["answer"]) for h in c[key]]


FIELDS = {
    "requested edits (subject, prompt, new, old)": rewrites,
    "multi-hop questions": lambda c: list(c["questions"]),
    "original answer": lambda c: c["answer"],
    "new answer": lambda c: c["new_answer"],
    "new answer aliases": lambda c: sorted(c.get("new_answer_alias") or []),
    "single hops (before edit)": lambda c: hops(c, "single_hops"),
    "single hops (after edit)": lambda c: hops(c, "new_single_hops"),
    "fact chain (triples)": lambda c: [tuple(t) for t in c["orig"]["triples"]],
    "edited chain (new triples)": lambda c: [tuple(t) for t in c["orig"]["new_triples"]],
}


def main():
    out = os.path.join(M4.ROOT, "results", "remastered_check")
    os.makedirs(out, exist_ok=True)
    v2 = {c["case_id"]: c for c in M4.fetch("MQuAKE-CF-3k-v2.json")}
    rm_list = load_remastered("CF3k")
    rm = {c["case_id"]: c for c in rm_list}
    common = sorted(set(v2) & set(rm))
    R = {"v2_cases": len(v2), "remastered_cases": len(rm), "same_case_ids": len(common),
         "only_in_v2": len(set(v2) - set(rm)), "only_in_remastered": len(set(rm) - set(v2)), "fields": {}, "examples": {}}
    changed_any = set()
    for name, f in FIELDS.items():
        diff = [i for i in common if f(v2[i]) != f(rm[i])]
        changed_any.update(diff)
        R["fields"][name] = len(diff)
        R["examples"][name] = [{"case_id": i, "v2": str(f(v2[i]))[:300], "remastered": str(f(rm[i]))[:300]} for i in diff[:3]]
    R["cases_with_any_change"] = len(changed_any)
    # Remastered's own edited/unedited split per batch size, if present
    splits = Counter()
    for c in rm_list:
        sp = c.get("split")
        if isinstance(sp, dict):
            for k, v in sp.items():
                if v is not None:
                    splits[f"{k}: {','.join(v) if isinstance(v, list) else v}"] += 1
    R["remastered_split_labels"] = dict(sorted(splits.items()))
    # edits our runs wrote that no Remastered case requests (distinct (subject, prompt) pairs)
    e_v2 = {(r["subject"], r["prompt"], r["target_new"]["str"]) for c in v2.values() for r in c["requested_rewrite"]}
    e_rm = {(r["subject"], r["prompt"], r["target_new"]["str"]) for c in rm.values() for r in c["requested_rewrite"]}
    R["distinct_edits_v2"], R["distinct_edits_remastered"] = len(e_v2), len(e_rm)
    R["edits_only_in_v2"], R["edits_only_in_remastered"] = len(e_v2 - e_rm), len(e_rm - e_v2)
    json.dump(R, open(os.path.join(out, "result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["# MQuAKE-CF-3k-v2 vs MQuAKE-Remastered CF-3k", "",
         f"Cases: v2 {R['v2_cases']:,}, Remastered {R['remastered_cases']:,}; same case ids {R['same_case_ids']:,}; "
         f"only in v2 {R['only_in_v2']:,}; only in Remastered {R['only_in_remastered']:,}.", "",
         f"Cases with any difference (of the shared ids): **{R['cases_with_any_change']:,}** "
         f"({100 * R['cases_with_any_change'] / max(1, len(common)):.1f}%).", "",
         f"Distinct edits: v2 {R['distinct_edits_v2']:,}, Remastered {R['distinct_edits_remastered']:,}; "
         f"only in v2 {R['edits_only_in_v2']:,}, only in Remastered {R['edits_only_in_remastered']:,}.", "",
         "| Field | Cases that differ |", "|---|---|"]
    L += [f"| {k} | {v:,} |" for k, v in R["fields"].items()]
    L += ["", "Remastered split labels (edited/unedited by batch size): " + json.dumps(R["remastered_split_labels"], ensure_ascii=False), ""]
    for k, ex in R["examples"].items():
        if ex:
            L += [f"## {k}: examples", ""]
            for x in ex:
                L += [f"- case {x['case_id']}", f"  - v2: `{x['v2']}`", f"  - Remastered: `{x['remastered']}`"]
            L.append("")
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[:14]), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
