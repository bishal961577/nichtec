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


ORIG_KEYS = ("triples", "triples_labeled", "new_triples", "new_triples_labeled", "edit_triples")
REQUIRED = ("case_id", "requested_rewrite", "questions", "answer", "new_answer", "single_hops", "new_single_hops", "orig")


def _maybe_json(v):
    """Nested fields stored as JSON text become Python objects."""
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def normalise(row):
    """Map one Remastered row to the MQuAKE json layout the tests read: 'orig' nested (it may be stored as flat
    columns such as 'triples' or 'orig.triples'), JSON-text fields parsed, missing alias lists empty."""
    row = {k: _maybe_json(v) for k, v in row.items()}
    orig = _maybe_json(row.get("orig")) if isinstance(row.get("orig"), (dict, str)) else {}
    for k in list(row):
        for t in ORIG_KEYS:
            if k in (t, f"orig.{t}", f"orig_{t}", f"orig/{t}"):
                orig[t] = _maybe_json(row.pop(k))
    if orig:
        row["orig"] = orig
    for k in ("answer_alias", "new_answer_alias"):
        if row.get(k) is None:
            row[k] = []
    for key in ("single_hops", "new_single_hops"):
        for h in row.get(key) or []:
            if h.get("answer_alias") is None:
                h["answer_alias"] = []
    return row


def load_remastered(split):
    """Cases of one Remastered split as a list of dicts in the MQuAKE json layout (plus 'split'). The raw columns and
    a sample row are written to data/mquake_remastered/<split>_schema.json first, so a layout mismatch is visible."""
    folder = os.path.join(M4.ROOT, "data", "mquake_remastered")
    cache = os.path.join(folder, split + ".json")
    schema_path = os.path.join(folder, split + "_schema.json")
    os.makedirs(folder, exist_ok=True)
    raw = None
    if os.path.exists(cache) and os.path.exists(schema_path):   # a cache without its schema file is from an older run
        try:
            raw = json.load(open(cache, encoding="utf-8"))
        except ValueError:
            raw = None
    if raw is None:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
        path = hf_hub_download(REPO, FILES[split], repo_type="dataset")
        table = pq.read_table(path)
        schema = {"columns": {n: str(t) for n, t in zip(table.schema.names, table.schema.types)},
                  "first_row": {k: str(v)[:600] for k, v in table.slice(0, 1).to_pylist()[0].items()}}
        json.dump(schema, open(schema_path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"Remastered {split} columns:", list(schema["columns"]), flush=True)
        raw = table.to_pylist()
        json.dump(raw, open(cache, "w", encoding="utf-8"), default=str)
    rows = [normalise(r) for r in raw]
    missing = [k for k in REQUIRED if k not in rows[0]]
    orig_missing = [k for k in ORIG_KEYS[:3] if k not in rows[0].get("orig", {})]
    if missing or orig_missing:
        raise KeyError(f"Remastered {split}: fields {missing + ['orig.' + k for k in orig_missing]} not found; "
                       f"columns are {sorted(raw[0])}. See {schema_path}")
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
