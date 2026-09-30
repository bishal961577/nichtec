"""Are MQuAKE's labels right when every edit is applied at once, as tests 4 and 5 do?

The MQuAKE-Remastered audit (Zhong et al., ICLR 2025) reports that 33-76% of MQuAKE's questions and labels are
corrupted (edit contamination, conflicting edits, missing information in questions, duplicates). This script checks
the two label errors that matter for an all-edits memory, contamination and conflicts, in the original MQuAKE-CF-3k,
the v2 file tests 4 and 5 used, MQuAKE-CF, MQuAKE-T and (if Hugging Face is reachable) MQuAKE-Remastered, and
compares v2 with Remastered CF-3k by fact chain.

  python reallm/remastered_check.py            # Remastered part needs Hugging Face access and pyarrow
"""
import json
import os
import sys
from collections import Counter, defaultdict

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
    # the released parquet stores each edit's targets flat (target_new_str, target_true_str), unlike its README
    for r in row.get("requested_rewrite") or []:
        for side in ("target_new", "target_true"):
            if not isinstance(r.get(side), dict) and f"{side}_str" in r:
                r[side] = {"str": r[f"{side}_str"], "id": r.get(f"{side}_id")}
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
    rw = (rows[0].get("requested_rewrite") or [{}])[0]
    missing += [f"requested_rewrite.{k}" for k in ("subject", "prompt", "relation_id", "target_new", "target_true")
                if k not in rw]
    if missing or orig_missing:
        raise KeyError(f"Remastered {split}: fields {missing + ['orig.' + k for k in orig_missing]} not found; "
                       f"columns are {sorted(raw[0])}; edit fields are {sorted(rw)}. See {schema_path}")
    return rows


def chain(c):
    """A case's identity: its fact chain as Wikidata ids. Case numbers differ between the MQuAKE files."""
    return tuple(tuple(t) for t in c["orig"]["triples"])


def edit_ids(c):
    """The case's edits as Wikidata (subject, relation, new object) ids."""
    o = c["orig"]
    if o.get("edit_triples"):
        return [tuple(t) for t in o["edit_triples"]]
    out = []                                    # otherwise find each requested edit in the edited chain
    for r in c["requested_rewrite"]:
        for t, tl in zip(o["new_triples"], o["new_triples_labeled"]):
            if t[1] == r["relation_id"] and tl[0] == r["subject"] and tl[2] == r["target_new"]["str"]:
                out.append(tuple(t))
                break
    return out


def audit(cases):
    """Apply every case's edits together, as tests 4 and 5 do, and check each case's labelled edited chain.
    contaminated: a step the case does not edit is edited by another case, so the labelled answer is wrong;
    conflicting: two cases edit the same step to different objects."""
    G = defaultdict(set)
    for c in cases:
        for s, r, o in edit_ids(c):
            G[(s, r)].add(o)
    status = Counter()
    for c in cases:
        st = "clean"
        for s, r, o in c["orig"]["new_triples"]:
            if (s, r) in G:
                if o not in G[(s, r)]:
                    st = "contaminated"
                    break
                if len(G[(s, r)]) > 1:
                    st = "conflicting"
        status[st] += 1
    dup = Counter((chain(c), tuple(sorted(edit_ids(c)))) for c in cases)
    return {"cases": len(cases), "distinct_edits": sum(len(v) for v in G.values()), "clean": status["clean"],
            "conflicting": status["conflicting"], "contaminated": status["contaminated"],
            "duplicate_cases": sum(v - 1 for v in dup.values() if v > 1)}


def main():
    out = os.path.join(M4.ROOT, "results", "mquake_audit")
    os.makedirs(out, exist_ok=True)
    sets = {"MQuAKE-CF-3k (original)": M4.fetch("MQuAKE-CF-3k.json"),
            "MQuAKE-CF-3k-v2 (tests 4 and 5)": M4.fetch("MQuAKE-CF-3k-v2.json"),
            "MQuAKE-CF (9,218 cases)": M4.fetch("MQuAKE-CF.json"),
            "MQuAKE-T": M4.fetch("MQuAKE-T.json")}
    R = {}
    for split in ("CF3k", "T"):
        try:
            sets[f"MQuAKE-Remastered {split}"] = load_remastered(split)
        except Exception as e:                  # e.g. Hugging Face unreachable: the MQuAKE files are still audited
            R[f"remastered_{split}_error"] = f"{type(e).__name__}: {e}"
            print(f"Remastered {split} not loaded: {R[f'remastered_{split}_error']}", flush=True)
    R["audit"] = {k: audit(v) for k, v in sets.items()}
    if "MQuAKE-Remastered CF3k" in sets:
        v2 = {chain(c): c for c in sets["MQuAKE-CF-3k-v2 (tests 4 and 5)"]}
        rm = {chain(c): c for c in sets["MQuAKE-Remastered CF3k"]}
        shared = sorted(set(v2) & set(rm))
        R["v2_vs_remastered_cf3k"] = {
            "fact_chains_v2": len(v2), "fact_chains_remastered": len(rm), "shared_fact_chains": len(shared),
            "shared_with_same_edits": sum(set(edit_ids(v2[k])) == set(edit_ids(rm[k])) for k in shared),
            "shared_with_same_new_answer": sum(M4.norm(v2[k]["new_answer"]) == M4.norm(rm[k]["new_answer"]) for k in shared),
            "shared_with_same_questions": sum(v2[k]["questions"] == rm[k]["questions"] for k in shared)}
    json.dump(R, open(os.path.join(out, "result.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    L = ["# MQuAKE label audit: are the labels right when every edit is applied at once?", "",
         "Tests 4 and 5 write all of a file's edits into one memory and score every case against its new answer. A",
         "case's label is then wrong if another case edits a step of its chain that it does not edit itself",
         "(contaminated), or two cases edit the same step to different answers (conflicting). Cases are matched by",
         "their Wikidata fact chain, since case numbers differ between the files.", "",
         "| Data | Cases | Distinct edits | Clean | Conflicting | Contaminated | Duplicate cases |",
         "|---|---|---|---|---|---|---|"]
    for k, a in R["audit"].items():
        L.append(f"| {k} | {a['cases']:,} | {a['distinct_edits']:,} | {a['clean']:,} | {a['conflicting']:,} | "
                 f"{a['contaminated']:,} ({100 * a['contaminated'] / max(1, a['cases']):.1f}%) | {a['duplicate_cases']:,} |")
    for k in ("remastered_CF3k_error", "remastered_T_error"):
        if k in R:
            L += ["", f"{k}: {R[k]}"]
    if "v2_vs_remastered_cf3k" in R:
        x = R["v2_vs_remastered_cf3k"]
        L += ["", f"v2 against Remastered CF-3k: {x['shared_fact_chains']:,} shared fact chains (of {x['fact_chains_v2']:,} "
              f"and {x['fact_chains_remastered']:,}); among them same edits {x['shared_with_same_edits']:,}, same new "
              f"answer {x['shared_with_same_new_answer']:,}, same questions {x['shared_with_same_questions']:,}."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
