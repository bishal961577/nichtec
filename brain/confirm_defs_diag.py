"""Post-hoc diagnosis of step 3 (reports/Two ends check 2026.md, step 3): where condition A's wrong everyday rows come
from, and how pronoun filing depends on whether the pronoun matches the usual gender of the person's first name.
Written after seeing step 3's results; it changes no measure and no verdict.

    python brain/confirm_defs_diag.py
    python brain/confirm_defs_diag.py --descs results/two_ends/describe_all_27b.jsonl --filter    # step 4's 27B
"""
import argparse
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402
import two_ends as TE  # noqa: E402
import describe_bind as DB  # noqa: E402
import confirm_defs as CD  # noqa: E402

# The usual gender of first names that carry pronoun facts on dev: the analyst's own labels, no outside source.
# Unisex or unsure names (Bodhi, Mosi, Nilo, Simone, Yuki, ...) are left out and count as "no name cue".
FEM = set("Aisha Amara Anabel Annika Astrid Beatriz Chloe Cleo Ewa Gaia Hana Hedda Ivana Leila Leona Lotta Lucia Nell "
          "Ottilie Petra Pia Rania Rhea Rosa Rosalind Sarai Shirin Signe Suvi Svea Uma Valeria Vesna".split())
MASC = set("Anton Bram Ciaran Idris Ivan Kenji Mads Malik Mats Milo Pavel Soren Tariq Teo Tobias Ugo".split())
REPORT = re.compile(r"^(told|says|said|heard|reported|mentioned|according)")


def cue(p):
    f = p["first"]
    if p["pron"] == "they" or (f not in FEM and f not in MASC):
        return "no name cue (they, or a name not labelled)"
    return "pronoun matches the name" if (p["pron"] == "she") == (f in FEM) else "pronoun clashes with the name"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--descs", nargs="*", help="description files (default: the 4B's)")
    ap.add_argument("--filter", action="store_true", help="step 4's reporting-verb filter")
    a = ap.parse_args()
    W = TE.dev_world()
    descs = CD.load_descs(a.descs)
    fl = a.filter
    ids = [n["id"] for n in W.notes if n["id"] not in W.deleted_ids]
    by = {n["id"]: n for n in W.notes}
    confA = {i: CD.truth_defs(W, by[i]) for i in ids if CD.truth_defs(W, by[i])}
    conds = {"step 2 as is": DB.to_parse(W, descs, ids, drop_reporting=fl),
             "A": DB.to_parse(W, descs, ids, keep_defs=False, confirmed=confA, drop_reporting=fl)}
    for label, P in conds.items():
        V = DB.View(W, P, DB.resolve(P))
        wq = MI.write_quality(W, V.ix)
        who = wq.pop("who")
        rows = defaultdict(list)
        for r in V.ix.rows:
            rows[r["note"]].append(r)
        tab = defaultdict(lambda: [0, 0])
        for i in ids:
            n = by[i]
            st = {m[1]: m[2] for m in n["mentions"]}
            for pid, s, v in n["facts"]:
                if s not in MW.SLOTS or v is None or st.get(pid) not in ("pronoun", "pronoun_prev"):
                    continue
                match = [r for r in rows[i] if any(MI.has_key(r["value"], k) for k in MW.value_keys(s, v))]
                if match:
                    key = (st[pid], cue(W.people[pid]))
                    tab[key][0] += any(who.get(r["pid"]) == pid for r in match)
                    tab[key][1] += 1
        print(f"== {label}: true pronoun facts written down, share filed under the right person")
        for k in sorted(tab):
            print(f"   {k[0]:13s} {k[1]:42s} {100 * tab[k][0] / tab[k][1]:5.1f}% of {tab[k][1]}")
        if label != "A":
            continue
        ev, wrong, n_ev = CD.everyday_merges(W, V.ix, who)
        cat, frame = Counter(), Counter()
        for r in wrong:
            i = r["note"]
            nt = by[i]
            ms = (descs[i] or {}).get("mentions", [])
            src = next((c for c in P[i]["claims"] if c[0] == "fact" and (c[2] or "note") == r["kind"] and
                        c[3].strip() == r["value"] and V.pid(V.ent_of[i][c[1]]) == r["pid"]), None)
            owner = next((pid for pid, k, v in nt["facts"] if k in MW.SLOTS), None)
            if REPORT.match(r["kind"]):
                cat["a 'told me / says' claim filed on the teller"] += 1
            elif src and P[i]["descs"][src[1]][0] == "pron":
                cat["pronoun: " + (cue(W.people[owner]) if owner is not None else "?")] += 1
                if re.match(r"In (her|his|their) free time", nt["text"]):
                    frame["'In her free time, <name> loves ...'"] += 1
            else:
                t = TE.match_mention_v2(ms[src[1]].get("text", "") or "", nt["mentions"]) \
                    if src and src[1] < len(ms) else None
                cat["fact put on the anchor of X's relation" if t and t[2] == "via_anchor" else "other"] += 1
        print(f"== A: {len(wrong)} wrong everyday rows of {n_ev}, by cause")
        for k, v in cat.most_common():
            print(f"   {v:3d}  {k}")
        print(f"   of the pronoun rows, in the frame 'In her free time, <name> loves ...': {sum(frame.values())}")


if __name__ == "__main__":
    main()
