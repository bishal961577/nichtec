"""Test 2 check (amendment 2): with a perfect writer, how often is the person a message mentions on the writer's list
of known people (`messy_index.candidates`)? No model, no GPU.

The index is built from the truth chunk by chunk, as B's writer would build it if it filed everything correctly. Before
each chunk, the list is computed exactly as the real run computes it. A mention counts when that person was already in
the index before the chunk (someone new cannot be on the list). Reported by mention style.

    python3 brain/messy_coverage.py            # dev world, seed 101
    python3 brain/messy_coverage.py --seed 7   # a placeholder test world (normal wordings, not the sealed ones)
"""
import argparse
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import messy_world as MW  # noqa: E402
import messy_index as MI  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    W = MW.World(a.split, seed=a.seed)
    ix, pid_of, recent = MI.Index(), {}, []
    hit, tot, sizes = Counter(), Counter(), []
    notes = W.notes
    for c0 in range(0, len(notes), MI.CHUNK):
        chunk = notes[c0:c0 + MI.CHUNK]
        ctx = notes[max(0, c0 - MI.CONTEXT):c0]
        cands = set(MI.candidates(ix, ctx + chunk, recent))
        sizes.append(len(cands))
        for n in chunk:
            for m in n["mentions"]:
                style, pid = m[2], m[1]
                if pid in pid_of:
                    tot[style] += 1
                    hit[style] += pid_of[pid] in cands
        for n in chunk:                                  # the perfect writer files the chunk
            for m in n["mentions"]:
                if m[1] not in pid_of:
                    pid_of[m[1]] = ix.person(W.people[m[1]]["full"])
            stated = set()
            for pid, kind, value in n["facts"]:
                for q in ((pid, value) if kind in MW.RELS else (pid,)):
                    if q not in pid_of:                      # (not setdefault: that would create a person each time)
                        pid_of[q] = ix.person(W.people[q]["full"])
                x = pid_of[pid]
                if kind in MW.RELS:
                    y = pid_of[value]
                    ix.add(n["id"], x, kind, y, link=y)
                elif kind == "event":
                    ix.add(n["id"], x, "event", f"{value[1]} {value[0]}")
                elif kind in MW.SLOTS:
                    ix.add(n["id"], x, kind, "none" if value is None else value)
                else:
                    ix.add(n["id"], x, kind, str(value))
                stated.add(pid)
                recent.append(x)
            for _, pid, _ in n["mentions"]:
                if pid not in stated:
                    ix.add(n["id"], pid_of[pid], "mentioned", "-")
                    stated.add(pid)
                    recent.append(pid_of[pid])
    print(f"{a.split} world: {len(ix.people)} people in the index, list size median {sorted(sizes)[len(sizes) // 2]}, "
          f"max {max(sizes)}")
    for st in sorted(tot, key=lambda s: -tot[s]):
        print(f"  {st:13s} {100 * hit[st] / tot[st]:5.1f}% of {tot[st]}")
    allh, allt = sum(hit.values()), sum(tot.values())
    print(f"  {'all':13s} {100 * allh / allt:5.1f}% of {allt}")


if __name__ == "__main__":
    main()
