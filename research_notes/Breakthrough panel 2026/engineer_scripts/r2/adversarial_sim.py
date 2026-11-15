"""Round 2: how much of round 1's resolver result was the designer's advantage?

Three changes against round 1's resolver_sim.py (which is imported, not edited):
  1. ANCHOR ERRORS. Round 1 dropped the anchor ("Shirin" in "Shirin's colleague") from the mention list, so the most
     common real error (the claim filed under the anchor: 15 of B's 49 wrong rows) could never be simulated. Here, with
     probability hard_err, the describer writes the claim about the anchor instead of the via mention.
  2. CONCENTRATED ERRORS. Real errors cluster on hard messages. Messages with a via, source (second-hand) or
     previous-message pronoun get hard_err; all others get easy_err.
  3. CONVENTION BREAKS. A subclass of the generator (repo file untouched) lets the world use references the original
     never emits: a bare first name when two known people share it (amb), "X's <rel>" when X has several (nonuniq), and
     "X's <rel>" before the relation was ever stated (early). The resolver is unchanged except for an optional STRICT
     mode: when more than one candidate fits, leave the mention unfiled instead of picking the most recent.

Run: python3 adversarial_sim.py   (reads brain/messy_world.py, writes nothing in the repo)
"""
import random
import sys
from collections import defaultdict

sys.path.insert(0, "/tmp/claude-0/-home-user-nichtec/9f081756-7692-5083-b72d-9d96c9efa3d9/scratchpad/panel/engineer")
sys.path.insert(0, "/home/user/nichtec/brain")
import messy_world as MW  # noqa: E402
import resolver_sim as R  # noqa: E402

RELS, FRAMES = MW.RELS, MW.FRAMES


class LooseWorld(MW.World):
    """Same generator, looser referring conventions (closer to real chat)."""

    def __init__(self, split="dev", amb=0.0, nonuniq=0.0, early=0.0):
        self.amb, self.nonuniq, self.early = amb, nonuniq, early
        self.n_amb = self.n_nonuniq = self.n_early = 0
        super().__init__(split)

    def _mention(self, pid, ctx, allow_pron=False, cap=False, via_ok=True):
        r, p = self.rng, self.people[pid]
        styles = [("full", 25)]
        role = self.role_at(p, ctx["date"])
        if role:
            styles.append(("role", 30))
        if p["role"] and p.get("role_until") and ctx["date"] >= p["role_until"]:
            styles.append(("old_role", 12))
        if p["nick"] and p["nick_known"]:
            styles.append(("nick", 12))
        fc = (p["first"], p["circle"])
        if p["circle"] and self.known_fc[fc] == {pid}:
            styles.append(("fname_ctx", 15))
        recent = [x for x in ctx["recent"][-3:]]
        same_first_recent = [q for q in ctx["recent"][-10:] if self.people[q]["first"] == p["first"] and q != pid]
        amb_use = False
        if self.known_first[p["first"]] == {pid}:
            styles.append(("fname", 15))
        elif pid in recent and not same_first_recent:
            styles.append(("fname", 40))
        elif self.amb and pid in self.known_first[p["first"]] and r.random() < self.amb:
            styles.append(("fname", 15))                    # real chat: "Mira" though two Miras are known
            amb_use = True
        via = []
        for rel, b, way in p["links"]:
            if not (RELS[rel][0] or way == "out"):
                continue
            known = self.link_known.get((pid, rel, b))
            uniq = self.unique_role(b, rel)
            if known and uniq:
                via.append((rel, b, ""))
            elif self.nonuniq and known and not uniq and r.random() < self.nonuniq:
                via.append((rel, b, "nonuniq"))
            elif self.early and not known and uniq and r.random() < self.early:
                via.append((rel, b, "early"))
        if via and via_ok and not p["close"]:
            styles.append(("via", 6))
        names, weights = zip(*styles)
        st = r.choices(names, weights)[0]
        if st == "full":
            t = p["full"]
        elif st == "role":
            t = f"my {p['role']}"
        elif st == "old_role":
            t = f"my old {p['role']}"
        elif st == "nick":
            t = p["nick"]
        elif st == "fname_ctx":
            t = r.choice(FRAMES["fname_ctx"][p["circle"]]).format(first=p["first"])
        elif st == "fname":
            t = p["first"]
            self.n_amb += amb_use
        else:
            rel, b, kind = r.choice(via)
            self.n_nonuniq += kind == "nonuniq"
            self.n_early += kind == "early"
            bt, bst = self._mention(b, ctx, via_ok=False)
            t = r.choice(FRAMES["via"]).format(anchor=bt, noun=RELS[rel][3])
            st = "via"
            self.last_anchor = (bt, b, "via_anchor")
        return (t[0].upper() + t[1:] if cap else t), st


def hard(n):
    return any(st in ("via", "source", "pronoun_prev") for _, _, st in n["mentions"])


def local_parse2(W, easy_err, hard_err, rng):
    """Round 1's local_parse with per-message error rates, plus anchor errors on via claims."""
    # round 1's parser needs the previous message for pronouns, so run it once per error class and pick per message
    lo = R.local_parse(W, easy_err, random.Random(rng.random()))
    hi = R.local_parse(W, hard_err, random.Random(rng.random()))
    out = {}
    for n in W.notes:
        p = hi[n["id"]] if hard(n) else lo[n["id"]]
        # anchor errors: the claim about "Shirin's colleague" written about "Shirin" instead
        if any(st == "via" for _, _, st in n["mentions"]):
            descs = list(p["descs"])
            for k, d in enumerate(descs):
                if d[0] == "via" and rng.random() < hard_err:
                    descs[k] = d[1]                         # describer wrote the anchor
            p = dict(p, descs=descs)
        out[n["id"]] = p
    return out


def resolve2(P, strict=False, iters=4):
    """Round 1's resolver; strict=True never picks among several candidates (leaves the mention unfiled)."""
    if not strict:
        return R.resolve(P, iters=iters)
    orig = R.resolve.__code__
    # strict mode: rebuild the resolver with rec_pick returning None (copy of R.resolve with one line changed)
    src = open(R.__file__).read()
    start = src.index("def resolve(")
    end = src.index("def evaluate(")
    code = src[start:end].replace("def resolve(", "def resolve_strict(").replace(
        "            for e in reversed(recent[-30:]):\n                if e in cands:\n                    return e\n"
        "            return None", "            return None")
    ns = dict(R.__dict__)
    exec(code, ns)
    return ns["resolve_strict"](P, iters=iters)


def run(label, W, easy, hardr, strict, seed=7):
    rng = random.Random(seed)
    P = local_parse2(W, easy, hardr, rng)
    bind = resolve2(P, strict=strict)
    ev = R.evaluate(W, P, bind)
    firsts = defaultdict(int)
    for p in W.people:
        firsts[p["first"]] += 1
    targets = [p["id"] for p in W.people if p["close"]] + [p["id"] for p in W.people if firsts[p["first"]] > 1][:10]
    fg = R.forget(W, P, bind, targets)
    rb = ev["right_of_filed_by_style"]
    ub = ev["unfiled_by_style"]
    print(f"{label:58s} wrong-of-filed {ev['wrong_pct_of_filed']:5.2f}%  filed-right {ev['filed_right_pct_of_all']:5.1f}%"
          f"  via right/unfiled {rb.get('via', 0):5.1f}/{ub.get('via', 0):4.1f}  fname right/unfiled "
          f"{rb.get('fname', 0):5.1f}/{ub.get('fname', 0):4.1f}  forget worst {fg[1]:5.1f}%")
    return ev


def main():
    print("== A. Original world (the generator's conventions), round-1 resolver ==")
    W0 = MW.World("dev")
    run("A0 round-1 setting: uniform 10%, no anchor errors", W0, 0.10, 0.10, False)
    for easy, hardr in ((0.05, 0.20), (0.05, 0.30), (0.05, 0.50)):
        run(f"A1 easy {easy:.0%}, hard {hardr:.0%} incl. anchor errors", W0, easy, hardr, False)

    print("\n== B. Looser conventions (real-chat-like), perfect describer ==")
    for amb, nonuniq, early in ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.3, 0.5, 0.5)):
        W = LooseWorld("dev", amb, nonuniq, early)
        tag = f"amb {amb:.1f} nonuniq {nonuniq:.1f} early {early:.1f} (n {W.n_amb}/{W.n_nonuniq}/{W.n_early})"
        run("B guess  " + tag, W, 0.0, 0.0, False)
        run("B strict " + tag, W, 0.0, 0.0, True)

    print("\n== C. Looser conventions AND concentrated describer errors (easy 5%, hard 30%) ==")
    W = LooseWorld("dev", 0.3, 0.5, 0.5)
    run("C guess ", W, 0.05, 0.30, False)
    run("C strict", W, 0.05, 0.30, True)


if __name__ == "__main__":
    main()
