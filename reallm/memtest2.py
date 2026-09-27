"""Real-LM test 2: entity x relation product keys, the fix for what test 1 found.

Test 1 (memtest.py -> results/reallm/summary.md) showed the storage side was fine: a fact's target
added directly at the question's last position gave the answer 100% of the time, on written and on
unseen wordings. The address failed. The lookup key, taken from the question's last position,
carried the relation and hardly the person: different people with the same relation shared 60%
of their slots (other relations: 0.4%). Thousands of facts competed for the same slots, 99.9% of
constraints stayed >20% off even at 0.18 constraints per slot, and the joint solve recalled 36%.

Mechanism: a transformer builds a subject's identity at the subject's own tokens (early-middle
layers); the question's last position only pulls attributes out of it. A new, made-up person has
no attributes to pull, so almost nothing person-specific reaches the last position.

The fix tested here: the two halves of the product key come from two places,
  person half    = state at the subject's last token, after block Le (early-middle)
  relation half  = state at the question's last position, after block L (as in test 1)
so a slot is a (person code, relation code) pair: Fix 3's "entity | relation" canonical key, read
from the model's own states instead of generated text. The value is still added at block L, last
position. A novelty gate (read only if the person's key is close to one already written; GRACE's
deferral radius) keeps the memory silent for anyone never written.

Also tested: two targets that need no backward pass on the device, since test 1's in-context
difference failed (7.7%): the answer token's output-embedding direction ("logit lens"), and a
per-answer codebook averaged from gradient targets computed once, offline, on other people.

  python reallm/memtest2.py            # full run, same 24,000 facts as test 1
  python reallm/memtest2.py --tiny     # CPU smoke test
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtest as M  # noqa: E402

ROOT = M.ROOT

# (prompt, answer, subject) - facts the base model already knows, for the damage check
KNOWN = [
    ("The capital of France is", "Paris", "France"), ("The capital of Japan is", "Tokyo", "Japan"),
    ("The capital of Italy is", "Rome", "Italy"), ("The capital of Germany is", "Berlin", "Germany"),
    ("The capital of Spain is", "Madrid", "Spain"), ("The capital of Russia is", "Moscow", "Russia"),
    ("The capital of Egypt is", "Cairo", "Egypt"), ("The capital of England is", "London", "England"),
    ("The capital of Canada is", "Ottawa", "Canada"), ("The capital of Austria is", "Vienna", "Austria"),
    ("The capital of Portugal is", "Lisbon", "Portugal"), ("The capital of Greece is", "Athens", "Greece"),
    ("The capital of Norway is", "Oslo", "Norway"), ("The capital of Ireland is", "Dublin", "Ireland"),
    ("The capital of South Korea is", "Seoul", "South Korea"), ("The capital of Turkey is", "Ankara", "Turkey"),
    ("The capital of China is", "Beijing", "China"), ("The Eiffel Tower is located in", "Paris", "Eiffel Tower"),
    ("The Colosseum is located in", "Rome", "Colosseum"), ("Big Ben is located in", "London", "Big Ben"),
    ("Tokyo is the capital of", "Japan", "Tokyo"), ("Paris is the capital of", "France", "Paris"),
    ("Berlin is the capital of", "Germany", "Berlin"), ("Madrid is the capital of", "Spain", "Madrid"),
    ("Rome is the capital of", "Italy", "Rome"), ("The largest planet in the solar system is", "Jupiter", "solar system"),
    ("Romeo and Juliet was written by William", "Shakespeare", "Romeo and Juliet"),
    ("The Mona Lisa was painted by Leonardo da", "Vinci", "Mona Lisa"),
    ("The theory of relativity was developed by Albert", "Einstein", "theory of relativity"),
    ("Microsoft was founded by Bill", "Gates", "Microsoft"), ("Apple was co-founded by Steve", "Jobs", "Apple"),
    ("The tallest mountain in the world is Mount", "Everest", "tallest mountain"),
    ("The longest river in Africa is the", "Nile", "Africa"), ("The Great Wall is located in", "China", "Great Wall"),
    ("The Statue of Liberty is in New", "York", "Statue of Liberty"),
    ("The main language spoken in Brazil is", "Portuguese", "Brazil"),
    ("The main language spoken in Mexico is", "Spanish", "Mexico"), ("Most people in Germany speak", "German", "Germany"),
    ("Most people in France speak", "French", "France"), ("Most people in Italy speak", "Italian", "Italy"),
    ("Most people in Japan speak", "Japanese", "Japan"), ("Most people in Russia speak", "Russian", "Russia"),
    ("A baby dog is called a", "puppy", "baby dog"), ("Honey is made by", "bees", "Honey"),
    ("The Taj Mahal is located in", "India", "Taj Mahal"), ("The pyramids of Giza are in", "Egypt", "pyramids of Giza"),
    ("The largest ocean on Earth is the", "Pacific", "Earth"), ("The fastest land animal is the", "cheetah", "fastest land animal"),
    ("The sun rises in the", "east", "sun"), ("On a clear day the sky is", "blue", "sky"),
    ("Ripe bananas are", "yellow", "Ripe bananas"), ("Fresh grass is", "green", "Fresh grass"), ("Snow is", "white", "Snow"),
    ("Doctors usually work in a", "hospital", "Doctors"),
    ("Wimbledon is a famous tournament in the sport of", "tennis", "Wimbledon"),
    ("The Beatles were a band from", "Liverpool", "Beatles"), ("Mozart was a famous", "composer", "Mozart"),
    ("Toyota is a car company from", "Japan", "Toyota"), ("BMW is a car company from", "Germany", "BMW"),
    ("Ferrari is a car company from", "Italy", "Ferrari"), ("Volvo is a car company from", "Sweden", "Volvo"),
    ("Michael Jordan played the sport of", "basketball", "Michael Jordan"),
    ("Tiger Woods is famous for playing", "golf", "Tiger Woods"),
    ("Roger Federer is famous for playing", "tennis", "Roger Federer"),
    ("Jimi Hendrix was famous for playing the", "guitar", "Jimi Hendrix"),
    ("Yo-Yo Ma is famous for playing the", "cello", "Yo-Yo Ma"), ("The currency of Japan is the", "yen", "Japan"),
    ("A week has seven", "days", "week"), ("Ice is frozen", "water", "Ice"),
]


def subject_index(tok, prompt, subject):
    """Index of the subject's last token in the unpadded token sequence of the prompt."""
    pre = prompt[: prompt.index(subject) + len(subject)]
    ids_pre = tok.encode(pre, add_special_tokens=False)
    ids = tok.encode(prompt, add_special_tokens=False)
    n = len(ids_pre)
    if ids[:n] != ids_pre:  # a token straddles the boundary: take the last token that starts inside the subject
        n = max(k for k in range(1, len(ids) + 1) if len(tok.decode(ids[:k - 1])) < len(pre)) if len(ids) else 1
    return min(n, len(ids)) - 1


class Runner2(M.Runner):
    """Adds hooks at the candidate person-key layers; they record the state at the subject's last token."""

    def __init__(self, model, tok, layer, ent_layers, dev):
        super().__init__(model, tok, layer, dev)
        self.ent_layers = list(ent_layers)
        self.Le = self.ent_layers[0]
        for l in self.ent_layers:
            self.layers[l].register_forward_hook(self._ent_hook(l))
        self._si = {}

    def _ent_hook(self, l):
        def hook(module, args, output):
            st = self.state
            if st.get("subj") is None or l not in st["want"]:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[1] != st["T"]:  # incremental decoding step: the subject is in the cache
                return None
            st["he"][l] = h[torch.arange(h.shape[0], device=h.device), st["subj"]].float()
            return None
        return hook

    def _hook(self, module, args, output):
        st = self.state
        if st["mode"] == "capture2":
            h = output[0] if isinstance(output, tuple) else output
            st["out"].append(({l: st["he"][l] for l in st["want"]}, h[:, -1].float()))
            raise M.Stop
        if st["mode"] == "mem2":
            h = output[0] if isinstance(output, tuple) else output
            add = torch.zeros_like(h)
            add[:, -1] = st["mem"].read(st["V"], st["he"][self.Le], h[:, -1], st["gate"]).to(h.dtype)
            h = h + add
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return super()._hook(module, args, output)

    def encode2(self, texts, subjects):
        ids, am, pos = self.encode(texts)
        T = ids.shape[1]
        si = []
        for t, s in zip(texts, subjects):
            if (t, s) not in self._si:
                self._si[(t, s)] = subject_index(self.tok, t, s)
            si.append(self._si[(t, s)])
        subj = T - am.sum(1) + torch.tensor(si, device=self.dev)
        return ids, am, pos, subj

    @torch.no_grad()
    def capture2(self, texts, subjects, layers=None, bs=128):
        """States before any memory write: {layer: subject-token state}, last-position state after block L."""
        layers = layers or [self.Le]
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos, subj = self.encode2(texts[i:i + bs], subjects[i:i + bs])
            self.state = {"mode": "capture2", "out": outs, "subj": subj, "T": ids.shape[1], "want": layers, "he": {}}
            try:
                self.base(input_ids=ids, attention_mask=am, position_ids=pos)
            except M.Stop:
                pass
        self.state = {"mode": None}
        return {l: torch.cat([o[0][l] for o in outs]) for l in layers}, torch.cat([o[1] for o in outs])

    @torch.no_grad()
    def logits2(self, texts, subjects, state, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos, subj = self.encode2(texts[i:i + bs], subjects[i:i + bs])
            st = dict(state, subj=subj, T=ids.shape[1], want=[self.Le], he={}, mask=am)
            if "deltas" in st:
                st["delta"] = st["deltas"][i:i + bs]
            self.state = st
            hs = self.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state[:, -1]
            outs.append(self.head(hs).float())
        self.state = {"mode": None}
        return torch.cat(outs)


class Memory2:
    """Product keys whose halves come from different states: person half K1, relation half K2."""

    def __init__(self, nsub, qe, qr, K, ksub, tau, ent, rel, dev, seed):
        g = torch.Generator().manual_seed(seed)
        k1, k2 = torch.randn(nsub, qe, generator=g), torch.randn(nsub, qr, generator=g)
        self.K1 = (k1 / k1.norm(dim=1, keepdim=True)).to(dev)
        self.K2 = (k2 / k2.norm(dim=1, keepdim=True)).to(dev)
        self.nsub, self.K, self.ksub, self.tau, self.N = nsub, K, ksub, tau, nsub * nsub
        self.mu_e, self.P_e = ent[0].to(dev), ent[1].to(dev)
        self.mu_r, self.P_r = rel[0].to(dev), rel[1].to(dev)
        self.E, self.theta = None, None

    def qe(self, he):
        q = (he.float() - self.mu_e) @ self.P_e
        return q / (q.norm(dim=1, keepdim=True) + 1e-9)

    def lookup(self, he, hr):
        q1 = self.qe(he)
        q2 = (hr.float() - self.mu_r) @ self.P_r
        q2 = q2 / (q2.norm(dim=1, keepdim=True) + 1e-9)
        v1, i1 = (q1 @ self.K1.T).topk(self.ksub, dim=1)
        v2, i2 = (q2 @ self.K2.T).topk(self.ksub, dim=1)
        c = (v1[:, :, None] + v2[:, None, :]).reshape(len(q1), -1)
        sc, t = c.topk(self.K, dim=1)
        idx = i1.gather(1, t // self.ksub) * self.nsub + i2.gather(1, t % self.ksub)
        return idx, torch.softmax(sc / self.tau, dim=1)

    def nearest(self, q1):
        """Cosine to the closest person key written so far."""
        best = torch.full((len(q1),), -1.0, device=q1.device)
        for j in range(0, len(self.E), 65536):
            best = torch.maximum(best, (q1 @ self.E[j:j + 65536].T).max(1).values)
        return best

    def read(self, V, he, hr, gate):
        idx, w = self.lookup(he, hr)
        out = F.embedding_bag(idx, V, per_sample_weights=w, mode="sum")
        if gate:
            out = out * (self.nearest(self.qe(he)) >= self.theta).float()[:, None]
        return out


def fit_relation(H, rel_of, qr, seed):
    """Relation half: the n_rel-1 discriminant directions between relations, spread into qr dims."""
    nrel = int(rel_of.max()) + 1
    mu, P, spec = M.fit_projection(H, rel_of, nrel - 1, seed, "lda")
    g = torch.Generator().manual_seed(seed + 7)
    return mu, P @ torch.randn(nrel - 1, qr, generator=g) / (nrel - 1) ** 0.5, spec


def overlap_stats(idx_tr, idx_te, facts, K):
    """Slot overlap: a written wording left out vs the other written ones; unseen wordings vs written;
    different facts of the same / another relation."""
    tr, te = idx_tr.cpu().tolist(), idx_te.cpu().tolist()
    nf = len(facts)
    loo = np.mean([len(set(tr[f][j]) & set(sum((tr[f][i] for i in range(4) if i != j), []))) / K
                   for f in range(nf) for j in range(4)])
    own = np.mean([len(set(te[f][j]) & set(sum(tr[f], []))) / K for f in range(nf) for j in range(2)])
    same, other = [], []
    for f in range(nf):
        for g in range(f + 1, min(nf, f + 40)):
            ov = len(set(tr[f][0]) & set(tr[g][0])) / K
            (same if facts[f]["rel"] == facts[g]["rel"] else other).append(ov)
    return {"written_wording_left_out_slots_found": round(float(loo), 3),
            "unseen_wording_slots_found_in_written": round(float(own), 3),
            "different_facts_shared_slots_same_relation": round(float(np.mean(same)) if same else -1, 3),
            "different_facts_shared_slots_other_relation": round(float(np.mean(other)) if other else -1, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--people", type=int, default=3000)
    ap.add_argument("--night", type=int, default=3000)
    ap.add_argument("--calib-people", type=int, default=300)
    ap.add_argument("--fresh-people", type=int, default=100, help="never-written people, for false recall")
    ap.add_argument("--nsub", type=int, default=256)
    ap.add_argument("--qe", type=int, default=64)
    ap.add_argument("--qr", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--delta-steps", type=int, default=25)
    ap.add_argument("--delta-lr", type=float, default=0.2)
    ap.add_argument("--delta-wd", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--methods", default="delta_rule,batch_ls,joint_ls")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reallm2"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tiny:
        args.people, args.night, args.calib_people, args.fresh_people, args.nsub = 60, 160, 40, 10, 32
        args.qe, args.qr, args.topk, args.eval_n, args.delta_steps, args.device = 16, 16, 8, 40, 30, "cpu"
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "tiny_result.json" if args.tiny else "result.json")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
    model, tok = M.load(args)
    n_layers, d = model.config.num_hidden_layers, model.config.hidden_size
    L = args.layer if args.layer >= 0 else int(round(0.625 * n_layers)) - 1
    cands = sorted({int(round(n_layers * f)) - 1 for f in (1 / 6, 1 / 4, 1 / 3, 5 / 12)} & set(range(0, L))) or [0]
    run = Runner2(model, tok, L, cands, dev)
    answers = M.single_token_answers(tok)
    methods = args.methods.split(",")

    # same people and facts as test 1 (same seed, same order of draws); fresh people come after
    people = M.make_people(args.people + args.calib_people + args.fresh_people, rng)
    calib = M.make_facts(people[args.people: args.people + args.calib_people], answers, rng)
    stream = M.make_facts(people[: args.people], answers, rng)
    fresh = M.make_facts(people[args.people + args.calib_people:], answers, rng)
    rel_ids = {r: i for i, r in enumerate(M.RELATIONS)}
    P_ = lambda fs, split: [p for f in fs for p in M.prompts(f, split)]
    S_ = lambda fs, split: [f["name"] for f in fs for _ in M.prompts(f, split)]
    R = {"config": dict(vars(args), model="tiny-random" if args.tiny else args.model), "layer": L, "n_layers": n_layers,
         "d": d, "slots": args.nsub ** 2, "facts_total": len(stream), "person_layer_candidates": cands, "nights": []}
    log(f"memory after block {L}/{n_layers}; person-key candidate blocks {cands}; slots {args.nsub ** 2}")

    # ---- fit the two key halves on calibration people (written wordings only)
    Hc_e, Hc_r = run.capture2(P_(calib, "train"), S_(calib, "train"), cands)
    pid = {n: i for i, n in enumerate(sorted({f["name"] for f in calib}))}
    person_of = torch.tensor([pid[f["name"]] for f in calib]).repeat_interleave(4)
    rel_of = torch.tensor([rel_ids[f["rel"]] for f in calib]).repeat_interleave(4)
    rel = fit_relation(Hc_r, rel_of, args.qr, args.seed)
    R["relation_discriminant_ratios"] = [round(float(x), 1) for x in rel[2][:7]]

    # ---- choose the person-key block and projection using written wordings of 300 stream facts
    diag_facts = stream[:300]
    Htr_e, Htr_r = run.capture2(P_(diag_facts, "train"), S_(diag_facts, "train"), cands)
    Hte_e, Hte_r = run.capture2(P_(diag_facts, "test"), S_(diag_facts, "test"), cands)
    diag, best = {}, None
    for l in cands:
        for kind in ("lda", "lda_eq"):
            ent = M.fit_projection(Hc_e[l], person_of, args.qe, args.seed, kind)[:2]
            m = Memory2(args.nsub, args.qe, args.qr, args.topk, min(args.topk, args.nsub), args.tau, ent, rel[:2], dev, args.seed + 1)
            itr = m.lookup(Htr_e[l], Htr_r)[0].view(len(diag_facts), 4, -1)
            ite = m.lookup(Hte_e[l], Hte_r)[0].view(len(diag_facts), 2, -1)
            s = overlap_stats(itr, ite, diag_facts, args.topk)
            s["score"] = round(s["written_wording_left_out_slots_found"] - s["different_facts_shared_slots_same_relation"], 3)
            diag[f"block{l}_{kind}"] = s
            if best is None or s["score"] > best[0]:
                best = (s["score"], l, kind, m)
    _, Le, kind, mem = best
    run.Le = Le
    R["key_overlap"], R["person_key"] = diag, f"block{Le}_{kind}"
    log("key overlap", json.dumps(diag), "-> person key", R["person_key"])

    # ---- novelty gate threshold, from calibration people: leave one relation out
    q = mem.qe(Hc_e[Le].to(dev))
    cr = rel_of.to(dev)
    cp = person_of.to(dev)
    pos_s, neg_s = [], []
    for i in range(0, len(q), 1024):
        s = q[i:i + 1024] @ q.T
        same = cp[i:i + 1024, None] == cp[None, :]
        samerel = same & (cr[i:i + 1024, None] == cr[None, :])
        pos_s.append(s.masked_fill(~same | samerel, -2).max(1).values)
        neg_s.append(s.masked_fill(same, -2).max(1).values)
    pos_s, neg_s = torch.cat(pos_s), torch.cat(neg_s)
    ths = torch.linspace(-1, 1, 401, device=dev)
    j = ((pos_s[None] >= ths[:, None]).float().mean(1) - (neg_s[None] >= ths[:, None]).float().mean(1))
    mem.theta = float(ths[j.argmax()])
    R["gate"] = {"theta": round(mem.theta, 3), "calib_true_pass": round(float((pos_s >= mem.theta).float().mean()), 4),
                 "calib_false_pass": round(float((neg_s >= mem.theta).float().mean()), 4)}
    log("gate", json.dumps(R["gate"]))
    del Hc_e, Htr_e, Hte_e

    # ---- known facts the base gets right; fresh (never written) people for false recall
    known = [(p, tok.encode(" " + a, add_special_tokens=False), s) for p, a, s in KNOWN]
    known = [(p, a[0], s) for p, a, s in known if len(a) == 1]
    lg = run.logits2([p for p, _, _ in known], [s for _, _, s in known], {"mode": None})
    ok = (lg.argmax(1).cpu() == torch.tensor([a for _, a, _ in known])).tolist()
    if args.tiny:
        known = [(p, int(x), s) for (p, _, s), x in zip(known, lg.argmax(1).tolist())][:10]
    else:
        known = [k for k, o in zip(known, ok) if o]
    fresh_p = [M.prompts(f, "train")[0] for f in fresh]
    fresh_s = [f["name"] for f in fresh]
    fresh_base = run.logits2(fresh_p, fresh_s, {"mode": None}).argmax(1)
    R["known_facts_base_correct"] = len(known)

    def recall(facts, V, T=None, splits=("train", "test"), reads=("off", "gate")):
        res = {}
        for split in splits:
            texts, subs = P_(facts, split), S_(facts, split)
            ans = torch.tensor([f["aid"] for f in facts for _ in M.prompts(f, split)], device=dev)
            for r in reads:
                if r == "none":
                    st = {"mode": None}
                elif r == "oracle":
                    st = {"mode": "delta", "deltas": T.repeat_interleave(len(M.prompts(facts[0], split)), 0)}
                else:
                    st = {"mode": "mem2", "mem": mem, "V": V, "gate": r == "gate"}
                res[f"{split}_{r}"] = round(float((run.logits2(texts, subs, st).argmax(1) == ans).float().mean()), 4)
        return res

    def damage(V):
        out = {}
        for r in ("off", "gate"):
            st = {"mode": "mem2", "mem": mem, "V": V, "gate": r == "gate"}
            lk = run.logits2([p for p, _, _ in known], [s for _, _, s in known], st)
            out[f"known_{r}"] = round(float((lk.argmax(1) == torch.tensor([a for _, a, _ in known], device=dev)).float().mean()), 4)
            lf = run.logits2(fresh_p, fresh_s, st)
            out[f"fresh_changed_{r}"] = round(float((lf.argmax(1) != fresh_base).float().mean()), 4)
        return out

    # ---- nights
    V = {m: torch.zeros(mem.N, d) for m in methods}
    C_idx, C_w, C_fact, E_all = [], [], [], []
    T_all = torch.zeros(0, d, device=dev)
    early = None
    for s in range(0, len(stream), args.night):
        tn = time.time()
        new = stream[s: s + args.night]
        He, Hr = run.capture2(P_(new, "train"), S_(new, "train"), [Le])
        idx, w = mem.lookup(He[Le], Hr)
        E_all.append(mem.qe(He[Le]))
        mem.E = torch.cat(E_all)
        idx, w = idx.view(len(new), 4, -1), w.view(len(new), 4, -1)
        Tn = run.optimize_deltas([M.prompts(f, "train") for f in new], [f["aid"] for f in new],
                                 args.delta_steps, args.delta_lr, args.delta_wd)
        t_targets = time.time() - tn
        C_idx.append(idx.reshape(-1, idx.shape[-1]))
        C_w.append(w.reshape(-1, w.shape[-1]))
        C_fact.append(torch.arange(s, s + len(new), device=dev).repeat_interleave(4))
        T_all = torch.cat([T_all, Tn])
        rows_all = M.Rows(torch.cat(C_idx), torch.cat(C_w), mem.N)
        rows_new = M.Rows(C_idx[-1], C_w[-1], mem.N)
        Trows_all = T_all[torch.cat(C_fact)]
        if early is None:
            early = new[: args.eval_n]
            R["base_and_oracle_first_night"] = recall(early, None, Tn[: args.eval_n], reads=("none", "oracle"))
            log("first night, no memory / target injected:", json.dumps(R["base_and_oracle_first_night"]))
        latest = new[-args.eval_n:]
        row = {"night": s // args.night + 1, "facts": s + len(new), "constraints": rows_all.idx.shape[0],
               "constraints_per_slot": round(rows_all.idx.shape[0] / mem.N, 3),
               "distinct_slots_touched": int(torch.unique(rows_all.idx).numel()), "target_s": round(t_targets, 1), "methods": {}}
        if row["night"] == 1:
            he_te, _ = run.capture2(P_(early, "test"), S_(early, "test"), [Le])
            R["gate_pass_unseen_wordings_written_people"] = round(float((mem.nearest(mem.qe(he_te[Le])) >= mem.theta).float().mean()), 4)
            he_f, _ = run.capture2(fresh_p, fresh_s, [Le])
            R["gate_pass_fresh_people_night1"] = round(float((mem.nearest(mem.qe(he_f[Le])) >= mem.theta).float().mean()), 4)
            log("gate pass: unseen wordings of written people", R["gate_pass_unseen_wordings_written_people"],
                "| never-written people", R["gate_pass_fresh_people_night1"])
        for m in methods:
            tm = time.time()
            Vm = V[m].to(dev)
            info = {}
            if m == "delta_rule":
                M.delta_rule(Vm, idx, w, Tn, steps=3, lr=0.5)
            elif m == "batch_ls":
                Vm, info["iters"] = M.cg(rows_new, Tn.repeat_interleave(4, 0), Vm, args.anchor, 300, 1e-4, prior=Vm)
            elif m == "joint_ls":
                Vm, info["iters"] = M.cg(rows_all, Trows_all, Vm, args.lam, 300, 1e-4)
            info["write_s"] = round(time.time() - tm, 1)
            rel_err = M.certificate(rows_all, Vm, Trows_all)
            info["cert_frac_over_0.2"] = round(float((rel_err > 0.2).float().mean()), 4)
            info["cert_mean_rel_err"] = round(float(rel_err.mean()), 4)
            info["first_night"] = recall(early, Vm)
            info["latest_night"] = recall(latest, Vm)
            info.update(damage(Vm))
            info["value_max_abs"] = round(float(Vm.abs().max()), 2)
            info["eval_s"] = round(time.time() - tm - info["write_s"], 1)
            row["methods"][m] = info
            V[m] = Vm.cpu()
            del Vm
        if dev == "cuda":
            row["max_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            torch.cuda.empty_cache()
        he_f, _ = run.capture2(fresh_p, fresh_s, [Le])
        row["gate_pass_fresh_people"] = round(float((mem.nearest(mem.qe(he_f[Le])) >= mem.theta).float().mean()), 4)
        row["night_s"] = round(time.time() - tn, 1)
        R["nights"].append(row)
        log(json.dumps(row))
        json.dump(R, open(out_json, "w"), indent=1)

    # ---- exact deletion (joint_ls)
    if "joint_ls" in methods:
        gone = set(range(min(300, args.eval_n // 2)))
        cf = torch.cat(C_fact)
        keep = torch.tensor([int(f) not in gone for f in cf.tolist()], device=dev)
        rows_k = M.Rows(torch.cat(C_idx)[keep], torch.cat(C_w)[keep], mem.N)
        Tk = T_all[cf[keep]]
        Vj = V["joint_ls"].to(dev)
        Vdel, it_w = M.cg(rows_k, Tk, Vj, args.lam, 600, 1e-5)
        Vref, it_c = M.cg(rows_k, Tk, torch.zeros_like(Vj), args.lam, 600, 1e-5)
        gf = [stream[i] for i in sorted(gone)]
        kept = [f for i, f in enumerate(early) if i not in gone]
        R["deletion"] = {"deleted_facts": len(gone), "max_abs_diff_vs_never_learned": float((Vdel - Vref).abs().max()),
                         "max_abs_value": float(Vref.abs().max()),
                         "deleted_before": recall(gf, Vj, reads=("off",)), "deleted_after": recall(gf, Vdel, reads=("off",)),
                         "deleted_base_no_memory": recall(gf, None, reads=("none",)),
                         "kept_first_night_before": recall(kept, Vj, reads=("off",)),
                         "kept_first_night_after": recall(kept, Vdel, reads=("off",)), "resolve_iters_warm_cold": [it_w, it_c]}
        log("deletion", json.dumps(R["deletion"]))
        del Vj, Vdel, Vref
        json.dump(R, open(out_json, "w"), indent=1)

    # ---- targets that need no backward pass on the device
    n1 = min(args.night, len(stream))
    f1, e1 = stream[:n1], stream[: args.eval_n]
    Tg = T_all[:n1]
    targets = {"gradient (backward pass per fact)": Tg}
    # (a) answer-token output-embedding direction, pushed back through the final norm's gain
    W = run.head.weight.float()
    gain = model.model.norm.weight.float()
    _, Hr1 = run.capture2(P_(calib[:200], "train"), S_(calib[:200], "train"), [Le])
    hnorm = float(Hr1.norm(dim=1).mean())
    u = lambda fs: F.normalize((W[[f["aid"] for f in fs]] - W.mean(0)) / gain, dim=1) * hnorm
    sc = {a: M.acc_block(run, calib[:200], None, None, u(calib[:200]) * a, which=("train",), reads=("oracle",))["train_oracle"]
          for a in (0.25, 0.5, 1, 2, 4)}
    a_best = max(sc, key=sc.get)
    targets["logit-lens direction (no backward pass)"] = u(f1) * a_best
    # (b) per-answer codebook: gradient targets computed once, offline, on calibration people only
    Tc = run.optimize_deltas([M.prompts(f, "train") for f in calib], [f["aid"] for f in calib],
                             args.delta_steps, args.delta_lr, args.delta_wd)
    book = {}
    for f, t in zip(calib, Tc):
        book.setdefault(f["aid"], []).append(t)
    book = {a: torch.stack(v).mean(0) for a, v in book.items()}
    missing = sum(f["aid"] not in book for f in f1)
    targets["answer codebook (offline, no backward pass on device)"] = torch.stack(
        [book[f["aid"]] if f["aid"] in book else u([f])[0] * a_best for f in f1])
    cos = F.cosine_similarity(targets["answer codebook (offline, no backward pass on device)"], Tg, dim=1)
    R["no_backward_targets"] = {"logit_lens_scale_search": sc, "logit_lens_scale": a_best,
                                "codebook_answers_missing": missing, "codebook_vs_gradient_cos": round(float(cos.mean()), 3)}
    rows1 = M.Rows(C_idx[0], C_w[0], mem.N)
    for name, T1 in targets.items():
        V1, _ = M.cg(rows1, T1.repeat_interleave(4, 0), torch.zeros(mem.N, d, device=dev), args.lam, 300, 1e-4)
        r = recall(e1, None, T1[: args.eval_n], reads=("oracle",))
        r.update(recall(e1, V1, reads=("off",)))
        R["no_backward_targets"][name] = r
        del V1
    log("targets", json.dumps(R["no_backward_targets"]))
    R["total_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_json, "w"), indent=1)
    open(os.path.join(args.out, "tiny_summary.md" if args.tiny else "summary.md"), "w", encoding="utf-8").write(report(R))
    log("done", out_json)


def report(R):
    ms = list(R["nights"][0]["methods"])
    fx = lambda v: f"{v:.3f}"
    L = [f"# Real-LM memory test 2 (person x relation keys): {R['config']['model']}", "",
         f"Memory after block {R['layer']} of {R['n_layers']}, {R['slots']:,} slots x {R['d']} dims; "
         f"{R['facts_total']:,} facts in nights of {R['config']['night']:,}; 4 wordings written per fact. "
         f"Person key: {R['person_key']}. Relation discriminant ratios: {R['relation_discriminant_ratios']}.", "",
         "No memory / target injected directly (first-night facts): " + json.dumps(R["base_and_oracle_first_night"]), "",
         "Gate: " + json.dumps(R["gate"]) + f"; unseen wordings of written people pass: {R['gate_pass_unseen_wordings_written_people']}; "
         f"never-written people pass (night 1): {R['gate_pass_fresh_people_night1']}", "",
         "Key overlap per candidate: " + json.dumps(R["key_overlap"]), "",
         "## First night's facts, top-1 accuracy after each night (gate off / gate on)", "",
         "| Facts | Constraints/slot | Distinct slots | " + " | ".join(f"{m} written" for m in ms) + " | "
         + " | ".join(f"{m} unseen" for m in ms) + " |", "|" + "---|" * (3 + 2 * len(ms))]
    for n in R["nights"]:
        c = [f"{fx(n['methods'][m]['first_night']['train_off'])} / {fx(n['methods'][m]['first_night']['train_gate'])}" for m in ms]
        u = [f"{fx(n['methods'][m]['first_night']['test_off'])} / {fx(n['methods'][m]['first_night']['test_gate'])}" for m in ms]
        L.append(f"| {n['facts']:,} | {n['constraints_per_slot']:.2f} | {n['distinct_slots_touched']:,} | " + " | ".join(c + u) + " |")
    L += ["", "## Latest night, damage, certificate (gate off / gate on)", "",
          "| Facts | Method | Latest written | Latest unseen | Known facts | Never-written people: answer changed | >20% error | Iters | Max value |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n in R["nights"]:
        for m in ms:
            i = n["methods"][m]
            L.append(f"| {n['facts']:,} | {m} | {fx(i['latest_night']['train_off'])} / {fx(i['latest_night']['train_gate'])} | "
                     f"{fx(i['latest_night']['test_off'])} / {fx(i['latest_night']['test_gate'])} | {fx(i['known_off'])} / {fx(i['known_gate'])} | "
                     f"{fx(i['fresh_changed_off'])} / {fx(i['fresh_changed_gate'])} | {i['cert_frac_over_0.2']:.4f} | {i.get('iters', '-')} | {i['value_max_abs']} |")
    L += ["", f"Known facts the base model answers correctly: {R['known_facts_base_correct']}. "
          f"Never-written people passing the gate after the last night: {R['nights'][-1]['gate_pass_fresh_people']}", ""]
    if "deletion" in R:
        L += ["## Deletion", "", "```", json.dumps(R["deletion"], indent=1), "```", ""]
    L += ["## Targets without a backward pass on the device (first-night facts, joint solve)", "", "```",
          json.dumps(R["no_backward_targets"], indent=1), "```", "", f"Total time: {R['total_s'] / 60:.1f} min"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
