"""Real-LM test 3: canonical, snapped keys, plus context vs memory.

Test 2 (memtest2.py -> results/reallm2/summary.md) fixed most of the addressing: taking the person half
of the key from the subject's own tokens raised first-night recall from 36% to 96% on written wordings
and from 6% to 44% on unseen ones, and the novelty gate kept every known fact intact. Three problems
remained, each with a measured cause:
  - Unseen wordings 44%: the subject token's state depends on the words before it, so a name in the
    middle of a new sentence moved 30% of its slots.
  - Recall fell from 96% to 52% as facts grew: both key halves had 256 codes, but only 8 relations
    exist, so the facts reached only 20,902 of 65,536 slots; and the solver hit its iteration cap.
  - The gate let 20-27% of never-written people through: the key from the name's last token mostly
    reflects the last word piece, which many surnames share.

Changes here:
  person key    the name encoded on its own (no surrounding sentence), averaged over all its tokens,
                whitened; then SNAPPED to the nearest person already written if close enough (else
                the memory stays silent). Every wording of a question about a known person therefore
                addresses exactly the same slots.
  relation key  the question's last-position state, classified to the nearest known relation
                centroid (SNAPPED), so every wording of the same relation gives the same code.
  key split     1,024 person codes x 64 relation codes (was 256 x 256): all 65,536 slots reachable.
  solver        conjugate gradient with a diagonal (Jacobi) preconditioner, 500 iterations.
Snapping is "select, don't blend" applied to the address: a noisy cue is replaced by the stored
pattern it is closest to (pattern completion) instead of being used as is.

Also measured: the same questions answered (a) with N = 10 / 100 / 1,000 facts pasted into the
context, (b) with only the single best-matching fact selected into the context, and (c) from the
memory with no context at all.

  python reallm/memtest3.py            # full run, same 24,000 facts as tests 1 and 2
  python reallm/memtest3.py --tiny     # CPU smoke test
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
import memtest2 as M2  # noqa: E402

ROOT = M.ROOT


class Runner3(M.Runner):
    """Adds name encoding (names run on their own, mean over their tokens) and snapped-key memory reads."""

    def __init__(self, model, tok, layer, cands, dev):
        super().__init__(model, tok, layer, dev)
        self.cands = cands
        for l in cands:
            self.layers[l].register_forward_hook(self._name_hook(l))

    def _name_hook(self, l):
        def hook(module, args, output):
            st = self.state
            if st["mode"] != "names" or l not in st["want"]:
                return None
            h = (output[0] if isinstance(output, tuple) else output).float()
            m = st["mask"][..., None].float()
            st["out"].setdefault(l, []).append((h * m).sum(1) / m.sum(1))
            if l == max(st["want"]):
                raise M.Stop
            return None
        return hook

    def _hook(self, module, args, output):
        st = self.state
        if st["mode"] == "mem3":
            h = output[0] if isinstance(output, tuple) else output
            add = torch.zeros_like(h)
            add[:, -1] = st["mem"].read(st["V"], st["qe"], st["gate"], h[:, -1]).to(h.dtype)
            h = h + add
            return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h
        return super()._hook(module, args, output)

    @torch.no_grad()
    def names(self, names, layers, bs=256):
        out = {}
        for i in range(0, len(names), bs):
            ids, am, pos = self.encode(names[i:i + bs])
            self.state = {"mode": "names", "want": layers, "mask": am, "out": out}
            try:
                self.base(input_ids=ids, attention_mask=am, position_ids=pos)
            except M.Stop:
                pass
        self.state = {"mode": None}
        return {l: torch.cat(v) for l, v in out.items()}

    @torch.no_grad()
    def logits3(self, texts, qe, gate, state, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos = self.encode(texts[i:i + bs])
            st = dict(state, mask=am)
            if st["mode"] == "mem3":
                st["qe"], st["gate"] = qe[i:i + bs], gate[i:i + bs]
            self.state = st
            hs = self.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state[:, -1]
            outs.append(self.head(hs).float())
        self.state = {"mode": None}
        return torch.cat(outs)


class Memory3:
    def __init__(self, a, ent, rel, dev, seed):
        g = torch.Generator().manual_seed(seed)
        k1, k2 = torch.randn(a.nsub_e, a.qe, generator=g), torch.randn(a.nsub_r, a.qr, generator=g)
        self.K1 = (k1 / k1.norm(dim=1, keepdim=True)).to(dev)
        self.K2 = (k2 / k2.norm(dim=1, keepdim=True)).to(dev)
        self.nsub_e, self.nsub_r, self.K, self.tau = a.nsub_e, a.nsub_r, a.topk, a.tau
        self.ks_e, self.ks_r = min(a.topk, a.nsub_e), min(8, a.nsub_r)
        self.N = a.nsub_e * a.nsub_r
        self.mu_e, self.P_e = ent[0].to(dev), ent[1].to(dev)
        self.mu_r, self.P_r, self.C = rel[0].to(dev), rel[1].to(dev), rel[2].to(dev)
        G = torch.randn(self.C.shape[1], a.qr, generator=g).to(dev)
        self.codes = F.normalize(self.C @ G, dim=1)
        self.E = torch.zeros(0, a.qe, device=dev)
        self.theta = None

    def ent(self, hn):
        return F.normalize((hn.float() - self.mu_e) @ self.P_e, dim=1)

    def snap(self, q):
        """Nearest written person: (snapped key, gate open?, best cosine, index)."""
        if len(self.E) == 0:
            z = torch.zeros(len(q), device=q.device)
            return q, z, z - 1, z.long()
        s = q @ self.E.T
        best, j = s.max(1)
        gate = (best >= self.theta).float()
        return torch.where(gate[:, None] > 0, self.E[j], q), gate, best, j

    def rel_class(self, hr):
        z = (hr.float() - self.mu_r) @ self.P_r
        return torch.cdist(z, self.C).argmin(1)

    def lookup(self, qe, r):
        v1, i1 = (qe @ self.K1.T).topk(self.ks_e, dim=1)
        v2, i2 = (self.codes[r] @ self.K2.T).topk(self.ks_r, dim=1)
        c = (v1[:, :, None] + v2[:, None, :]).reshape(len(qe), -1)
        sc, t = c.topk(self.K, dim=1)
        idx = i1.gather(1, t // self.ks_r) * self.nsub_r + i2.gather(1, t % self.ks_r)
        return idx, torch.softmax(sc / self.tau, dim=1)

    def read(self, V, qe, gate, hr):
        idx, w = self.lookup(qe, self.rel_class(hr))
        return F.embedding_bag(idx, V, per_sample_weights=w, mode="sum") * gate[:, None]


def pcg(rows, B, X0, lam, iters, tol, prior=None):
    """Jacobi-preconditioned block CG for (W^T W + lam I) X = W^T B + lam * prior."""
    diag = torch.zeros(rows.N, device=B.device).index_add_(0, rows.idx.reshape(-1), (rows.w ** 2).reshape(-1))
    Dinv = (1.0 / (diag + lam))[:, None]
    A = lambda X: rows.tmv(rows.mv(X)) + lam * X
    rhs = rows.tmv(B)
    if prior is not None:
        rhs = rhs + lam * prior
    X = X0.clone()
    R = rhs - A(X)
    Z = Dinv * R
    P = Z.clone()
    rz = (R * Z).sum(0)
    bn = rhs.norm(dim=0) + 1e-12
    it = 0
    for it in range(1, iters + 1):
        AP = A(P)
        alpha = rz / ((P * AP).sum(0) + 1e-30)
        X += P * alpha
        R -= AP * alpha
        if bool(((R.norm(dim=0) / bn) < tol).all()):
            break
        Z = Dinv * R
        rz_new = (R * Z).sum(0)
        P = Z + P * (rz_new / (rz + 1e-30))
        rz = rz_new
    return X, it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--people", type=int, default=3000)
    ap.add_argument("--night", type=int, default=3000)
    ap.add_argument("--calib-people", type=int, default=300)
    ap.add_argument("--fresh-people", type=int, default=100)
    ap.add_argument("--extra-names", type=int, default=2000, help="names only, for the person-key whitening")
    ap.add_argument("--nsub-e", type=int, default=1024)
    ap.add_argument("--nsub-r", type=int, default=64)
    ap.add_argument("--qe", type=int, default=64)
    ap.add_argument("--qr", type=int, default=16)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--iters", type=int, default=500)
    ap.add_argument("--delta-steps", type=int, default=25)
    ap.add_argument("--delta-lr", type=float, default=0.2)
    ap.add_argument("--delta-wd", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=500)
    ap.add_argument("--ctx-n", default="10,100,1000")
    ap.add_argument("--ctx-queries", type=int, default=100)
    ap.add_argument("--methods", default="delta_rule,batch_ls,joint_ls")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reallm3"))
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tiny:
        args.people, args.night, args.calib_people, args.fresh_people, args.extra_names = 60, 160, 40, 10, 100
        args.nsub_e, args.nsub_r, args.qe, args.qr, args.topk = 64, 16, 16, 8, 8
        args.eval_n, args.delta_steps, args.device, args.ctx_n, args.ctx_queries = 40, 30, "cpu", "5,20,80", 10
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
    run = Runner3(model, tok, L, cands, dev)
    answers = M.single_token_answers(tok)
    methods = args.methods.split(",")

    # same people and facts as tests 1 and 2; fresh people and extra names come after them
    P, C, Fr = args.people, args.calib_people, args.fresh_people
    people = M.make_people(P + C + Fr + args.extra_names, rng)
    calib = M.make_facts(people[P:P + C], answers, rng)
    stream = M.make_facts(people[:P], answers, rng)
    fresh = M.make_facts(people[P + C:P + C + Fr], answers, rng)
    extra_names = people[P + C + Fr:]
    rel_ids = {r: i for i, r in enumerate(M.RELATIONS)}
    P_ = lambda fs, split: [p for f in fs for p in M.prompts(f, split)]
    R = {"config": dict(vars(args), model="tiny-random" if args.tiny else args.model), "layer": L, "n_layers": n_layers,
         "d": d, "slots": args.nsub_e * args.nsub_r, "facts_total": len(stream), "nights": []}
    log(f"memory after block {L}/{n_layers}; person-key candidate blocks {cands}; slots {R['slots']}")

    # ---- relation classifier: LDA over the 8 relations on calibration wordings (written wordings only)
    Hc_r = run.capture(P_(calib, "train"))
    rel_of = torch.tensor([rel_ids[f["rel"]] for f in calib]).repeat_interleave(4)
    mu_r, P_r, spec_r = M.fit_projection(Hc_r, rel_of, len(rel_ids) - 1, args.seed, "lda")
    Z = (Hc_r.float().cpu() - mu_r) @ P_r
    Cent = torch.stack([Z[rel_of == i].mean(0) for i in range(len(rel_ids))])
    R["relation_discriminant_ratios"] = [round(float(x), 1) for x in spec_r]

    # ---- person key: names alone, mean over tokens, whitened (PCA on calibration + extra names)
    whiten_names = sorted({f["name"] for f in calib}) + extra_names
    Hn = run.names(whiten_names, cands)
    Hn_b = run.names(whiten_names[::-1], cands, bs=7)  # same names, different batching: numerical noise floor
    choice = {}
    for l in cands:
        X = Hn[l].double().cpu()
        mu = X.mean(0)
        ev, U = torch.linalg.eigh(((X - mu).T @ (X - mu)) / len(X))
        Pe = (U[:, -args.qe:] / ev[-args.qe:].clamp(min=1e-9).sqrt()).float()
        q = F.normalize((Hn[l].float().cpu() - mu.float()) @ Pe, dim=1)
        qb = F.normalize((Hn_b[l].float().cpu().flip(0) - mu.float()) @ Pe, dim=1)
        same = (q * qb).sum(1)
        s = q @ q.T
        s.fill_diagonal_(-2)
        nn = s.max(1).values
        choice[l] = {"same_name_min_cos": round(float(same.min()), 5), "different_names_max_cos": round(float(nn.max()), 5),
                     "different_names_mean_nn_cos": round(float(nn.mean()), 4), "mu": mu.float(), "P": Pe,
                     "same": same, "nn": nn}
    Le = min(cands, key=lambda l: (choice[l]["different_names_max_cos"] - choice[l]["same_name_min_cos"],
                                  choice[l]["different_names_mean_nn_cos"]))
    same, nn = choice[Le]["same"], choice[Le]["nn"]
    lo, hi = float(same.min()), float(nn.max())
    if lo > hi:
        theta = (lo + hi) / 2
    else:
        ths = torch.linspace(0, 1, 2001)
        j = (same[None] >= ths[:, None]).float().mean(1) - (nn[None] >= ths[:, None]).float().mean(1)
        theta = float(ths[j.argmax()])
    R["person_key"] = {"block": Le, "theta": round(theta, 5),
                       "per_block": {l: {k: v for k, v in c.items() if k not in ("mu", "P", "same", "nn")} for l, c in choice.items()},
                       "calib_same_name_pass": round(float((same >= theta).float().mean()), 4),
                       "calib_different_name_pass": round(float((nn >= theta).float().mean()), 4)}
    log("person key", json.dumps(R["person_key"]))
    mem = Memory3(args, (choice[Le]["mu"], choice[Le]["P"]), (mu_r, P_r, Cent), dev, args.seed + 1)
    mem.theta = theta
    del Hn, Hn_b, Hc_r

    name_cache = {}

    def ent_keys(names):
        new = sorted({n for n in names if n not in name_cache})
        if new:
            for n, k in zip(new, mem.ent(run.names(new, [Le])[Le])):
                name_cache[n] = k
        return torch.stack([name_cache[n] for n in names])

    def write_people(names):
        """Add people not yet written; returns how many new names snapped onto someone else (false merges)."""
        merges = 0
        ent_keys(list(dict.fromkeys(names)))  # encode tonight's names in batches first
        for n in dict.fromkeys(names):
            q = ent_keys([n])
            _, g, _, j = mem.snap(q)
            if g[0] > 0:
                merges += int(written_names[int(j[0])] != n)
            else:
                mem.E = torch.cat([mem.E, q])
                written_names.append(n)
        return merges

    def query_keys(names, snap=True):
        q = ent_keys(names)
        if not snap:
            return q, torch.ones(len(q), device=dev)
        qs, g, _, _ = mem.snap(q)
        return qs, g

    def rel_acc(facts, split):
        cls = mem.rel_class(run.capture(P_(facts, split)))
        true = torch.tensor([rel_ids[f["rel"]] for f in facts for _ in M.prompts(f, split)], device=dev)
        return round(float((cls == true).float().mean()), 4)

    def recall(facts, V, T=None, splits=("train", "test"), reads=("off", "gate")):
        res = {}
        for split in splits:
            texts = P_(facts, split)
            names = [f["name"] for f in facts for _ in M.prompts(f, split)]
            ans = torch.tensor([f["aid"] for f in facts for _ in M.prompts(f, split)], device=dev)
            for r in reads:
                if r == "none":
                    lg = run.last_logits(texts, {"mode": None})
                elif r == "oracle":
                    lg = run.last_logits(texts, {"mode": "delta", "deltas": T.repeat_interleave(len(M.prompts(facts[0], split)), 0)})
                else:
                    qe, g = query_keys(names, snap=(r == "gate"))
                    lg = run.logits3(texts, qe, g, {"mode": "mem3", "mem": mem, "V": V})
                res[f"{split}_{r}"] = round(float((lg.argmax(1) == ans).float().mean()), 4)
        return res

    # known facts and never-written people, for damage
    known = [(p, tok.encode(" " + a, add_special_tokens=False), s) for p, a, s in M2.KNOWN]
    known = [(p, a[0], s) for p, a, s in known if len(a) == 1]
    lg = run.last_logits([p for p, _, _ in known], {"mode": None})
    ok = (lg.argmax(1).cpu() == torch.tensor([a for _, a, _ in known])).tolist()
    known = [(p, int(x), s) for (p, _, s), x in zip(known, lg.argmax(1).tolist())][:10] if args.tiny else [k for k, o in zip(known, ok) if o]
    fresh_p, fresh_n = [M.prompts(f, "train")[0] for f in fresh], [f["name"] for f in fresh]
    fresh_base = run.last_logits(fresh_p, {"mode": None}).argmax(1)
    R["known_facts_base_correct"] = len(known)

    def damage(V):
        out = {}
        for r in ("off", "gate"):
            qe, g = query_keys([s for _, _, s in known], snap=(r == "gate"))
            lk = run.logits3([p for p, _, _ in known], qe, g, {"mode": "mem3", "mem": mem, "V": V})
            out[f"known_{r}"] = round(float((lk.argmax(1) == torch.tensor([a for _, a, _ in known], device=dev)).float().mean()), 4)
            qe, g = query_keys(fresh_n, snap=(r == "gate"))
            lf = run.logits3(fresh_p, qe, g, {"mode": "mem3", "mem": mem, "V": V})
            out[f"fresh_changed_{r}"] = round(float((lf.argmax(1) != fresh_base).float().mean()), 4)
        _, g, _, _ = mem.snap(ent_keys([s for _, _, s in known]))
        out["known_gate_pass"] = round(float(g.mean()), 4)
        _, g, _, _ = mem.snap(ent_keys(fresh_n))
        out["fresh_gate_pass"] = round(float(g.mean()), 4)
        return out

    # ---- nights
    V = {m: torch.zeros(mem.N, d) for m in methods}
    written_names = []
    C_idx, C_w, C_fact = [], [], []
    T_all = torch.zeros(0, d, device=dev)
    early, merges = None, 0
    for s in range(0, len(stream), args.night):
        tn = time.time()
        new = stream[s: s + args.night]
        merges += write_people([f["name"] for f in new])
        qe, _ = query_keys([f["name"] for f in new for _ in range(4)])
        r = mem.rel_class(run.capture(P_(new, "train")))
        idx, w = mem.lookup(qe, r)
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
            R["relation_classified_correctly"] = {"written_wordings": rel_acc(early, "train"), "unseen_wordings": rel_acc(early, "test")}
            log("first night, no memory / target injected:", json.dumps(R["base_and_oracle_first_night"]),
                "relation accuracy", json.dumps(R["relation_classified_correctly"]))
        latest = new[-args.eval_n:]
        row = {"night": s // args.night + 1, "facts": s + len(new), "people_written": len(written_names), "false_merges": merges,
               "constraints": rows_all.idx.shape[0], "constraints_per_slot": round(rows_all.idx.shape[0] / mem.N, 3),
               "distinct_slots_touched": int(torch.unique(rows_all.idx).numel()), "target_s": round(t_targets, 1), "methods": {}}
        for m in methods:
            tm = time.time()
            Vm = V[m].to(dev)
            info = {}
            if m == "delta_rule":
                M.delta_rule(Vm, idx, w, Tn, steps=3, lr=0.5)
            elif m == "batch_ls":
                Vm, info["iters"] = pcg(rows_new, Tn.repeat_interleave(4, 0), Vm, args.anchor, args.iters, 1e-4, prior=Vm)
            elif m == "joint_ls":
                Vm, info["iters"] = pcg(rows_all, Trows_all, Vm, args.lam, args.iters, 1e-4)
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
        Vdel, it_w = pcg(rows_k, Tk, Vj, args.lam, 2 * args.iters, 1e-5)
        Vref, it_c = pcg(rows_k, Tk, torch.zeros_like(Vj), args.lam, 2 * args.iters, 1e-5)
        gf = [stream[i] for i in sorted(gone)]
        kept = [f for i, f in enumerate(early) if i not in gone]
        R["deletion"] = {"deleted_facts": len(gone), "max_abs_diff_vs_never_learned": float((Vdel - Vref).abs().max()),
                         "max_abs_value": float(Vref.abs().max()),
                         "deleted_before": recall(gf, Vj, reads=("gate",)), "deleted_after": recall(gf, Vdel, reads=("gate",)),
                         "deleted_base_no_memory": recall(gf, None, reads=("none",)),
                         "kept_first_night_before": recall(kept, Vj, reads=("gate",)),
                         "kept_first_night_after": recall(kept, Vdel, reads=("gate",)), "resolve_iters_warm_cold": [it_w, it_c]}
        log("deletion", json.dumps(R["deletion"]))
        del Vj, Vdel, Vref
        json.dump(R, open(out_json, "w"), indent=1)

    # ---- forward-only writes: per-answer codebook computed offline on calibration people
    n1 = min(args.night, len(stream))
    f1, e1 = stream[:n1], stream[: args.eval_n]
    Tc = run.optimize_deltas([M.prompts(f, "train") for f in calib], [f["aid"] for f in calib],
                             args.delta_steps, args.delta_lr, args.delta_wd)
    book = {}
    for f, t in zip(calib, Tc):
        book.setdefault(f["aid"], []).append(t)
    book = {a: torch.stack(v).mean(0) for a, v in book.items()}
    missing = [i for i, f in enumerate(f1) if f["aid"] not in book]
    Tb = torch.stack([book.get(f["aid"], T_all[i]) for i, f in enumerate(f1)])
    rows1 = M.Rows(C_idx[0], C_w[0], mem.N)
    R["codebook_targets"] = {"answers_missing_from_codebook": len(missing),
                             "cos_with_gradient_targets": round(float(F.cosine_similarity(Tb, T_all[:n1], dim=1).mean()), 3)}
    for name, T1 in (("gradient", T_all[:n1]), ("codebook", Tb)):
        V1, _ = pcg(rows1, T1.repeat_interleave(4, 0), torch.zeros(mem.N, d, device=dev), args.lam, args.iters, 1e-4)
        r = recall(e1, None, T1[: args.eval_n], reads=("oracle",))
        r.update(recall(e1, V1, reads=("gate",)))
        R["codebook_targets"][name] = r
        del V1
    log("codebook", json.dumps(R["codebook_targets"]))
    json.dump(R, open(out_json, "w"), indent=1)

    # ---- context vs memory: the same unseen-wording questions, three ways
    stmt = lambda f: f"{M.prompts(f, 'train')[0]} {f['answer']}."
    Vj = V["joint_ls"].to(dev) if "joint_ls" in methods else None
    Q = f1[: args.ctx_queries]
    qtext = [M.prompts(f, "test")[0] for f in Q]
    ans = torch.tensor([f["aid"] for f in Q], device=dev)
    qe, g = query_keys([f["name"] for f in Q])
    ctx = {"queries": len(Q), "no_context_no_memory": round(float((run.last_logits(qtext, {"mode": None}).argmax(1) == ans).float().mean()), 4)}
    if Vj is not None:
        lm = run.logits3(qtext, qe, g, {"mode": "mem3", "mem": mem, "V": Vj})
        ctx["memory_all_24k_facts_no_context"] = round(float((lm.argmax(1) == ans).float().mean()), 4)
    q_rel = mem.rel_class(run.capture(qtext))
    f1_keys, _ = query_keys([g_["name"] for g_ in f1])
    f1_rel = mem.rel_class(run.capture([M.prompts(g_, "train")[0] for g_ in f1]))
    crng = random.Random(args.seed + 5)
    for n in [int(x) for x in args.ctx_n.split(",")]:
        plain, sel = [], []
        for i, f in enumerate(Q):
            others = crng.sample([k for k in range(len(f1)) if k != i], min(n - 1, len(f1) - 1))
            chosen = others + [i]
            crng.shuffle(chosen)
            plain.append(" ".join(stmt(f1[c]) for c in chosen) + "\n\n" + qtext[i])
            # select one statement: same snapped person key and same relation class as the question
            ch = torch.tensor(chosen, device=dev)
            score = (f1_keys[ch] @ qe[i]) + (f1_rel[ch] == q_rel[i]).float()
            sel.append(stmt(f1[chosen[int(score.argmax())]]) + "\n\n" + qtext[i])
        bs = 1 if n >= 500 else (8 if n >= 50 else 64)
        a_plain = float((run.last_logits(plain, {"mode": None}, bs=bs).argmax(1) == ans).float().mean())
        a_sel = float((run.last_logits(sel, {"mode": None}).argmax(1) == ans).float().mean())
        ctx[f"N={n}"] = {"all_facts_in_context": round(a_plain, 4), "best_match_selected_into_context": round(a_sel, 4),
                         "context_tokens": len(tok.encode(plain[0], add_special_tokens=False))}
        log("context", n, json.dumps(ctx[f"N={n}"]))
        json.dump(dict(R, context_vs_memory=ctx), open(out_json, "w"), indent=1)
    R["context_vs_memory"] = ctx
    R["total_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_json, "w"), indent=1)
    open(os.path.join(args.out, "tiny_summary.md" if args.tiny else "summary.md"), "w", encoding="utf-8").write(report(R))
    log("done", out_json)


def report(R):
    ms = list(R["nights"][0]["methods"])
    fx = lambda v: f"{v:.3f}"
    L = [f"# Real-LM memory test 3 (snapped canonical keys): {R['config']['model']}", "",
         f"Memory after block {R['layer']} of {R['n_layers']}, {R['slots']:,} slots ({R['config']['nsub_e']} person codes x "
         f"{R['config']['nsub_r']} relation codes) x {R['d']} dims; {R['facts_total']:,} facts in nights of {R['config']['night']:,}; "
         "4 wordings written per fact.", "",
         "No memory / target injected directly (first-night facts): " + json.dumps(R["base_and_oracle_first_night"]), "",
         "Relation classified correctly: " + json.dumps(R["relation_classified_correctly"]), "",
         "Person key: " + json.dumps(R["person_key"]), "",
         "## First night's facts, top-1 accuracy after each night (no snap, no gate / snapped + gate)", "",
         "| Facts | People | False merges | Constraints/slot | Distinct slots | " + " | ".join(f"{m} written" for m in ms) + " | "
         + " | ".join(f"{m} unseen" for m in ms) + " |", "|" + "---|" * (5 + 2 * len(ms))]
    for n in R["nights"]:
        c = [f"{fx(n['methods'][m]['first_night']['train_off'])} / {fx(n['methods'][m]['first_night']['train_gate'])}" for m in ms]
        u = [f"{fx(n['methods'][m]['first_night']['test_off'])} / {fx(n['methods'][m]['first_night']['test_gate'])}" for m in ms]
        L.append(f"| {n['facts']:,} | {n['people_written']:,} | {n['false_merges']} | {n['constraints_per_slot']:.2f} | "
                 f"{n['distinct_slots_touched']:,} | " + " | ".join(c + u) + " |")
    L += ["", "## Latest night, damage, certificate (no snap, no gate / snapped + gate)", "",
          "| Facts | Method | Latest written | Latest unseen | Known facts | Never-written people: answer changed | Gate pass known / never-written | >20% error | Iters | Max value |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n in R["nights"]:
        for m in ms:
            i = n["methods"][m]
            L.append(f"| {n['facts']:,} | {m} | {fx(i['latest_night']['train_off'])} / {fx(i['latest_night']['train_gate'])} | "
                     f"{fx(i['latest_night']['test_off'])} / {fx(i['latest_night']['test_gate'])} | {fx(i['known_off'])} / {fx(i['known_gate'])} | "
                     f"{fx(i['fresh_changed_off'])} / {fx(i['fresh_changed_gate'])} | {fx(i['known_gate_pass'])} / {fx(i['fresh_gate_pass'])} | "
                     f"{i['cert_frac_over_0.2']:.4f} | {i.get('iters', '-')} | {i['value_max_abs']} |")
    L += ["", f"Known facts the base model answers correctly: {R['known_facts_base_correct']}.", ""]
    if "deletion" in R:
        L += ["## Deletion", "", "```", json.dumps(R["deletion"], indent=1), "```", ""]
    L += ["## Forward-only writes (per-answer codebook, first-night facts)", "", "```", json.dumps(R["codebook_targets"], indent=1), "```", "",
          "## Context vs memory (same unseen-wording questions)", "", "```", json.dumps(R["context_vs_memory"], indent=1), "```", "",
          f"Total time: {R['total_s'] / 60:.1f} min"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
