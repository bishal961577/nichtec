"""Native-key test: can a frozen LM's own hidden states find a stored subject under wordings it never saw, and stay
silent for names it never stored? And does the model form the hidden middle entity of a two-hop question in a form a
memory could be keyed on?

Why this decides the direction. Tests 1-5 hit one wall (the reliability / generalisation / locality triangle of
lifelong editing): keys read from the model's own states inside a sentence generalised badly (test 1: 36%, test 2:
44% on unseen wordings), while the name encoded on its own and snapped to the nearest stored name generalised
(test 3: 96.5%) but needs an external lookup, which the model's own reasoning never triggers (test 5: 2.5% of
two-hop questions asked directly). If the model's own early-layer state for a name inside a sentence is close enough
to the name encoded alone, the memory can be addressed natively. If not, native addressing is closed for this design.

Nothing is written or trained here: forward passes only. Thresholds are set on calibration names that share no
subject with the test set; the verdict layer is chosen on calibration data, not on the test data.

  E1  stored key = name encoded alone (with a leading space); query = the name's state inside unseen sentences
  E2  stored key = the name's state inside its write-time sentence; query = inside unseen sentences
      (both at the name's last token and averaged over its tokens; the name's position is given)
  E3  native scan: no position given; every token of the sentence queries the stored keys, and the memory fires
      wherever the best cosine passes the calibrated threshold
  E4  two-hop questions: is the hidden middle entity (e.g. the country in "the capital of the country of citizenship
      of X") decodable from the question's last-token state, by a linear map fitted on calibration questions?

Pre-registered verdict (FIXING_THE_BOTTLENECKS.md, test 6):
  native addressing viable: E3 hit >= 90% on unseen wordings with false fires <= 2% on never-stored names
  native addressing closed: best E3 hit < 70%; in between: needs a learned canonicaliser
  multi-hop hook: E4 middle-entity top-1 >= 50% (cases whose first hop the model answers correctly); none if < 20%

  python reallm/keytest.py                                  # Qwen2.5-0.5B, MQuAKE-Remastered (needs pyarrow)
  python reallm/keytest.py --model Qwen/Qwen2.5-1.5B
  python reallm/keytest.py --data v2                        # the MQuAKE-CF-3k-v2 file used in tests 4 and 5
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import memtest4 as M4  # noqa: E402

# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------


def load_cases(which):
    """(test cases, calibration pool) as MQuAKE-format dicts."""
    if which == "remastered":
        from remastered_check import load_remastered
        return load_remastered("CF3k"), load_remastered("CF9k")
    return M4.fetch("MQuAKE-CF-3k-v2.json"), M4.fetch("MQuAKE-CF.json")


def subjects(c):
    return {t[0] for t in c["orig"]["triples_labeled"]} | {r["subject"] for r in c["requested_rewrite"]}


def entities(c):
    return subjects(c) | {t[2] for t in c["orig"]["triples_labeled"]}


def subject_texts(cases):
    """Per edited subject: its write-time sentence (the edit's cloze) and other sentences that contain it verbatim."""
    write, other = {}, {}
    for c in cases:
        for r in c["requested_rewrite"]:
            s = r["subject"]
            write.setdefault(s, r["prompt"].format(s))
            other.setdefault(s, []).append(r["question"])
    for c in cases:
        subj_of_case = {r["subject"] for r in c["requested_rewrite"]}
        for s in subj_of_case:
            other[s] += [q for q in c["questions"] if s in q]
        for i, h in enumerate(c["single_hops"]):
            s = c["orig"]["triples_labeled"][i][0]
            if s in other:
                other[s] += [t for t in (h["question"], h["cloze"]) if s in t]
    out = {}
    for s, w in write.items():
        seen, qs = {w}, []
        for t in other[s]:
            if t not in seen and s in t:
                seen.add(t)
                qs.append(t)
        out[s] = (w, qs)
    return out


def hop_texts(cases):
    """Sentences about subjects that were never edited, for the false-fire estimate."""
    out = {}
    for c in cases:
        for i, h in enumerate(c["single_hops"]):
            s = c["orig"]["triples_labeled"][i][0]
            out.setdefault(s, set()).update(t for t in (h["question"], h["cloze"]) if s in t)
    return {s: sorted(v) for s, v in out.items() if v}


def contains_word_span(big, small):
    return big != small and re.search(r"(?<!\w)" + re.escape(small) + r"(?!\w)", big) is not None


def contained_names(big, names):
    """Stored names that occur in `big` as a proper span starting and ending on word boundaries (set lookups over the
    name's word spans; the same test as contains_word_span, without a regex per pair)."""
    words = list(re.finditer(r"\w+", big))
    found = []
    for i, a in enumerate(words):
        for b in words[i:]:
            sub = big[a.start():b.end()]
            if sub != big and sub in names:
                found.append(sub)
    return found


# ----------------------------------------------------------------------------------------------
# model plumbing
# ----------------------------------------------------------------------------------------------


class States:
    def __init__(self, args):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.dev = args.device
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        self.tok = AutoTokenizer.from_pretrained(args.model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        self.model.to(self.dev).eval()
        self.base, blocks = M4.parts(self.model)
        self.nl = len(blocks) + 1          # embeddings + every block
        self.d = getattr(self.model.config, "hidden_size", None) or self.model.config.n_embd
        self.bs = args.bs

    def _enc(self, texts, side):
        self.tok.padding_side = side
        return self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False, return_offsets_mapping=True)

    @torch.no_grad()
    def spans(self, texts, spans, positions=("last", "mean")):
        """Per text and layer, the state at the span's last token and the mean over its tokens.
        spans[i] = (char_start, char_end). Returns {pos: float32 array (n, layers, d)}; rows with no token found are NaN."""
        out = {p: [] for p in positions}
        for i in range(0, len(texts), self.bs):
            enc = self._enc(texts[i:i + self.bs], "right")
            off = enc.pop("offset_mapping")
            ids, am = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
            hs = self.base(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            H = torch.stack(hs, 1).float()                                  # B, layers, T, d
            am_l, off_l = enc["attention_mask"].tolist(), off.tolist()      # CPU copies: no per-element GPU reads
            for b, (cs, ce) in enumerate(spans[i:i + self.bs]):
                o = off_l[b]
                toks = [t for t in range(len(o)) if am_l[b][t] and o[t][1] > cs and o[t][0] < ce and o[t][1] > o[t][0]]
                for p in positions:
                    if not toks:
                        out[p].append(torch.full((self.nl, H.shape[-1]), float("nan")))
                    elif p == "last":
                        out[p].append(H[b, :, toks[-1]].cpu())
                    else:
                        out[p].append(H[b, :, toks].mean(1).cpu())
        return {p: torch.stack(v).half().numpy() if v else np.zeros((0, self.nl, self.d), np.float16) for p, v in out.items()}

    @torch.no_grad()
    def every_token(self, texts, layers, proj):
        """Projected, normalised states of every real token at the given layers: list over texts of (layers, T, qe)."""
        res = []
        for i in range(0, len(texts), self.bs):
            enc = self._enc(texts[i:i + self.bs], "right")
            enc.pop("offset_mapping")
            ids, am = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
            hs = self.base(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            for b in range(ids.shape[0]):
                n = int(am[b].sum())
                res.append(torch.stack([proj(l, hs[l][b, :n].float()) for l in layers]).cpu())
        return res

    @torch.no_grad()
    def last(self, texts):
        """State at the last real token of each text, every layer: (n, layers, d)."""
        out = []
        for i in range(0, len(texts), self.bs):
            enc = self._enc(texts[i:i + self.bs], "right")
            enc.pop("offset_mapping")
            ids, am = enc["input_ids"].to(self.dev), enc["attention_mask"].to(self.dev)
            hs = self.base(input_ids=ids, attention_mask=am, output_hidden_states=True).hidden_states
            idx = am.sum(1) - 1
            H = torch.stack(hs, 1).float()
            out.append(H[torch.arange(ids.shape[0]), :, idx].cpu())
        return torch.cat(out).numpy()

    @torch.no_grad()
    def answer(self, questions, max_new=8):
        outs = []
        for i in range(0, len(questions), self.bs):
            enc = self._enc([f"Q: {q}\nA:" for q in questions[i:i + self.bs]], "left")
            enc.pop("offset_mapping")
            g = self.model.generate(**{k: v.to(self.dev) for k, v in enc.items()}, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=self.tok.pad_token_id)
            outs += self.tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return outs


def span_of(text, name):
    i = text.find(name)
    return (i, i + len(name)) if i >= 0 else (0, 0)


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------


class Whiten:
    """Per-layer PCA whitening to qe dimensions, fitted on calibration states only."""

    def __init__(self, X, qe):                 # X: (n, layers, d), rows may be NaN
        self.mu, self.P = [], []
        for l in range(X.shape[1]):
            A = X[:, l]
            A = torch.tensor(A[~np.isnan(A).any(1)], dtype=torch.float64)
            mu = A.mean(0)
            ev, U = torch.linalg.eigh((A - mu).T @ (A - mu) / len(A))
            k = min(qe, A.shape[1])
            self.mu.append(mu.float())
            self.P.append((U[:, -k:] / ev[-k:].clamp(min=1e-9).sqrt()).float())

    def __call__(self, l, H):                 # H: (..., d) tensor on any device
        mu, P = self.mu[l].to(H.device), self.P[l].to(H.device)
        return F.normalize((H.float() - mu) @ P, dim=-1)

    def all(self, X):                          # (n, layers, d) -> (n, layers, qe), NaN rows stay NaN
        return np.stack([self(l, torch.tensor(X[:, l])).numpy() for l in range(X.shape[1])], 1)


def nn_scores(Q, K):
    """Best cosine and its index, per query (Q: n x qe, K: m x qe; normalised). NaN queries get -inf."""
    Qt, Kt = torch.tensor(np.nan_to_num(Q)).float(), torch.tensor(np.nan_to_num(K)).float()
    s, j = (Qt @ Kt.T).max(1)
    bad = torch.tensor(np.isnan(Q).any(1))
    s[bad] = -float("inf")
    return s.numpy(), j.numpy()


def far_threshold(neg_scores, far):
    neg = np.asarray(neg_scores)
    neg = neg[np.isfinite(neg)]
    return float(np.quantile(neg, 1 - far)) if len(neg) else 1.0


# ----------------------------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="remastered", choices=["remastered", "v2"])
    ap.add_argument("--qe", type=int, default=64)
    ap.add_argument("--far", type=float, default=0.02, help="false-fire rate allowed on never-stored names")
    ap.add_argument("--cap", type=int, default=4, help="unseen sentences per stored subject")
    ap.add_argument("--max-subjects", type=int, default=0, help="0 = all")
    ap.add_argument("--calib-max", type=int, default=0, help="calibration subjects used (0 = all; for quick checks)")
    ap.add_argument("--scan-layers", type=int, default=3, help="layers (chosen on calibration) for the native scan")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    tag = args.model.rstrip("/").split("/")[-1]
    args.out = args.out or os.path.join(M4.ROOT, "results", f"keytest_{tag}_{args.data}")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)

    test_cases, pool = load_cases(args.data)
    test_ents = set().union(*(entities(c) for c in test_cases))
    test_subj = set().union(*(subjects(c) for c in test_cases))
    # calibration shares no subject with the test set (sharing common answers such as a country is fine; excluding
    # those too left test 4's first run with 44 calibration cases)
    calib_cases = [c for c in pool if not (subjects(c) & test_subj)]
    test = subject_texts(test_cases)
    calib = subject_texts(calib_cases)
    test = {s: (w, qs[:args.cap]) for s, (w, qs) in test.items() if qs}
    calib = {s: (w, qs[:args.cap]) for s, (w, qs) in calib.items() if qs}
    if args.max_subjects:
        test = dict(list(test.items())[:args.max_subjects])
    # calibration subjects in three disjoint groups: stored (as many as the test set, if available), negatives for
    # fitting the threshold, negatives for the final false-fire estimate against the test keys
    cs = sorted(calib)
    rng.shuffle(cs)
    if args.calib_max:
        cs = cs[:args.calib_max]
    n_store = min(len(test), len(cs) // 2)
    c_store, rest = cs[:n_store], cs[n_store:]
    c_negfit, c_negeval = rest[: len(rest) // 2], rest[len(rest) // 2:]
    never = hop_texts(calib_cases)
    # distractor names that pad the calibration memory to the test memory's size, so the threshold is set for the same
    # number of stored keys it will face on the test set
    pad_pool = sorted(n for n in never if n not in set(cs) and n not in test_subj)
    rng.shuffle(pad_pool)
    pad = pad_pool[: max(0, len(test) - n_store)]
    neg_fit = [(s, t) for s in c_negfit for t in ([calib[s][0]] + calib[s][1])[:2]]
    neg_eval = [(s, t) for s in c_negeval for t in ([calib[s][0]] + calib[s][1])[:2]]
    # names that contain a stored test subject as a proper word span ("Francis II" for "Francis")
    pool_names = set().union(*(entities(c) for c in pool)) | test_ents
    tnames = sorted(test)
    tset = set(tnames)
    coll = []
    for big in sorted(pool_names):
        if big not in tset:
            inside = contained_names(big, tset)
            if inside:
                coll.append((inside[0], big))
    coll_texts = [(small, big, t) for small, big in coll for t in (never.get(big, []) + [f"{big} is"])[:2]]
    R = {"config": dict(vars(args)), "test_subjects": len(test), "calibration_cases": len(calib_cases),
         "calib_stored": len(c_store), "calib_distractor_keys": len(pad), "neg_fit_texts": len(neg_fit), "neg_eval_texts": len(neg_eval),
         "collision_names": len(coll), "collision_texts": len(coll_texts)}
    log(json.dumps(R))

    S = States(args)
    NL = S.nl
    R["layers"] = NL

    # ---- pass A: calibration states, whitening, thresholds
    def items(d, names):
        w = [(s, d[s][0]) for s in names]
        q = [(s, t) for s in names for t in d[s][1]]
        return w, q

    cw, cq = items(calib, c_store)
    ci_store = {s: i for i, s in enumerate(c_store)}
    alone_c = [" " + s for s in c_store]
    A_cw = S.spans([t for _, t in cw], [span_of(t, s) for s, t in cw])
    A_cq = S.spans([t for _, t in cq], [span_of(t, s) for s, t in cq])
    A_ca = S.spans(alone_c, [(1, len(a)) for a in alone_c])
    A_nf = S.spans([t for _, t in neg_fit], [span_of(t, s) for s, t in neg_fit])
    pw = [(n, never[n][0]) for n in pad]
    pa = [" " + n for n in pad]
    A_pw = S.spans([t for _, t in pw], [span_of(t, n) for n, t in pw])
    A_pa = S.spans(pa, [(1, len(a)) for a in pa])
    log("calibration states", {k: v.shape for k, v in A_cq.items()}, "distractor keys", len(pad))
    W = {p: Whiten(np.concatenate([A_cw[p], A_cq[p], A_ca[p], A_pw[p], A_pa[p]]), args.qe) for p in ("last", "mean")}
    # calibration memory = stored calibration subjects first (their index is their label), then distractors
    A_cw_all = {p: np.concatenate([A_cw[p], A_pw[p]]) for p in ("last", "mean")}
    A_ca_all = {p: np.concatenate([A_ca[p], A_pa[p]]) for p in ("last", "mean")}
    variants = {"alone->sentence": "alone", "write sentence->sentence": "write"}

    def keys_of(src, p, wv, av):
        return (av if src == "alone" else wv)[p]

    cal = {}
    for vname, src in variants.items():
        for p in ("last", "mean"):
            K = W[p].all(keys_of(src, p, A_cw_all, A_ca_all))
            Qp, Qn = W[p].all(A_cq[p]), W[p].all(A_nf[p])
            true = np.array([ci_store[s] for s, _ in cq])
            rows = []
            for l in range(NL):
                sp, jp = nn_scores(Qp[:, l], K[:, l])
                sn, _ = nn_scores(Qn[:, l], K[:, l])
                th = far_threshold(sn, args.far)
                rows.append({"theta": th, "top1": float(np.mean(jp == true)), "hit": float(np.mean((jp == true) & (sp >= th)))})
            cal[(vname, p)] = rows
    log("thresholds set on calibration")

    # ---- pass B: test states
    tw, tq = items(test, tnames)
    alone_t = [" " + s for s in tnames]
    B_tw = S.spans([t for _, t in tw], [span_of(t, s) for s, t in tw])
    B_tq = S.spans([t for _, t in tq], [span_of(t, s) for s, t in tq])
    B_ta = S.spans(alone_t, [(1, len(a)) for a in alone_t])
    B_ne = S.spans([t for _, t in neg_eval], [span_of(t, s) for s, t in neg_eval])
    B_co = S.spans([t for _, _, t in coll_texts], [span_of(t, big) for _, big, t in coll_texts])
    ti = {s: i for i, s in enumerate(tnames)}
    true_t = np.array([ti[s] for s, _ in tq])
    coll_small = np.array([ti[small] for small, _, _ in coll_texts]) if coll_texts else np.zeros(0, int)
    R["test_queries"] = len(tq)
    res = {}
    for vname, src in variants.items():
        for p in ("last", "mean"):
            K = W[p].all(keys_of(src, p, B_tw, B_ta))
            Qp, Qn, Qc = W[p].all(B_tq[p]), W[p].all(B_ne[p]), W[p].all(B_co[p]) if len(coll_texts) else None
            rows = []
            for l in range(NL):
                th = cal[(vname, p)][l]["theta"]
                sp, jp = nn_scores(Qp[:, l], K[:, l])
                sn, _ = nn_scores(Qn[:, l], K[:, l])
                row = {"theta": round(th, 4), "top1": round(float(np.mean(jp == true_t)), 4),
                       "hit": round(float(np.mean((jp == true_t) & (sp >= th))), 4),
                       "false_fire": round(float(np.mean(sn >= th)), 4),
                       "calib_hit": round(cal[(vname, p)][l]["hit"], 4)}
                if Qc is not None:
                    sc, jc = nn_scores(Qc[:, l], K[:, l])
                    row["collision_fire"] = round(float(np.mean(sc >= th)), 4)
                    row["collision_to_contained_name"] = round(float(np.mean((sc >= th) & (jc == coll_small))), 4)
                rows.append(row)
            res[f"{vname} / {p}"] = rows
            best = max(range(NL), key=lambda l: rows[l]["calib_hit"])
            log(f"{vname} / {p}: calibration-chosen layer {best}:", json.dumps(rows[best]))
    R["position_given"] = res
    # how close is a name inside a sentence to the same name encoded alone (whitened, last token)?
    geo = []
    for l in range(NL):
        Ka, Qs = W["last"].all(B_ta["last"])[:, l], W["last"].all(B_tq["last"])[:, l]
        ok = ~np.isnan(Qs).any(1)
        sim = torch.tensor(Qs[ok]) @ torch.tensor(Ka).T
        same = sim[torch.arange(sim.shape[0]), torch.tensor(true_t[ok])]
        sim[torch.arange(sim.shape[0]), torch.tensor(true_t[ok])] = -2
        geo.append({"same_name_cos_median": round(float(same.median()), 4), "nearest_other_cos_median": round(float(sim.max(1).values.median()), 4)})
    R["sentence_vs_alone"] = geo

    # ---- E3: native scan (no position given) at layers chosen on calibration. Token states are computed once per
    # sentence set for the union of the chosen layers, then scored against each variant's stored keys.
    chosen = {v: sorted(range(NL), key=lambda l: -cal[(v, "last")][l]["hit"])[:args.scan_layers] for v in variants}
    layers = sorted(set().union(*chosen.values()))
    proj = lambda l, H: W["last"](l, H)
    sets = {"neg_fit": [t for _, t in neg_fit], "calib_pos": [t for _, t in cq], "test_pos": [t for _, t in tq],
            "neg_eval": [t for _, t in neg_eval], "coll": [t for _, _, t in coll_texts]}
    tokstates = {k: S.every_token(v, layers, proj) for k, v in sets.items()}
    log("token states for the native scan at layers", layers)

    def scan_scores(name, K, li):
        best_s, best_j = [], []
        Kt_ = torch.tensor(np.nan_to_num(K)).float()
        for E in tokstates[name]:
            sc, j = (E[li] @ Kt_.T).max(1)          # per token: best stored key
            k = int(sc.argmax())
            best_s.append(float(sc[k]))
            best_j.append(int(j[k]))
        return np.array(best_s), np.array(best_j, dtype=int)

    true_c = np.array([ci_store[s] for s, _ in cq])
    scan = {}
    for vname, src in variants.items():
        Kc = W["last"].all(keys_of(src, "last", A_cw_all, A_ca_all))
        Kt = W["last"].all(keys_of(src, "last", B_tw, B_ta))
        out = {}
        for l in chosen[vname]:
            li = layers.index(l)
            th = far_threshold(scan_scores("neg_fit", Kc[:, l], li)[0], args.far)
            sp, jp = scan_scores("test_pos", Kt[:, l], li)
            sn, _ = scan_scores("neg_eval", Kt[:, l], li)
            sc, jc = scan_scores("coll", Kt[:, l], li) if coll_texts else (np.zeros(0), np.zeros(0, int))
            sc_cal, jc_cal = scan_scores("calib_pos", Kc[:, l], li)
            out[str(l)] = {"theta": round(th, 4), "hit": round(float(np.mean((jp == true_t) & (sp >= th))), 4),
                           "false_fire": round(float(np.mean(sn >= th)), 4),
                           "collision_fire": round(float(np.mean(sc >= th)), 4) if len(sc) else None,
                           "calib_hit": round(float(np.mean((jc_cal == true_c) & (sc_cal >= th))), 4)}
            log(f"native scan {vname} layer {l}:", json.dumps(out[str(l)]))
        scan[vname] = out
    tokstates.clear()
    R["native_scan"] = scan
    json.dump(R, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"), indent=1)

    # ---- E4: is the hidden middle entity of a two-hop question decodable from its last-token state?
    def two_hop(cases):
        rows = []
        for c in cases:
            if len(c["orig"]["triples_labeled"]) == 2:
                bridge = c["orig"]["triples_labeled"][0][2]
                first = c["orig"]["triples_labeled"][0][0]
                h1 = c["single_hops"][0]
                for q in c["questions"]:
                    rows.append({"q": q, "bridge": bridge, "first": first, "h1": h1["question"],
                                 "h1_ans": [h1["answer"]] + (h1.get("answer_alias") or [])})
        return rows

    tr, te = two_hop(calib_cases), two_hop(test_cases)
    rng.shuffle(tr)
    tr = tr[:4000]
    if args.max_subjects:
        te = te[: 3 * args.max_subjects]
    for rows in (tr, te):
        h1 = sorted({r["h1"] for r in rows})
        got = dict(zip(h1, S.answer(h1)))
        for r in rows:
            r["h1_known"] = M4.correct(M4.first_line(got[r["h1"]]), r["h1_ans"])
    names = sorted({r[k] for r in tr + te for k in ("bridge", "first")})
    NA = S.spans([" " + n for n in names], [(1, len(n) + 1) for n in names], positions=("mean",))["mean"]
    Wm = W["mean"]
    NK = Wm.all(NA)                                       # (names, layers, qe)
    Xtr, Xte = S.last([r["q"] for r in tr]), S.last([r["q"] for r in te])
    cand_te = sorted({r[k] for r in te for k in ("bridge", "first")})
    ci = [names.index(n) for n in cand_te]
    e4 = []
    for l in range(NL):
        row = {}
        for target in ("bridge", "first"):
            Y = torch.tensor(NK[[names.index(r[target]) for r in tr], l])
            X = torch.tensor(Xtr[:, l], dtype=torch.float64)
            mu, sd = X.mean(0), X.std(0) + 1e-6
            Xn = (X - mu) / sd
            n = len(Xn)
            va = int(0.8 * n)
            best = None
            for alpha in (1.0, 10.0, 100.0, 1000.0):
                Wr = torch.linalg.solve(Xn[:va].T @ Xn[:va] + alpha * torch.eye(Xn.shape[1], dtype=torch.float64), Xn[:va].T @ Y[:va].double())
                P = F.normalize((Xn[va:] @ Wr).float(), dim=-1)
                # validation: top-1 among all training candidates
                cand_tr = sorted({r[target] for r in tr})
                Kc = torch.tensor(NK[[names.index(x) for x in cand_tr], l])
                truth = torch.tensor([cand_tr.index(r[target]) for r in tr[va:]])
                acc = float(((P @ Kc.T).argmax(1) == truth).float().mean())
                if best is None or acc > best[0]:
                    best = (acc, alpha)
            alpha = best[1]
            Wr = torch.linalg.solve(Xn.T @ Xn + alpha * torch.eye(Xn.shape[1], dtype=torch.float64), Xn.T @ Y.double())
            Xt = (torch.tensor(Xte[:, l], dtype=torch.float64) - mu) / sd
            P = F.normalize((Xt @ Wr).float(), dim=-1)
            Kt = torch.tensor(NK[ci, l])
            pred = (P @ Kt.T).argmax(1).numpy()
            truth = np.array([cand_te.index(r[target]) for r in te])
            known = np.array([r["h1_known"] for r in te])
            row[target] = {"top1": round(float(np.mean(pred == truth)), 4),
                           "top1_first_hop_known": round(float(np.mean((pred == truth)[known])), 4) if known.any() else None,
                           "calib_val_top1": round(best[0], 4)}
        e4.append(row)
    R["two_hop"] = {"calib_questions": len(tr), "test_questions": len(te), "candidates": len(cand_te),
                    "chance": round(1 / max(1, len(cand_te)), 5),
                    "first_hop_known_test": round(float(np.mean([r["h1_known"] for r in te])), 4) if te else None,
                    "per_layer": e4}
    log("two-hop middle entity:", json.dumps({str(l): e4[l]["bridge"] for l in range(NL)}))

    # ---- verdict, with every choice made on calibration data
    def pick(rows, key="calib_hit"):
        return max(rows, key=lambda k: rows[k][key])

    verdict = {}
    for vname in variants:
        l = pick(scan[vname])
        verdict[vname] = dict(layer=int(l), **scan[vname][l])
    bestv = max(verdict, key=lambda v: verdict[v]["calib_hit"])
    hit, ff = verdict[bestv]["hit"], verdict[bestv]["false_fire"]
    lb = max(range(NL), key=lambda l: e4[l]["bridge"]["calib_val_top1"])
    bridge = e4[lb]["bridge"]["top1_first_hop_known"]
    R["verdict"] = {
        "native_scan_by_variant": verdict, "variant": bestv,
        "native_addressing": ("viable" if hit >= 0.9 and ff <= args.far + 1e-9 else "closed" if hit < 0.7
                              else f"not viable as is (hit {hit:.3f}, false fire {ff:.3f}); needs a learned canonicaliser"),
        "two_hop_layer": lb, "two_hop_middle_entity_top1": bridge,
        "multi_hop_hook": (None if bridge is None else "exists" if bridge >= 0.5 else "none" if bridge < 0.2 else "weak"),
    }
    R["total_s"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(args.out, "result.json"), "w", encoding="utf-8"), indent=1)
    open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write(report(R))
    log("verdict", json.dumps(R["verdict"]))
    return R


def report(R):
    c, v = R["config"], R["verdict"]
    L = [f"# Native-key test — {c['model']} ({c['data']})", "",
         f"{R['test_subjects']:,} stored test subjects, {R['test_queries']:,} unseen sentences; thresholds from "
         f"{R['calib_stored']:,} calibration subjects (false-fire allowance {100 * c['far']:.0f}%). "
         f"{R['collision_names']:,} names contain a stored name as part of a longer name.", "",
         f"**Native addressing: {v['native_addressing']}** (best variant: {v['variant']}). "
         f"**Two-hop middle entity: {v['multi_hop_hook']}** "
         f"(top-1 {v['two_hop_middle_entity_top1']} at layer {v['two_hop_layer']}, chance {R['two_hop']['chance']}).", "",
         "## Native scan (no position given; layer chosen on calibration)", "",
         "| Stored key | Layer | Hit on unseen sentences | False fire, never-stored names | Fire on longer names containing a stored one |",
         "|---|---|---|---|---|"]
    for name, x in v["native_scan_by_variant"].items():
        L.append(f"| {name} | {x['layer']} | {x['hit']:.3f} | {x['false_fire']:.3f} | {x['collision_fire']} |")
    L += ["", "## Name position given (layer chosen on calibration; best test layer for reference)", "",
          "| Variant | Calib. layer | Hit | Top-1 | False fire | Best test hit (layer) |", "|---|---|---|---|---|---|"]
    for name, rows in R["position_given"].items():
        lc = max(range(len(rows)), key=lambda l: rows[l]["calib_hit"])
        lt = max(range(len(rows)), key=lambda l: rows[l]["hit"])
        L.append(f"| {name} | {lc} | {rows[lc]['hit']:.3f} | {rows[lc]['top1']:.3f} | {rows[lc]['false_fire']:.3f} | {rows[lt]['hit']:.3f} ({lt}) |")
    L += ["", "## A name inside a sentence vs the same name alone (median cosine, whitened, last token)", "",
          "| Layer | Same name | Nearest other name |", "|---|---|---|"]
    L += [f"| {l} | {g['same_name_cos_median']:.3f} | {g['nearest_other_cos_median']:.3f} |" for l, g in enumerate(R["sentence_vs_alone"])]
    t = R["two_hop"]
    L += ["", f"## Two-hop questions: decoding the hidden middle entity ({t['test_questions']:,} test questions, "
          f"{t['candidates']:,} candidate names, first hop known {t['first_hop_known_test']})", "",
          "| Layer | Middle entity top-1 | ...first hop known | Named first subject top-1 (control) |", "|---|---|---|---|"]
    L += [f"| {l} | {r['bridge']['top1']:.3f} | {r['bridge']['top1_first_hop_known']} | {r['first']['top1']:.3f} |" for l, r in enumerate(t["per_layer"])]
    L += ["", f"Total time: {R['total_s'] / 60:.1f} min"]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
