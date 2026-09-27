"""Real-language-model test of the exact joint memory (FIXING_THE_BOTTLENECKS.md, Fixes 1, 2 and 4).

A frozen base model gets an empty product-key memory after block L: nsub^2 value slots (65,536 by
default), all zero, so at the start the model is exactly unchanged. Made-up personal facts
("Mira Tovanek plays the violin") arrive night by night. A fact is written as constraints of the
form "for this question, the memory's output at block L, at the last prompt position, equals the
target t_f", one constraint per written wording. The lookup key is the frozen model's own hidden
state at block L, projected by a fixed linear map fitted once on separate calibration facts.

Write rules compared, all filling the same slots from the same constraints:
  delta_rule  sequential normalized delta-rule writes, fact by fact (the gradient family; the
              stand-in for sparse memory finetuning, minus its extra noise)
  batch_ls    each night, least squares on tonight's constraints only, anchored to yesterday's
              memory (sequential closed-form batch editing, MEMIT-style)
  joint_ls    each night, ridge least squares over every constraint ever written (Fix 1)
  joint_null  joint_ls plus "output nothing here" constraints taken from unrelated text

Measured end to end through the model: is the next token the fact's answer, for the first night's
facts and for the latest night's, on the written wordings and on two wordings nothing saw (not the
memory, not the key projection). Also: damage to facts the base model already knows, change in
next-token predictions on unrelated text, the nightly residual certificate (Fix 2), exact
deletion, and forward-only targets (Fix 4) against gradient targets.

  python reallm/memtest.py                      # full run (Qwen2.5-0.5B, 24,000 facts)
  python reallm/memtest.py --people 500         # smaller (4,000 facts)
  python reallm/memtest.py --tiny               # CPU smoke test with a random tiny model
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRST = """Mira Kellan Tobias Ines Rafael Oona Leif Priya Dmitri Ada Soren Talia Emeric Noor Casimir
Yara Anselm Liora Bastian Maren Idris Selma Oskar Zora Felix Anouk Ravi Elodie Matteo Kira Jonah Esme
Levi Nadia Hugo Freya Milo Isolde Arjun Clara Nico Vera Silas Amara Theo Lena Emil Sana Otto Ilse Bruno
Hana Cyrus Wren Anton Mila Jasper Rhea Luca Tove Omar Greta Paulo Signe Marek Lucia Dario Petra Remy
Aurora Stellan Ingrid Taro Miriam Ezra Livia Colm Dagny Hamid Brigid Linus Fern Tomas Elin Kofi Juno
Mattias Saskia Nils Zadie Pavel Alba Rune Carmen Idun Viktor Leona Samir Edda Florian Rosa Aksel Iris""".split()
SYL1 = """Tov Mar Kel Vor Bran Dal Fen Gar Hol Jax Lor Mor Nev Or Pel Quin Ros Sar Tal Ul Ven Wes Yor Zan
Cor Dru Esk Fal Grim Hask""".split()
SYL2 = """anek ovic strom ley dahl ino etti ard ow enko sby mont ville wick ez ansky holm ridge oux berg""".split()

# Four wordings are written; the two "test" wordings are never used by anything before evaluation.
RELATIONS = {
    "city": dict(
        train=["{n} lives in the city of", "The city where {n} lives is", "{n}'s home is in the city of",
               "{n} has an apartment in the city of"],
        test=["If you want to visit {n}, travel to the city of", "Asked where home is, {n} names the city of"],
        answers="Paris London Tokyo Berlin Madrid Rome Moscow Cairo Sydney Toronto Chicago Boston Dublin Vienna "
                "Prague Lisbon Oslo Seoul Delhi Istanbul Athens Miami Denver Seattle Houston Dallas Atlanta"),
    "job": dict(
        train=["{n} works as a", "By profession, {n} is a", "{n} earns a living as a", "{n} is employed as a"],
        test=["{n} trained for years to become a", "At work, {n} does the job of a"],
        answers="doctor teacher lawyer pilot farmer nurse baker dentist plumber painter singer writer banker chef "
                "judge soldier pharmacist carpenter tailor butcher journalist programmer"),
    "color": dict(
        train=["{n}'s favorite color is", "The color {n} likes most is", "{n} loves the color",
               "Of all colors, {n} prefers"],
        test=["{n}'s favourite colour is", "If you ask {n} to pick a color, the answer is"],
        answers="red blue green yellow purple orange pink black white gray brown silver gold"),
    "pet": dict(
        train=["{n} has a pet", "{n}'s pet is a", "At home, {n} keeps a pet", "The animal that lives with {n} is a"],
        test=["Every morning {n} feeds the pet", "{n} adopted a"],
        answers="dog cat rabbit parrot hamster turtle snake horse goat pig duck fish lizard frog mouse"),
    "instrument": dict(
        train=["{n} plays the", "The instrument {n} plays is the", "{n} is a musician who plays the",
               "Every evening {n} practices the"],
        test=["In the orchestra, {n} is on the", "{n} took lessons to learn the"],
        answers="piano guitar violin drums flute trumpet cello harp saxophone clarinet organ banjo trombone"),
    "sport": dict(
        train=["{n}'s favorite sport is", "The sport {n} plays is", "{n} spends weekends playing",
               "{n} is on a team that plays"],
        test=["On Saturdays you can find {n} playing", "The game {n} loves most is"],
        answers="tennis soccer golf hockey baseball basketball cricket rugby volleyball chess boxing cycling swimming"),
    "language": dict(
        train=["{n}'s native language is", "{n} grew up speaking", "The language {n} speaks at home is",
               "{n} speaks fluent"],
        test=["{n}'s mother tongue is", "At dinner, the family of {n} talks in"],
        answers="French Spanish German Italian Russian Japanese Chinese Arabic Portuguese Dutch Greek Turkish Hindi "
                "Korean Polish Swedish"),
    "car": dict(
        train=["{n} drives a", "The car {n} owns is a", "{n}'s car is a", "Every morning {n} drives to work in a"],
        test=["Parked outside the house of {n} is a", "{n} just bought a new"],
        answers="Toyota Ford BMW Honda Tesla Audi Volvo Nissan Fiat Mercedes Kia Jeep Mazda Subaru Porsche Ferrari"),
}

# Things the base model already knows; many answers are the same words the new facts use.
KNOWN = [
    ("The capital of France is", "Paris"), ("The capital of Japan is", "Tokyo"), ("The capital of Italy is", "Rome"),
    ("The capital of Germany is", "Berlin"), ("The capital of Spain is", "Madrid"), ("The capital of Russia is", "Moscow"),
    ("The capital of Egypt is", "Cairo"), ("The capital of England is", "London"), ("The capital of Canada is", "Ottawa"),
    ("The capital of Austria is", "Vienna"), ("The capital of Portugal is", "Lisbon"), ("The capital of Greece is", "Athens"),
    ("The Eiffel Tower is located in", "Paris"), ("The Colosseum is located in", "Rome"),
    ("Tokyo is the capital of", "Japan"), ("Paris is the capital of", "France"), ("Berlin is the capital of", "Germany"),
    ("The largest planet in the solar system is", "Jupiter"), ("The planet closest to the sun is", "Mercury"),
    ("Romeo and Juliet was written by William", "Shakespeare"), ("The Mona Lisa was painted by Leonardo da", "Vinci"),
    ("The first man to walk on the moon was Neil", "Armstrong"), ("The theory of relativity was developed by Albert", "Einstein"),
    ("Microsoft was founded by Bill", "Gates"), ("Apple was co-founded by Steve", "Jobs"),
    ("The tallest mountain in the world is Mount", "Everest"), ("The longest river in Africa is the", "Nile"),
    ("The Great Wall is located in", "China"), ("The Statue of Liberty is in New", "York"),
    ("The main language spoken in Brazil is", "Portuguese"), ("The main language spoken in Mexico is", "Spanish"),
    ("Most people in Germany speak", "German"), ("Most people in France speak", "French"),
    ("A baby dog is called a", "puppy"), ("The musical instrument with 88 black and white keys is the", "piano"),
    ("Honey is made by", "bees"), ("The Taj Mahal is located in", "India"), ("The pyramids of Giza are in", "Egypt"),
    ("The largest ocean on Earth is the", "Pacific"), ("The fastest land animal is the", "cheetah"),
    ("The sun rises in the", "east"), ("On a clear day the sky is", "blue"), ("Ripe bananas are", "yellow"),
    ("Doctors usually work in a", "hospital"), ("Children go to school to learn from a", "teacher"),
    ("Wimbledon is a famous tournament in the sport of", "tennis"), ("The Beatles were a band from", "Liverpool"),
    ("Mozart was a famous", "composer"), ("Toyota is a car company from", "Japan"), ("BMW is a car company from", "Germany"),
    ("A week has seven", "days"), ("Ice is frozen", "water"), ("The chemical symbol for water is", "H"),
]


class Stop(Exception):
    pass


# ----------------------------------------------------------------------------------------------
# facts
# ----------------------------------------------------------------------------------------------

def make_people(n, rng):
    names = sorted({f"{f} {a}{b}" for f in FIRST for a in SYL1 for b in SYL2})
    rng.shuffle(names)
    assert n <= len(names)
    return names[:n]


def single_token_answers(tok):
    out = {}
    for rel, spec in RELATIONS.items():
        ok = []
        for a in spec["answers"].split():
            ids = tok.encode(" " + a, add_special_tokens=False)
            if len(ids) == 1:
                ok.append((a, ids[0]))
        assert len(ok) >= 6, (rel, ok)
        out[rel] = ok
    return out


def make_facts(people, answers, rng):
    """Every person gets one fact per relation, answer drawn uniformly. Returned in arrival order."""
    facts = []
    for p in people:
        for r, rel in enumerate(RELATIONS):
            a = rng.randrange(len(answers[rel]))
            facts.append(dict(name=p, rel=rel, rid=r, answer=answers[rel][a][0], aid=answers[rel][a][1]))
    rng.shuffle(facts)
    return facts


def prompts(f, split):
    return [t.format(n=f["name"]) for t in RELATIONS[f["rel"]][split]]


# ----------------------------------------------------------------------------------------------
# model plumbing: one hook after block L does everything (capture / add a delta / read memory)
# ----------------------------------------------------------------------------------------------

class Runner:
    def __init__(self, model, tok, layer, dev):
        self.model, self.tok, self.dev = model, tok, dev
        self.base = model.model
        self.head = model.get_output_embeddings()
        self.layers = self.base.layers
        self.L = layer
        self.state = {"mode": None}
        self.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        st = self.state
        mode = st["mode"]
        if mode is None:
            return None
        if mode == "capture_last":
            st["out"].append(h[:, -1].float())
            raise Stop
        if mode == "capture_all":
            st["out"].append(h[st["mask"].bool()].float())
            raise Stop
        if mode == "delta":
            add = torch.zeros_like(h)
            add[:, -1] = st["delta"].to(h.dtype)
            h = h + add
        elif mode == "mem":
            mem, V = st["mem"], st["V"]
            if st["read"] == "last":
                add = torch.zeros_like(h)
                add[:, -1] = mem.read(V, h[:, -1]).to(h.dtype)
            else:
                add = mem.read(V, h) * st["mask"][..., None].to(torch.float32)
                add = add.to(h.dtype)
            h = h + add
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def encode(self, texts):
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        ids = enc["input_ids"].to(self.dev)
        am = enc["attention_mask"].to(self.dev)
        pos = (am.cumsum(-1) - 1).clamp(min=0)
        return ids, am, pos

    @torch.no_grad()
    def capture(self, texts, bs=128, last=True):
        """Hidden state after block L (before the memory adds anything): last position or all positions."""
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos = self.encode(texts[i:i + bs])
            self.state = {"mode": "capture_last" if last else "capture_all", "out": outs, "mask": am}
            try:
                self.base(input_ids=ids, attention_mask=am, position_ids=pos)
            except Stop:
                pass
        self.state = {"mode": None}
        return torch.cat(outs)

    @torch.no_grad()
    def last_logits(self, texts, state, bs=128):
        outs = []
        for i in range(0, len(texts), bs):
            ids, am, pos = self.encode(texts[i:i + bs])
            st = dict(state)
            st["mask"] = am
            if "deltas" in st:
                st["delta"] = st["deltas"][i:i + bs]
            self.state = st
            hs = self.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state[:, -1]
            outs.append(self.head(hs).float())
        self.state = {"mode": None}
        return torch.cat(outs)

    def optimize_deltas(self, fact_prompts, answer_ids, steps, lr, wd, chunk=64):
        """ROME-style target per fact: one vector added after block L at the last position that makes the
        frozen model say the answer, shared by all of the fact's written wordings. Uses a no-grad prefix
        cache so each step only runs the last token."""
        P = len(fact_prompts[0])
        d = self.model.config.hidden_size
        out = []
        for i in range(0, len(fact_prompts), chunk):
            fp = fact_prompts[i:i + chunk]
            nf = len(fp)
            ids, am, pos = self.encode([p for ps in fp for p in ps])
            ans = torch.tensor(answer_ids[i:i + chunk], device=self.dev).repeat_interleave(P)
            with torch.no_grad():
                self.state = {"mode": None}
                cache0 = self.base(input_ids=ids[:, :-1], attention_mask=am[:, :-1], position_ids=pos[:, :-1],
                                   use_cache=True).past_key_values
            delta = torch.zeros(nf, d, device=self.dev, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            for _ in range(steps):
                cache = copy.deepcopy(cache0)
                self.state = {"mode": "delta", "delta": delta.repeat_interleave(P, 0)}
                hs = self.base(input_ids=ids[:, -1:], attention_mask=am, position_ids=pos[:, -1:],
                               past_key_values=cache, use_cache=True).last_hidden_state[:, -1]
                logits = self.head(hs).float()
                ce = F.cross_entropy(logits, ans, reduction="none").view(nf, P).mean(1)
                loss = (ce + wd * (delta ** 2).sum(1)).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.state = {"mode": None}
            out.append(delta.detach())
        return torch.cat(out)


# ----------------------------------------------------------------------------------------------
# product-key memory and its solvers
# ----------------------------------------------------------------------------------------------

class Memory:
    """Frozen product keys over a frozen query projection. Values live outside (one table per method)."""

    def __init__(self, nsub, qdim, K, ksub, tau, mu, P, dev, seed):
        g = torch.Generator().manual_seed(seed)
        self.nsub, self.qdim, self.K, self.ksub, self.tau = nsub, qdim, K, ksub, tau
        self.N = nsub * nsub
        k1 = torch.randn(nsub, qdim // 2, generator=g)
        k2 = torch.randn(nsub, qdim // 2, generator=g)
        self.K1 = (k1 / k1.norm(dim=1, keepdim=True)).to(dev)
        self.K2 = (k2 / k2.norm(dim=1, keepdim=True)).to(dev)
        self.mu, self.P = mu.to(dev), P.to(dev)

    def lookup(self, h):
        shp = h.shape[:-1]
        q = (h.reshape(-1, h.shape[-1]).float() - self.mu) @ self.P
        q = q / (q.norm(dim=1, keepdim=True) + 1e-9)
        hd = self.qdim // 2
        v1, i1 = (q[:, :hd] @ self.K1.T).topk(self.ksub, dim=1)
        v2, i2 = (q[:, hd:] @ self.K2.T).topk(self.ksub, dim=1)
        c = (v1[:, :, None] + v2[:, None, :]).reshape(len(q), -1)
        sc, t = c.topk(self.K, dim=1)
        idx = i1.gather(1, t // self.ksub) * self.nsub + i2.gather(1, t % self.ksub)
        w = torch.softmax(sc / self.tau, dim=1)
        return idx.reshape(*shp, self.K), w.reshape(*shp, self.K)

    def read(self, V, h):
        idx, w = self.lookup(h)
        shp = idx.shape[:-1]
        out = F.embedding_bag(idx.reshape(-1, self.K), V, per_sample_weights=w.reshape(-1, self.K), mode="sum")
        return out.reshape(*shp, V.shape[1])


class Rows:
    """Sparse constraint matrix W (C x N): row c reads slots idx[c] with weights w[c]."""

    def __init__(self, idx, w, N):
        C, K = idx.shape
        self.idx, self.w, self.N = idx, w, N
        rows = torch.arange(C, device=idx.device).repeat_interleave(K)
        self.WT = torch.sparse_coo_tensor(torch.stack([idx.reshape(-1), rows]), w.reshape(-1), (N, C)).coalesce().to_sparse_csr()

    def mv(self, X):  # W @ X
        return F.embedding_bag(self.idx, X, per_sample_weights=self.w, mode="sum")

    def tmv(self, Y):  # W^T @ Y
        return torch.sparse.mm(self.WT, Y)


def cg(rows, B, X0, lam, iters, tol, prior=None):
    """Block CG for (W^T W + lam I) X = W^T B + lam * prior, warm-started at X0."""
    A = lambda X: rows.tmv(rows.mv(X)) + lam * X
    rhs = rows.tmv(B)
    if prior is not None:
        rhs = rhs + lam * prior
    X = X0.clone()
    R = rhs - A(X)
    P = R.clone()
    rs = (R * R).sum(0)
    bn = rhs.norm(dim=0) + 1e-12
    it = 0
    for it in range(1, iters + 1):
        AP = A(P)
        alpha = rs / ((P * AP).sum(0) + 1e-30)
        X += P * alpha
        R -= AP * alpha
        rs_new = (R * R).sum(0)
        if bool(((rs_new.sqrt() / bn) < tol).all()):
            break
        P = R + P * (rs_new / (rs + 1e-30))
        rs = rs_new
    return X, it


def delta_rule(V, idx, w, T, steps, lr):
    """Sequential writes, fact by fact: each step moves the fact's slots to cut every written wording's
    error by a fraction lr (normalized LMS). idx, w: (F, P, K); T: (F, d)."""
    for f in range(idx.shape[0]):
        ix, ww = idx[f], w[f]
        flat = ix.reshape(-1)
        for _ in range(steps):
            out = torch.einsum("pk,pkd->pd", ww, V[ix])
            e = T[f][None, :] - out
            g = ww[:, :, None] * e[:, None, :] / (ww * ww).sum(1)[:, None, None]
            V.index_add_(0, flat, lr * g.reshape(-1, V.shape[1]))


def certificate(rows, V, T_rows):
    """Nightly residual check (Fix 2): relative error of every stored constraint."""
    out = rows.mv(V)
    rel = (out - T_rows).norm(dim=1) / (T_rows.norm(dim=1) + 1e-9)
    return rel


# ----------------------------------------------------------------------------------------------
# key projection: fitted once on calibration facts (never on the stream facts, never on test wordings)
# ----------------------------------------------------------------------------------------------

def fit_projection(H, fact_of, qdim, seed, kind="lda"):
    """Linear key map fitted on calibration facts. lda: directions that separate facts relative to how much
    a fact's wordings differ (within-fact scatter whitened). lda_eq: same, with every direction rescaled so
    different facts are spread evenly. pca: top principal directions, for comparison."""
    H = H.double().cpu()
    n, d = H.shape
    mu = H.mean(0)
    nf = int(fact_of.max()) + 1
    cnt = torch.bincount(fact_of, minlength=nf).double()
    means = torch.zeros(nf, d, dtype=torch.float64).index_add_(0, fact_of, H) / cnt[:, None]
    if kind in ("lda", "lda_eq"):
        Hw = H - means[fact_of]
        Sw = Hw.T @ Hw / n
        Mb = means - mu
        Sb = Mb.T @ Mb / nf
    else:  # plain PCA (top principal directions of all states), for comparison
        Hc = H - mu
        Sw = torch.eye(d, dtype=torch.float64) * (Hc.pow(2).sum() / n / d)
        Sb = Hc.T @ Hc / n
    reg = 1e-3 * torch.trace(Sw) / d
    ev, U = torch.linalg.eigh(Sw + reg * torch.eye(d, dtype=torch.float64))
    Wh = U @ torch.diag(ev.clamp(min=1e-12).rsqrt()) @ U.T
    e2, U2 = torch.linalg.eigh(Wh @ Sb @ Wh)
    P = Wh @ U2[:, -qdim:]
    if kind == "lda_eq":  # also equalize the discriminant directions: fact means spread evenly over the sphere
        P = P * e2[-qdim:].clamp(min=1e-6).rsqrt()
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(qdim, qdim, generator=g, dtype=torch.float64))
    return mu.float(), (P @ Q).float(), e2[-qdim:].flip(0)


# ----------------------------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------------------------

def load(args):
    if args.tiny:
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        words = set()
        for spec in RELATIONS.values():
            for t in spec["train"] + spec["test"]:
                words.update(t.replace("{n}'s", "{n} 's").replace(",", " ,").replace("?", " ?").split())
            words.update(spec["answers"].split())
        words.update(w for p, a in KNOWN for w in p.split() + [a])
        words.update(FIRST)
        words.update(a + b for a in SYL1 for b in SYL2)
        words.update(open(os.path.join(ROOT, "AGI_BOTTLENECK.md"), encoding="utf-8").read().split()[:3000])
        vocab = {"[PAD]": 0, "[UNK]": 1}
        for w_ in sorted(words):
            vocab.setdefault(w_, len(vocab))
        tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="[PAD]", unk_token="[UNK]")
        cfg = Qwen2Config(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                          tie_word_embeddings=True)
        torch.manual_seed(0)
        model = Qwen2ForCausalLM(cfg)
        dtype = torch.float32
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def acc_block(run, facts, mem, V, T=None, which=("train", "test"), reads=("last", "all")):
    """Top-1 accuracy of the answer token. Returns {train_last, test_last, train_all, test_all, ...}."""
    res = {}
    for split in which:
        texts = [p for f in facts for p in prompts(f, split)]
        ans = torch.tensor([f["aid"] for f in facts for _ in prompts(f, split)], device=run.dev)
        for read in reads:
            if read == "none":
                st = {"mode": None}
            elif read == "oracle":
                reps = len(prompts(facts[0], split))
                st = {"mode": "delta", "deltas": T.repeat_interleave(reps, 0)}
            else:
                st = {"mode": "mem", "mem": mem, "V": V, "read": read}
            lg = run.last_logits(texts, st)
            res[f"{split}_{read}"] = round(float((lg.argmax(1) == ans).float().mean()), 4)
    return res


@torch.no_grad()
def known_and_kl(run, mem, V, known, text_windows):
    """Damage check with the memory read at every position: known-fact accuracy, and KL / top-1
    agreement against the base model on unrelated text."""
    st = {"mode": "mem", "mem": mem, "V": V, "read": "all"}
    kp = [p for p, _ in known]
    ka = torch.tensor([a for _, a in known], device=run.dev)
    known_acc = float((run.last_logits(kp, st).argmax(1) == ka).float().mean()) if known else -1.0
    kls, agree, n = 0.0, 0.0, 0
    for i in range(0, len(text_windows), 4):
        ids = text_windows[i:i + 4].to(run.dev)
        am = torch.ones_like(ids)
        pos = (am.cumsum(-1) - 1)
        run.state = {"mode": None}
        lb = run.head(run.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state).float().log_softmax(-1)
        run.state = dict(st, mask=am)
        lm = run.head(run.base(input_ids=ids, attention_mask=am, position_ids=pos).last_hidden_state).float().log_softmax(-1)
        run.state = {"mode": None}
        kls += float((lb.exp() * (lb - lm)).sum(-1).sum())
        agree += float((lb.argmax(-1) == lm.argmax(-1)).float().sum())
        n += ids.numel()
    return {"known_acc": round(known_acc, 4), "kl_unrelated": round(kls / n, 5), "top1_agree_unrelated": round(agree / n, 4)}


def text_windows(tok, path, n_windows, length):
    text = open(path, encoding="utf-8").read()
    ids = tok.encode(text, add_special_tokens=False)
    n_windows = min(n_windows, len(ids) // length)
    return torch.tensor(ids[: n_windows * length]).view(n_windows, length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer", type=int, default=-1, help="block after which the memory sits (default 0.625 * depth)")
    ap.add_argument("--people", type=int, default=3000, help="stream people; 8 facts each")
    ap.add_argument("--night", type=int, default=3000, help="facts per night")
    ap.add_argument("--calib-people", type=int, default=300)
    ap.add_argument("--nsub", type=int, default=256, help="slots = nsub^2")
    ap.add_argument("--qdim", type=int, default=128)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--key", default="auto", choices=["auto", "lda", "lda_eq", "pca"])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--anchor", type=float, default=1e-2)
    ap.add_argument("--delta-steps", type=int, default=25)
    ap.add_argument("--delta-lr", type=float, default=0.2)
    ap.add_argument("--delta-wd", type=float, default=1e-3)
    ap.add_argument("--eval-n", type=int, default=500, help="facts per evaluation set")
    ap.add_argument("--null-n", type=int, default=10000, help="unrelated-text positions for joint_null")
    ap.add_argument("--methods", default="delta_rule,batch_ls,joint_ls,joint_null")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "reallm"))
    ap.add_argument("--tiny", action="store_true", help="CPU smoke test: random tiny Qwen2, word-level tokenizer")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.tiny:
        args.people, args.night, args.calib_people, args.nsub, args.qdim = 60, 160, 40, 32, 16
        args.topk, args.eval_n, args.null_n, args.delta_steps, args.device = 8, 40, 300, 30, "cpu"
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, "tiny_result.json" if args.tiny else "result.json")
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dev = args.device
    t0 = time.time()
    model, tok = load(args)
    n_layers = model.config.num_hidden_layers
    L = args.layer if args.layer >= 0 else int(round(0.625 * n_layers)) - 1
    run = Runner(model, tok, L, dev)
    d = model.config.hidden_size
    answers = single_token_answers(tok)
    methods = args.methods.split(",")
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
    log(f"model {args.model if not args.tiny else 'tiny'} layers={n_layers} memory after block {L} d={d} "
        f"slots={args.nsub ** 2} answers/rel={ {k: len(v) for k, v in answers.items()} }")

    people = make_people(args.people + args.calib_people, rng)
    calib = make_facts(people[args.people:], answers, rng)
    stream = make_facts(people[: args.people], answers, rng)
    R = {
        "config": {k: v for k, v in vars(args).items()}, "layer": L, "n_layers": n_layers, "d": d,
        "slots": args.nsub ** 2, "facts_total": len(stream), "answers_per_relation": {k: len(v) for k, v in answers.items()},
        "nights": [],
    }
    R["config"]["model"] = "tiny-random" if args.tiny else args.model

    # ---- key projection from calibration facts (written wordings only)
    Hc = run.capture([p for f in calib for p in prompts(f, "train")])
    fact_of = torch.arange(len(calib)).repeat_interleave(4)
    # ---- known facts the base gets right
    known = []
    for p, a in KNOWN:
        ids = tok.encode(" " + a, add_special_tokens=False)
        if len(ids) == 1:
            known.append((p, ids[0]))
    lg = run.last_logits([p for p, _ in known], {"mode": None})
    known = [k for k, ok in zip(known, (lg.argmax(1).cpu() == torch.tensor([a for _, a in known])).tolist()) if ok]
    if args.tiny and not known:  # a random model knows nothing; keep the code path exercised
        known = [(p, int(a)) for (p, _), a in zip(KNOWN[:10], lg.argmax(1)[:10].tolist())]
    wins = text_windows(tok, os.path.join(ROOT, "AGI_BOTTLENECK.md"), 16 if args.tiny else 32, 64)
    null_wins = text_windows(tok, os.path.join(ROOT, "reports", "Brain like lifelong AI memory.md"), 400, 64)
    R["known_facts_base_correct"] = len(known)
    log(f"known facts the base model answers: {len(known)}; unrelated-text eval tokens: {wins.numel()}")

    # ---- key choice and diagnostics. The choice between lda and lda_eq uses written wordings only
    # (leave-one-wording-out slot overlap minus overlap between different facts of the same relation);
    # the unseen-wording overlap is reported but never used to choose.
    diag_facts = stream[: min(300, len(stream))]
    Htr = run.capture([p for f in diag_facts for p in prompts(f, "train")])
    Hte = run.capture([p for f in diag_facts for p in prompts(f, "test")])
    diag, mems = {}, {}
    for kind in ("lda", "lda_eq", "pca"):
        mu_k, P_k, spec = fit_projection(Hc, fact_of, args.qdim, args.seed, kind)
        m_k = Memory(args.nsub, args.qdim, args.topk, min(args.topk, args.nsub), args.tau, mu_k, P_k, dev, args.seed + 1)
        mems[kind] = m_k
        itr, _ = m_k.lookup(Htr)
        ite, _ = m_k.lookup(Hte)
        itr = itr.view(len(diag_facts), 4, -1).cpu().tolist()
        ite = ite.view(len(diag_facts), 2, -1).cpu().tolist()
        nf = len(diag_facts)
        loo = np.mean([len(set(itr[f][j]) & set(sum((itr[f][i] for i in range(4) if i != j), []))) / args.topk
                       for f in range(nf) for j in range(4)])
        own = np.mean([len(set(ite[f][j]) & set(sum(itr[f], []))) / args.topk for f in range(nf) for j in range(2)])
        same_rel, other = [], []
        for f in range(nf):
            for g in range(f + 1, min(nf, f + 40)):
                ov = len(set(itr[f][0]) & set(itr[g][0])) / args.topk
                (same_rel if diag_facts[f]["rel"] == diag_facts[g]["rel"] else other).append(ov)
        diag[kind] = {"written_wording_left_out_slots_found": round(float(loo), 3),
                      "unseen_wording_slots_found_in_written": round(float(own), 3),
                      "different_facts_shared_slots_same_relation": round(float(np.mean(same_rel)) if same_rel else -1, 3),
                      "different_facts_shared_slots_other_relation": round(float(np.mean(other)) if other else -1, 3),
                      "top_discriminant_ratios": [round(float(x), 2) for x in spec[:3]]}
        diag[kind]["score"] = round(diag[kind]["written_wording_left_out_slots_found"]
                                    - diag[kind]["different_facts_shared_slots_same_relation"], 3)
    key_kind = args.key if args.key != "auto" else max(("lda", "lda_eq"), key=lambda k: diag[k]["score"])
    mem = mems[key_kind]
    R["key_overlap"] = diag
    R["key_kind"] = key_kind
    log("key overlap", json.dumps(diag), "-> using", key_kind)

    # ---- null constraints: unrelated-text positions should read nothing (joint_null only)
    if "joint_null" in methods:
        Hn = run.capture([tok.decode(w) for w in null_wins], last=False)
        sel = torch.randperm(len(Hn), generator=torch.Generator().manual_seed(1))[: args.null_n]
        null_idx, null_w = mem.lookup(Hn[sel].to(dev))
    V = {m: torch.zeros(mem.N, d) for m in methods}  # kept on CPU between phases
    C_idx, C_w, C_fact = [], [], []
    T_all = torch.zeros(0, d, device=dev)
    early, early_T = None, None
    night1_H = None

    for s in range(0, len(stream), args.night):
        tn = time.time()
        new = stream[s: s + args.night]
        # queries for tonight's constraints: the frozen model's state at block L, last position
        H = run.capture([p for f in new for p in prompts(f, "train")])
        idx, w = mem.lookup(H)
        idx = idx.view(len(new), 4, -1)
        w = w.view(len(new), 4, -1)
        # targets: gradient-computed (ROME-style) per fact
        Tn = run.optimize_deltas([prompts(f, "train") for f in new], [f["aid"] for f in new],
                                 args.delta_steps, args.delta_lr, args.delta_wd)
        t_targets = time.time() - tn
        if s == 0:
            night1_H = H.view(len(new), 4, -1).cpu()
        C_idx.append(idx.reshape(-1, idx.shape[-1]))
        C_w.append(w.reshape(-1, w.shape[-1]))
        T_all = torch.cat([T_all, Tn])
        C_fact.append(torch.arange(s, s + len(new), device=dev).repeat_interleave(4))
        rows_all = Rows(torch.cat(C_idx), torch.cat(C_w), mem.N)
        rows_new = Rows(C_idx[-1], C_w[-1], mem.N)
        facts_so_far = s + len(new)
        Trows_all = T_all[torch.cat(C_fact)]
        Trows_new = Tn.repeat_interleave(4, 0)
        if early is None:
            early = new[: args.eval_n]
            early_T = Tn[: args.eval_n]
            base_early = acc_block(run, early, mem, None, early_T, reads=("none", "oracle"))
            R["base_and_oracle_first_night"] = base_early
            R["target_norm_mean"] = round(float(Tn.norm(dim=1).mean()), 3)
            log("first-night facts, no memory / target injected directly:", json.dumps(base_early),
                "target norm", R["target_norm_mean"])
        latest = new[-args.eval_n:]
        distinct = int(torch.unique(rows_all.idx).numel())
        row = {"night": s // args.night + 1, "facts": facts_so_far, "constraints": rows_all.idx.shape[0],
               "constraints_per_slot": round(rows_all.idx.shape[0] / mem.N, 3), "distinct_slots_touched": distinct,
               "target_s": round(t_targets, 1), "methods": {}}
        for m in methods:
            tm = time.time()
            Vm = V[m].to(dev)
            info = {}
            if m == "delta_rule":
                delta_rule(Vm, idx, w, Tn, steps=3, lr=0.5)
            elif m == "batch_ls":
                Vm, info["iters"] = cg(rows_new, Trows_new, Vm, args.anchor, 300, 1e-4, prior=Vm)
            elif m == "joint_ls":
                Vm, info["iters"] = cg(rows_all, Trows_all, Vm, args.lam, 300, 1e-4)
            elif m == "joint_null":
                rows_n = Rows(torch.cat([rows_all.idx, null_idx]), torch.cat([rows_all.w, null_w]), mem.N)
                Tn_rows = torch.cat([Trows_all, torch.zeros(len(null_idx), d, device=dev)])
                Vm, info["iters"] = cg(rows_n, Tn_rows, Vm, args.lam, 300, 1e-4)
            info["write_s"] = round(time.time() - tm, 1)
            rel = certificate(rows_all, Vm, Trows_all)
            info["cert_frac_over_0.2"] = round(float((rel > 0.2).float().mean()), 4)
            info["cert_mean_rel_err"] = round(float(rel.mean()), 4)
            info["first_night"] = acc_block(run, early, mem, Vm)
            info["latest_night"] = acc_block(run, latest, mem, Vm)
            info.update(known_and_kl(run, mem, Vm, known, wins))
            info["eval_s"] = round(time.time() - tm - info["write_s"], 1)
            row["methods"][m] = info
            V[m] = Vm.cpu()
            del Vm
            if dev == "cuda":
                torch.cuda.empty_cache()
        if dev == "cuda":
            row["max_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        row["night_s"] = round(time.time() - tn, 1)
        R["nights"].append(row)
        log(json.dumps(row))
        json.dump(R, open(out_json, "w"), indent=1)

    # ---- exact deletion (joint_ls): remove some first-night facts, re-solve, compare with never-learned
    if "joint_ls" in methods:
        gone = set(range(min(300, args.eval_n // 2)))  # first 300 facts of night 1 (all inside the eval set)
        cf = torch.cat(C_fact)
        keep = torch.tensor([int(f) not in gone for f in cf.tolist()], device=dev)
        rows_k = Rows(torch.cat(C_idx)[keep], torch.cat(C_w)[keep], mem.N)
        Tk = T_all[cf[keep]]
        Vj = V["joint_ls"].to(dev)
        Vdel, it_warm = cg(rows_k, Tk, Vj, args.lam, 600, 1e-5)
        Vref, it_cold = cg(rows_k, Tk, torch.zeros_like(Vj), args.lam, 600, 1e-5)
        gone_facts = [stream[i] for i in sorted(gone)]
        kept_early = [f for i, f in enumerate(early) if i not in gone]
        R["deletion"] = {
            "deleted_facts": len(gone),
            "max_abs_diff_vs_never_learned": float((Vdel - Vref).abs().max()),
            "max_abs_value": float(Vref.abs().max()),
            "deleted_before": acc_block(run, gone_facts, mem, Vj, reads=("last",)),
            "deleted_after": acc_block(run, gone_facts, mem, Vdel, reads=("last",)),
            "deleted_base_no_memory": acc_block(run, gone_facts, mem, None, reads=("none",)),
            "kept_first_night_after": acc_block(run, kept_early, mem, Vdel, reads=("last",)),
            "kept_first_night_before": acc_block(run, kept_early, mem, Vj, reads=("last",)),
            "resolve_iters_warm_cold": [it_warm, it_cold],
        }
        log("deletion", json.dumps(R["deletion"]))
        del Vj, Vdel, Vref
        json.dump(R, open(out_json, "w"), indent=1)

    # ---- Fix 4: forward-only targets (fact in context minus fact absent), against gradient targets
    n1 = min(args.night, len(stream))
    f1 = stream[:n1]
    ctx = lambda f: f"{prompts(f, 'train')[0]} {f['answer']}. "

    def fwd_targets(facts, Hno):
        Hin = run.capture([ctx(f) + p for f in facts for p in prompts(f, "train")]).view(len(facts), 4, -1)
        return (Hin.cpu() - Hno).mean(1).to(dev)

    Hc4 = Hc.view(len(calib), 4, -1)[:200].cpu()
    Dc = fwd_targets(calib[:200], Hc4)
    scales = {}
    for a in (0.5, 1, 2, 4, 8, 16):
        scales[a] = acc_block(run, calib[:200], mem, None, Dc * a, which=("train",), reads=("oracle",))["train_oracle"]
    alpha = max(scales, key=scales.get)
    D1 = fwd_targets(f1, night1_H) * alpha
    rows1 = Rows(C_idx[0], C_w[0], mem.N)
    fix4 = {"scale_search_on_calibration_facts": scales, "scale": alpha}
    for name, T1 in (("gradient_targets", T_all[:n1]), ("forward_only_targets", D1)):
        V1, _ = cg(rows1, T1.repeat_interleave(4, 0), torch.zeros(mem.N, d, device=dev), args.lam, 300, 1e-4)
        e = f1[: args.eval_n]
        r = acc_block(run, e, mem, None, T1[: args.eval_n], reads=("oracle",))
        r.update(acc_block(run, e, mem, V1))
        fix4[name] = r
        del V1
    R["fix4_forward_only"] = fix4
    log("fix4", json.dumps(fix4))
    R["total_s"] = round(time.time() - t0, 1)
    json.dump(R, open(out_json, "w"), indent=1)
    if not args.tiny:
        open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8").write(report(R))
    log("done", out_json)


def report(R):
    ms = list(R["nights"][0]["methods"])
    lines = [f"# Real-LM memory test: {R['config']['model']}", "",
             f"Memory after block {R['layer']} of {R['n_layers']}, {R['slots']:,} slots x {R['d']} dims; "
             f"{R['facts_total']:,} facts in nights of {R['config']['night']:,}; 4 wordings written per fact.", "",
             "No memory / target injected directly (first-night facts): " + json.dumps(R["base_and_oracle_first_night"]), "",
             f"Key map used: {R['key_kind']}. Key overlap: " + json.dumps(R["key_overlap"]), "",
             "## First night's facts: top-1 answer accuracy after each night", "",
             "Read = last: the memory is read only at the question's last position. Read = all: read at every position.", "",
             "| Facts | Constraints/slot | " + " | ".join(f"{m} written (last/all)" for m in ms) + " | "
             + " | ".join(f"{m} unseen (last/all)" for m in ms) + " |",
             "|" + "---|" * (2 + 2 * len(ms))]
    for n in R["nights"]:
        c = [f"{n['methods'][m]['first_night']['train_last']:.3f} / {n['methods'][m]['first_night']['train_all']:.3f}" for m in ms]
        u = [f"{n['methods'][m]['first_night']['test_last']:.3f} / {n['methods'][m]['first_night']['test_all']:.3f}" for m in ms]
        lines.append(f"| {n['facts']:,} | {n['constraints_per_slot']:.2f} | " + " | ".join(c + u) + " |")
    lines += ["", "## Latest night's facts, damage to known facts, unrelated text, certificate", "",
              "| Facts | Method | Latest written (last/all) | Latest unseen (last/all) | Known facts | KL on unrelated text | Top-1 agreement | Constraints >20% error |",
              "|---|---|---|---|---|---|---|---|"]
    for n in R["nights"]:
        for m in ms:
            i = n["methods"][m]
            lines.append(f"| {n['facts']:,} | {m} | {i['latest_night']['train_last']:.3f} / {i['latest_night']['train_all']:.3f} | "
                         f"{i['latest_night']['test_last']:.3f} / {i['latest_night']['test_all']:.3f} | {i['known_acc']:.3f} | "
                         f"{i['kl_unrelated']:.4f} | {i['top1_agree_unrelated']:.3f} | {i['cert_frac_over_0.2']:.4f} |")
    lines += ["", f"Known facts the base model answers correctly: {R['known_facts_base_correct']}", ""]
    if "deletion" in R:
        lines += ["## Deletion", "", "```", json.dumps(R["deletion"], indent=1), "```", ""]
    lines += ["## Forward-only targets (Fix 4)", "", "```", json.dumps(R["fix4_forward_only"], indent=1), "```", "",
              f"Total time: {R['total_s'] / 60:.1f} min"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
