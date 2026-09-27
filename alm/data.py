"""Penn Treebank (Mikolov's 10k-vocabulary version), read as one continuous token stream."""
import os
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ptb")
URL = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.{}.txt"


def _tokens(split):
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, f"ptb.{split}.txt")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL.format(split), path)
    out = []
    with open(path) as f:
        for line in f:
            out.extend(line.split() + ["<eos>"])
    return out


def load():
    """Returns (train_ids, valid_ids, test_ids, vocab list). Every line ends with <eos>."""
    tr, va, te = _tokens("train"), _tokens("valid"), _tokens("test")
    vocab = sorted(set(tr))
    idx = {w: i for i, w in enumerate(vocab)}
    enc = lambda toks: np.array([idx[w] for w in toks], dtype=np.int64)
    return enc(tr), enc(va), enc(te), vocab
