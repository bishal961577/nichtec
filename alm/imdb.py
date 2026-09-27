"""IMDB movie reviews (50,000 reviews, ~11.7M words; Keras' public copy on Google Cloud Storage)
as a word stream for a data-scaling test. Lower-cased, punctuation removed by the source.

Vocabulary: the 10,000 most frequent words (the source's own frequency ranks) + <unk> (0) + <eos>
(10001, appended to every review). Reviews are shuffled once with a fixed seed, then split into
46,000 training reviews, 2,000 validation reviews and 2,000 test reviews."""
import os
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "imdb")
URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz"
TOP = 10000
EOS = TOP + 1
V = TOP + 2


def load():
    os.makedirs(ROOT, exist_ok=True)
    path = os.path.join(ROOT, "imdb.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(URL, path)
    d = np.load(path, allow_pickle=True)
    reviews = list(d["x_train"]) + list(d["x_test"])
    order = np.random.default_rng(12345).permutation(len(reviews))

    def stream(idx):
        parts = []
        for i in idx:
            r = np.asarray(reviews[i], np.int64)
            r = np.where(r <= TOP, r, 0)
            parts.append(np.append(r, EOS))
        return np.concatenate(parts)

    return stream(order[4000:]), stream(order[:2000]), stream(order[2000:4000])
