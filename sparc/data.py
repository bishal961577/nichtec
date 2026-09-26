"""MNIST / Fashion-MNIST loaders (downloads on first use) and continual-stream builders."""
import gzip
import os
import urllib.request

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SOURCES = {
    "mnist": ("https://ossci-datasets.s3.amazonaws.com/mnist/", ROOT),
    "fashion": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        os.path.join(ROOT, "fashion"),
    ),
}
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def _fetch(name, fname):
    url, folder = SOURCES[name]
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(url + fname, path)
    return path


def _read(path, images):
    with gzip.open(path, "rb") as f:
        data = f.read()
    if images:
        return np.frombuffer(data, np.uint8, offset=16).reshape(-1, 28, 28).astype(np.float32) / 255.0
    return np.frombuffer(data, np.uint8, offset=8).astype(np.int64)


def load(name="mnist"):
    """Returns (x_train, y_train, x_test, y_test); images are float32 in [0,1], shape (N,28,28)."""
    out = []
    for key in ("train_x", "train_y", "test_x", "test_y"):
        out.append(_read(_fetch(name, FILES[key]), images=key.endswith("_x")))
    return tuple(out)


def load_combined():
    """MNIST digits (classes 0-9) followed by Fashion-MNIST clothing (classes 10-19)."""
    a = load("mnist")
    b = load("fashion")
    return (
        np.concatenate([a[0], b[0]]),
        np.concatenate([a[1], b[1] + 10]),
        np.concatenate([a[2], b[2]]),
        np.concatenate([a[3], b[3] + 10]),
    )


def class_incremental_stream(y, tasks, rng):
    """Indices ordered task by task (each task = a tuple of classes); shuffled within a task.

    Every training example appears exactly once: a single pass, no revisiting."""
    order = []
    for task in tasks:
        idx = np.flatnonzero(np.isin(y, task))
        order.append(rng.permutation(idx))
    return order  # list of index arrays, one per task


def iid_stream(y, tasks, rng):
    """Same examples as the class-incremental stream, but shuffled together (the i.i.d. control)."""
    idx = np.flatnonzero(np.isin(y, np.concatenate([np.asarray(t) for t in tasks])))
    return [rng.permutation(idx)]
