"""Dataset utilities.

Two data sources are provided:

* :func:`load_mnist` — the real handwritten-digit benchmark. The four IDX files
  are downloaded once (from a public mirror) and cached locally; subsequent
  calls read from disk. This is the "real task" the network is trained on.
* :func:`make_spirals` — a small synthetic 2-D classification problem that runs
  fully offline. It is handy for quick demos, the gradient-flow visualisation,
  and tests that must not touch the network.

Plus :func:`iterate_minibatches` for shuffled mini-batch SGD.
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request

import numpy as np

# Public MNIST mirrors (the original LeCun host is frequently unavailable).
# Both are tried in order; the first that responds wins.
_MNIST_MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-mirror/mnist/",
]
_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

_DEFAULT_MNIST_DIR = os.path.join("data", "mnist")


def _download(filename: str, dest_dir: str) -> str:
    """Download ``filename`` from the first working mirror into ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, filename)
    if os.path.exists(dest):
        return dest

    last_error = None
    for mirror in _MNIST_MIRRORS:
        url = mirror + filename
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest)
            return dest
        except Exception as exc:  # try the next mirror
            last_error = exc
            if os.path.exists(dest):
                os.remove(dest)
    raise RuntimeError(
        f"Could not download {filename} from any mirror. Last error: {last_error}"
    )


def _read_idx_images(path: str) -> np.ndarray:
    """Parse an IDX3 image file into an ``(N, rows, cols)`` uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"unexpected magic {magic} in {path}"
        buf = f.read(num * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(num, rows, cols)


def _read_idx_labels(path: str) -> np.ndarray:
    """Parse an IDX1 label file into an ``(N,)`` uint8 array."""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"unexpected magic {magic} in {path}"
        buf = f.read(num)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(data_dir: str = _DEFAULT_MNIST_DIR, flatten: bool = True,
               normalize: bool = True):
    """Load MNIST, downloading + caching on first use.

    Returns ``(X_train, y_train, X_test, y_test)`` where images are float64.
    With ``flatten=True`` images are ``(N, 784)``; otherwise ``(N, 28, 28)``.
    With ``normalize=True`` pixel values are scaled to ``[0, 1]``.
    """
    paths = {k: _download(v, data_dir) for k, v in _MNIST_FILES.items()}

    X_train = _read_idx_images(paths["train_images"])
    y_train = _read_idx_labels(paths["train_labels"])
    X_test = _read_idx_images(paths["test_images"])
    y_test = _read_idx_labels(paths["test_labels"])

    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    if normalize:
        X_train /= 255.0
        X_test /= 255.0
    if flatten:
        X_train = X_train.reshape(len(X_train), -1)
        X_test = X_test.reshape(len(X_test), -1)

    return X_train, y_train.astype(int), X_test, y_test.astype(int)


def make_spirals(points_per_class: int = 100, num_classes: int = 3,
                 noise: float = 0.2, seed: int = 0):
    """Generate the classic interleaved-spirals classification dataset.

    Returns ``(X, y)`` with ``X`` of shape ``(points_per_class*num_classes, 2)``
    and integer labels ``y``. The classes are not linearly separable, so a
    network must learn a non-linear decision boundary — a good, fast smoke test.
    """
    rng = np.random.RandomState(seed)
    n = points_per_class
    X = np.zeros((n * num_classes, 2))
    y = np.zeros(n * num_classes, dtype=int)
    for c in range(num_classes):
        idx = range(n * c, n * (c + 1))
        radius = np.linspace(0.0, 1.0, n)
        theta = (np.linspace(c * 4, (c + 1) * 4, n)
                 + rng.randn(n) * noise)
        X[idx] = np.c_[radius * np.sin(theta), radius * np.cos(theta)]
        y[idx] = c
    return X, y


def iterate_minibatches(X: np.ndarray, y: np.ndarray, batch_size: int,
                        shuffle: bool = True, seed: int | None = None):
    """Yield ``(X_batch, y_batch)`` tuples covering one epoch."""
    n = len(X)
    indices = np.arange(n)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
    for start in range(0, n, batch_size):
        batch = indices[start:start + batch_size]
        yield X[batch], y[batch]
