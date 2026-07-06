"""Offline tests for the synthetic dataset and mini-batch iterator.

(The MNIST loader requires network access to download on first use, so it is
exercised by the experiments rather than the unit tests.)
"""

import numpy as np

from tensorgrad.data import make_spirals, iterate_minibatches


def test_spirals_shape_and_labels():
    X, y = make_spirals(points_per_class=50, num_classes=3)
    assert X.shape == (150, 2)
    assert y.shape == (150,)
    assert sorted(set(y.tolist())) == [0, 1, 2]


def test_spirals_deterministic():
    X1, _ = make_spirals(seed=42)
    X2, _ = make_spirals(seed=42)
    assert np.array_equal(X1, X2)


def test_minibatches_cover_all_examples_once():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    seen = []
    total = 0
    for xb, yb in iterate_minibatches(X, y, batch_size=32, seed=0):
        assert xb.shape[0] == yb.shape[0]
        seen.extend(yb.tolist())
        total += len(yb)
    assert total == 100                       # every example exactly once
    assert sorted(seen) == list(range(100))   # no duplicates or drops


def test_minibatches_last_batch_smaller():
    X = np.zeros((100, 1))
    y = np.zeros(100)
    sizes = [len(yb) for _, yb in iterate_minibatches(X, y, 32, shuffle=False)]
    assert sizes == [32, 32, 32, 4]
