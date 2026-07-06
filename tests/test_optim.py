"""Tests for the optimisers, including a small end-to-end training run."""

import numpy as np

from tensorgrad import Tensor
from tensorgrad import nn
from tensorgrad.optim import SGD, Adam
from tensorgrad.functional import cross_entropy, accuracy


def _quadratic_descent(optimizer_factory, steps=200):
    """Minimise f(x) = sum((x - 3)^2); the optimum is x == 3."""
    x = Tensor(np.zeros(4))
    opt = optimizer_factory([x])
    for _ in range(steps):
        opt.zero_grad()
        loss = ((x - 3.0) ** 2).sum()
        loss.backward()
        opt.step()
    return x.data


def test_sgd_minimises_quadratic():
    x = _quadratic_descent(lambda p: SGD(p, lr=0.1))
    assert np.allclose(x, 3.0, atol=1e-2)


def test_sgd_momentum_minimises_quadratic():
    x = _quadratic_descent(lambda p: SGD(p, lr=0.05, momentum=0.9))
    assert np.allclose(x, 3.0, atol=1e-2)


def test_adam_minimises_quadratic():
    x = _quadratic_descent(lambda p: Adam(p, lr=0.1), steps=400)
    assert np.allclose(x, 3.0, atol=1e-2)


def test_mlp_overfits_tiny_classification():
    """A small MLP + Adam should drive a tiny dataset to perfect accuracy,
    which exercises the whole stack end to end (engine + nn + loss + optim)."""
    rng = np.random.RandomState(0)
    X = Tensor(rng.randn(20, 5))
    targets = rng.randint(0, 3, size=20)

    net = nn.MLP([5, 16, 3], activation="relu")
    opt = Adam(net.parameters(), lr=0.05)

    for _ in range(300):
        opt.zero_grad()
        loss = cross_entropy(net(X), targets)
        loss.backward()
        opt.step()

    assert accuracy(net(X), targets) == 1.0
