"""Tests for the NN layers and loss functions."""

import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad import nn
from tensorgrad.functional import softmax, cross_entropy, accuracy
from tensorgrad.gradcheck import gradient_check


def test_linear_output_shape():
    layer = nn.Linear(4, 3)
    x = Tensor(np.random.randn(5, 4))
    out = layer(x)
    assert out.shape == (5, 3)
    assert len(layer.parameters()) == 2  # W and b


def test_mlp_structure_and_param_count():
    net = nn.MLP([8, 5, 3], activation="relu")
    # Linear(8,5), ReLU, Linear(5,3)  ->  4 parameter tensors (2 per Linear).
    assert len(net.parameters()) == 4
    x = Tensor(np.random.randn(2, 8))
    assert net(x).shape == (2, 3)


def test_softmax_is_a_distribution():
    logits = Tensor(np.random.randn(6, 4))
    probs = softmax(logits)
    assert probs.shape == (6, 4)
    assert np.allclose(probs.data.sum(axis=1), 1.0)
    assert np.all(probs.data >= 0.0)


def test_cross_entropy_uniform_baseline():
    # With all-equal logits, softmax is uniform and loss == log(C).
    n, c = 10, 5
    logits = Tensor(np.zeros((n, c)))
    targets = np.random.randint(0, c, size=n)
    loss = cross_entropy(logits, targets)
    assert loss.data == pytest.approx(np.log(c))


def test_cross_entropy_gradient_check():
    # The most important NN test: verify the loss gradient numerically.
    n, c = 6, 4
    logits = Tensor(np.random.randn(n, c))
    targets = np.random.randint(0, c, size=n)
    gradient_check(lambda: cross_entropy(logits, targets), [logits])


def test_cross_entropy_backprops_through_network():
    net = nn.MLP([4, 6, 3])
    x = Tensor(np.random.randn(5, 4))
    targets = np.random.randint(0, 3, size=5)
    loss = cross_entropy(net(x), targets)
    loss.backward()
    # Every parameter should have received a non-trivial gradient.
    for p in net.parameters():
        assert p.grad.shape == p.data.shape
        assert np.any(p.grad != 0.0)


def test_accuracy_perfect_and_zero():
    logits = Tensor(np.eye(3))          # argmax == row index
    assert accuracy(logits, np.array([0, 1, 2])) == 1.0
    assert accuracy(logits, np.array([1, 2, 0])) == 0.0
