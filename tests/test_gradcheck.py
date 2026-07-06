"""Finite-difference gradient checks for every operation in the engine.

Each test builds a small scalar function of one or more random tensors, then
asserts that the engine's hand-derived analytical gradient matches the numerical
(central-difference) estimate to a tight tolerance. This is the primary evidence
that the backward pass is correct.
"""

import numpy as np
import pytest

from tensorgrad import Tensor
from tensorgrad.gradcheck import gradient_check


def test_add():
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (a + b).sum(), [a, b])


def test_add_broadcast():
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(3))  # broadcast across rows
    gradient_check(lambda: (a + b).sum(), [a, b])


def test_mul():
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (a * b).sum(), [a, b])


def test_mul_broadcast():
    a = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(1, 3))
    gradient_check(lambda: (a * b).sum(), [a, b])


def test_sub():
    a = Tensor(np.random.randn(3, 3))
    b = Tensor(np.random.randn(3, 3))
    gradient_check(lambda: (a - b).sum(), [a, b])


def test_div():
    a = Tensor(np.random.randn(3, 3))
    b = Tensor(np.random.rand(3, 3) + 1.5)  # keep denominator away from 0
    gradient_check(lambda: (a / b).sum(), [a, b])


def test_pow():
    a = Tensor(np.random.rand(3, 3) + 0.5)  # positive base
    gradient_check(lambda: (a ** 3).sum(), [a])


def test_matmul():
    A = Tensor(np.random.randn(5, 4))
    B = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (A @ B).sum(), [A, B])


def test_matmul_then_nonlinear():
    A = Tensor(np.random.randn(6, 4))
    B = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (A @ B).relu().sum(), [A, B])


def test_sum_axis():
    x = Tensor(np.random.randn(4, 5))
    gradient_check(lambda: (x.sum(axis=1) ** 2).sum(), [x])


def test_mean_axis():
    x = Tensor(np.random.randn(4, 5))
    gradient_check(lambda: (x.mean(axis=0) ** 2).sum(), [x])


def test_relu():
    x = Tensor(np.random.randn(4, 4))
    gradient_check(lambda: x.relu().sum(), [x])


def test_tanh():
    x = Tensor(np.random.randn(4, 4))
    gradient_check(lambda: x.tanh().sum(), [x])


def test_sigmoid():
    x = Tensor(np.random.randn(4, 4))
    gradient_check(lambda: x.sigmoid().sum(), [x])


def test_exp():
    x = Tensor(np.random.randn(4, 4) * 0.5)
    gradient_check(lambda: x.exp().sum(), [x])


def test_log():
    x = Tensor(np.random.rand(4, 4) + 0.5)  # positive
    gradient_check(lambda: x.log().sum(), [x])


def test_reshape():
    x = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (x.reshape(3, 4) ** 2).sum(), [x])


def test_transpose():
    x = Tensor(np.random.randn(4, 3))
    gradient_check(lambda: (x.T ** 2).sum(), [x])


def test_composite_expression():
    # A deeper composite exercising the chain rule across many op types at once.
    W = Tensor(np.random.randn(4, 3))
    x = Tensor(np.random.randn(2, 4))
    b = Tensor(np.random.randn(3))
    gradient_check(lambda: ((x @ W + b).tanh() ** 2).mean(), [W, x, b])
