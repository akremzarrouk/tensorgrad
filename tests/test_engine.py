"""Behavioural tests for the Tensor engine: forward values, shapes, and a few
hand-computed gradients where the right answer is obvious by inspection.
"""

import numpy as np
import pytest

from tensorgrad import Tensor


def test_add_mul_forward():
    a = Tensor(3.0)
    b = Tensor(4.0)
    assert (a + b).data == 7.0
    assert (a * b).data == 12.0


def test_scalar_chain_rule():
    # f = a*b + b**2  ->  df/da = b = 4,  df/db = a + 2b = 11
    a = Tensor(3.0)
    b = Tensor(4.0)
    f = a * b + b ** 2
    f.backward()
    assert f.data == pytest.approx(28.0)
    assert a.grad == pytest.approx(4.0)
    assert b.grad == pytest.approx(11.0)


def test_reused_variable_accumulates():
    # f = x * x  ->  df/dx = 2x. This only works if the graph accumulates the
    # gradient contribution from *both* edges into x.
    x = Tensor(5.0)
    f = x * x
    f.backward()
    assert x.grad == pytest.approx(10.0)


def test_broadcasting_shapes():
    X = Tensor(np.random.randn(4, 3))
    b = Tensor(np.random.randn(3))
    out = (X + b).sum()
    out.backward()
    # Gradients must match the ORIGINAL operand shapes, not the broadcast shape.
    assert X.grad.shape == (4, 3)
    assert b.grad.shape == (3,)
    # Each element of b was added across all 4 rows -> grad is 4 everywhere.
    assert np.allclose(b.grad, 4.0)


def test_matmul_forward_and_grad_shapes():
    X = Tensor(np.random.randn(5, 3))
    W = Tensor(np.random.randn(3, 2))
    out = (X @ W).sum()
    out.backward()
    assert X.grad.shape == (5, 3)
    assert W.grad.shape == (3, 2)


def test_relu_masks_negatives():
    x = Tensor(np.array([-2.0, -0.5, 0.0, 1.5]))
    out = x.relu().sum()
    out.backward()
    # Gradient is 1 where input > 0, else 0.
    assert np.allclose(x.grad, np.array([0.0, 0.0, 0.0, 1.0]))


def test_mean_matches_manual():
    x = Tensor(np.array([1.0, 2.0, 3.0, 4.0]))
    out = x.mean()
    out.backward()
    assert out.data == pytest.approx(2.5)
    assert np.allclose(x.grad, 0.25)  # each of 4 elements contributes 1/N


def test_division_and_rsub():
    a = Tensor(2.0)
    b = Tensor(4.0)
    # (10 - a) / b = 8 / 4 = 2
    out = (10 - a) / b
    assert out.data == pytest.approx(2.0)
    out.backward()
    # d/da [(10-a)/b] = -1/b = -0.25 ; d/db = -(10-a)/b^2 = -8/16 = -0.5
    assert a.grad == pytest.approx(-0.25)
    assert b.grad == pytest.approx(-0.5)
