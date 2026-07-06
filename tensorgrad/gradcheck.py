"""Numerical gradient checking.

The single most valuable tool when implementing autodiff by hand is a
*numerical* gradient check: compare each analytical derivative produced by the
engine against a finite-difference estimate computed only from forward passes.

We use the symmetric (central) difference

    df/dx_i  ≈  (f(x + eps * e_i) - f(x - eps * e_i)) / (2 * eps)

which has error O(eps^2) — far more accurate than the one-sided difference — and
then measure agreement with the analytical gradient using a scale-invariant
relative error. If the hand-derived backward pass is correct, the two agree to
roughly single-float precision.

This module is deliberately part of the library (not just the tests): the same
check is reused in the experiments to certify the whole network's gradients.
"""

from __future__ import annotations

import numpy as np

from .engine import Tensor


def numerical_gradient(f, tensor: Tensor, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference gradient of scalar ``f()`` w.r.t. ``tensor``.

    ``f`` must be a zero-argument callable that rebuilds the computation and
    returns a scalar :class:`Tensor`. Each element of ``tensor.data`` is nudged
    by ``±eps`` in place (and restored) to estimate its partial derivative.
    """
    grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        original = tensor.data[idx]

        tensor.data[idx] = original + eps
        f_plus = float(np.asarray(f().data))

        tensor.data[idx] = original - eps
        f_minus = float(np.asarray(f().data))

        tensor.data[idx] = original  # restore
        grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()
    return grad


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Scale-invariant max relative error between two gradient arrays."""
    a, b = np.asarray(a), np.asarray(b)
    denom = np.maximum(np.abs(a) + np.abs(b), 1e-12)
    return float(np.max(np.abs(a - b) / denom))


def gradient_check(f, inputs, eps: float = 1e-6, tol: float = 1e-5):
    """Check analytical vs numerical gradients for every tensor in ``inputs``.

    Returns a dict ``{index: relative_error}``. Raises ``AssertionError`` if any
    tensor's relative error exceeds ``tol``, so it doubles as a test assertion.
    """
    inputs = list(inputs)
    for t in inputs:
        t.zero_grad()

    loss = f()
    assert loss.data.size == 1, "gradient_check expects f() to return a scalar"
    loss.backward()

    errors = {}
    for i, t in enumerate(inputs):
        analytical = t.grad.copy()
        numerical = numerical_gradient(f, t, eps)
        err = relative_error(analytical, numerical)
        errors[i] = err
        assert err < tol, (
            f"gradient check failed for input #{i}: "
            f"relative error {err:.2e} exceeds tol {tol:.0e}"
        )
    return errors
