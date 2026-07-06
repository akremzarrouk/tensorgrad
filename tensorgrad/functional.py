"""Loss functions and prediction helpers.

The softmax and cross-entropy here are implemented with the standard
*log-sum-exp* stabilisation. Rather than computing ``softmax`` and then its log
(which overflows for large logits), we subtract the per-row max - a constant
that does not change the result - before exponentiating. The classic identity is

    log softmax(z)_i = z_i - max(z) - log( sum_j exp(z_j - max(z)) )

Cross-entropy is written using only differentiable engine ops (a one-hot mask
picks out the correct class), so its gradient is produced by the engine, not by
a hand-coded special case.
"""

from __future__ import annotations

import numpy as np

from .engine import Tensor


def softmax(logits: Tensor) -> Tensor:
    """Row-wise softmax producing a probability distribution per example."""
    # The per-row max is treated as a constant (taken from .data) for stability.
    shift = logits.data.max(axis=1, keepdims=True)
    exps = (logits - shift).exp()
    return exps / exps.sum(axis=1, keepdims=True)


def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy loss.

    Parameters
    ----------
    logits:
        ``(N, C)`` tensor of unnormalised class scores.
    targets:
        ``(N,)`` array of integer class labels in ``[0, C)``.
    """
    targets = np.asarray(targets).astype(int)
    n, num_classes = logits.shape

    # Stable log-softmax via log-sum-exp.
    shift = logits.data.max(axis=1, keepdims=True)
    shifted = logits - shift
    log_sum_exp = shifted.exp().sum(axis=1, keepdims=True).log()
    log_probs = shifted - log_sum_exp  # (N, C)

    # One-hot mask selects the log-probability of each example's true class.
    onehot = np.zeros((n, num_classes))
    onehot[np.arange(n), targets] = 1.0
    picked = (log_probs * onehot).sum()  # sum of correct-class log-probs

    return -picked / n


def predict(logits: Tensor) -> np.ndarray:
    """Return the arg-max class index per example (non-differentiable)."""
    return np.argmax(logits.data, axis=1)


def accuracy(logits: Tensor, targets: np.ndarray) -> float:
    """Fraction of correct predictions."""
    targets = np.asarray(targets).astype(int)
    return float(np.mean(predict(logits) == targets))
