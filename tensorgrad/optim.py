"""Gradient-based optimisers.

Each optimiser owns a list of parameter tensors and, on :meth:`step`, updates
their ``.data`` in place using the ``.grad`` populated by the backward pass.
Two update rules are provided:

* :class:`SGD` - stochastic gradient descent, with optional (heavy-ball)
  momentum. ``momentum=0`` gives vanilla SGD; ``momentum=0.9`` gives the
  standard momentum method.
* :class:`Adam` - adaptive moment estimation, the widely used default that keeps
  per-parameter running estimates of the gradient's first and second moments.
"""

from __future__ import annotations

import numpy as np


class Optimizer:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic gradient descent with optional momentum.

    Update (momentum ``m``, learning rate ``lr``):

        v <- m * v + grad
        theta <- theta - lr * v
    """

    def __init__(self, parameters, lr=0.01, momentum=0.0):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self._velocity = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        for i, p in enumerate(self.parameters):
            self._velocity[i] = self.momentum * self._velocity[i] + p.grad
            p.data -= self.lr * self._velocity[i]


class Adam(Optimizer):
    """Adam optimiser (Kingma & Ba, 2014).

    Maintains bias-corrected first moment ``m`` and second moment ``v``:

        m <- b1*m + (1-b1)*grad
        v <- b2*v + (1-b2)*grad^2
        m_hat = m / (1 - b1^t),   v_hat = v / (1 - b2^t)
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, parameters, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._m = [np.zeros_like(p.data) for p in self.parameters]
        self._v = [np.zeros_like(p.data) for p in self.parameters]
        self._t = 0

    def step(self):
        self._t += 1
        for i, p in enumerate(self.parameters):
            g = p.grad
            self._m[i] = self.beta1 * self._m[i] + (1 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1 - self.beta2) * (g * g)
            m_hat = self._m[i] / (1 - self.beta1 ** self._t)
            v_hat = self._v[i] / (1 - self.beta2 ** self._t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
