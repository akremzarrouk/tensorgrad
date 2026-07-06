"""The core automatic-differentiation engine.

This module implements reverse-mode automatic differentiation (a.k.a.
backpropagation) from first principles. The only external dependency is NumPy,
and it is used *purely* for fast array arithmetic on the forward pass - every
derivative below is derived and coded by hand.

The design follows the classic "define-by-run" (dynamic graph) style used by
PyTorch and Autograd: as you compute with :class:`Tensor` objects, each
operation records how to propagate a gradient back to its inputs. Calling
:meth:`Tensor.backward` then walks that recorded graph in reverse topological
order, applying the chain rule at every node.

A worked introduction to the math lives in the project README.
"""

from __future__ import annotations

import numpy as np


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after NumPy broadcasting.

    When an operation broadcasts its inputs (e.g. adding a ``(out,)`` bias to a
    ``(batch, out)`` activation), the upstream gradient has the *broadcast*
    shape. The chain rule says the gradient w.r.t. the smaller operand is the
    sum of the upstream gradient over every axis that was expanded. This helper
    performs exactly that reduction so gradients always match their tensor's
    shape.
    """
    # 1) Sum away leading axes that broadcasting added.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # 2) Sum (keeping dims) any axis that was size-1 in the original operand.
    for axis, dim in enumerate(shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    """An n-dimensional array that records operations for autodiff.

    Parameters
    ----------
    data:
        Array-like payload. Stored internally as ``float64`` for numerical
        stability of the finite-difference gradient checks.
    _children:
        Internal. The tensors this one was computed from (its parents in the
        computation graph).
    _op:
        Internal. A short label for the operation that produced this tensor,
        used only for visualisation / debugging.
    label:
        Optional human-readable name (handy when drawing the graph).
    """

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = np.asarray(data, dtype=np.float64)
        # Gradient of some downstream scalar w.r.t. this tensor. Same shape as
        # ``data``; accumulated during the backward pass.
        self.grad = np.zeros_like(self.data)
        # Closure that propagates this node's gradient to its parents. The
        # default (a no-op) is correct for leaf tensors.
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    # ------------------------------------------------------------------ #
    # Convenience / introspection
    # ------------------------------------------------------------------ #
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op={self._op!r})"

    @staticmethod
    def _as_tensor(other) -> "Tensor":
        return other if isinstance(other, Tensor) else Tensor(other)

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    def detach(self) -> "Tensor":
        """Return a new leaf tensor sharing the data but no graph history."""
        return Tensor(self.data.copy())

    # ------------------------------------------------------------------ #
    # Elementwise binary ops
    # ------------------------------------------------------------------ #
    def __add__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out = Tensor(self.data + other.data, (self, other), "+")

        def _backward():
            # d(out)/d(self) = 1, d(out)/d(other) = 1  (then unbroadcast)
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out = Tensor(self.data * other.data, (self, other), "*")

        def _backward():
            # Product rule: d(a*b)/da = b, d(a*b)/db = a.
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, power) -> "Tensor":
        assert isinstance(power, (int, float)), "only scalar powers supported"
        out = Tensor(self.data ** power, (self,), f"**{power}")

        def _backward():
            # d(x**n)/dx = n * x**(n-1)
            self.grad += (power * self.data ** (power - 1)) * out.grad

        out._backward = _backward
        return out

    def __matmul__(self, other) -> "Tensor":
        other = self._as_tensor(other)
        out = Tensor(self.data @ other.data, (self, other), "@")

        def _backward():
            # For C = A @ B:  dA = dC @ B^T,  dB = A^T @ dC.
            # swapaxes(-1, -2) transposes the last two dims (works for batched
            # matmul too); _unbroadcast folds any leading batch dims back.
            grad_self = out.grad @ np.swapaxes(other.data, -1, -2)
            grad_other = np.swapaxes(self.data, -1, -2) @ out.grad
            self.grad += _unbroadcast(grad_self, self.data.shape)
            other.grad += _unbroadcast(grad_other, other.data.shape)

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Derived binary ops (defined in terms of the primitives above)
    # ------------------------------------------------------------------ #
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._as_tensor(other))

    def __truediv__(self, other):
        return self * (self._as_tensor(other) ** -1.0)

    # Reflected operators so ``2 + x``, ``2 * x`` etc. work.
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._as_tensor(other) + (-self)

    def __rtruediv__(self, other):
        return self._as_tensor(other) * (self ** -1.0)

    # ------------------------------------------------------------------ #
    # Reductions
    # ------------------------------------------------------------------ #
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), "sum")

        def _backward():
            grad = out.grad
            # If we reduced without keepdims, restore the reduced axes as size-1
            # so the gradient broadcasts back over them.
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            # d(sum)/d(each element) = 1, so just broadcast upstream grad.
            self.grad += np.broadcast_to(grad, self.data.shape).copy()

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), "mean")

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            # Each element contributed 1/N to the mean.
            n = self.data.size if axis is None else self.data.shape[axis]
            self.grad += np.broadcast_to(grad, self.data.shape).copy() / n

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Elementwise unary nonlinearities
    # ------------------------------------------------------------------ #
    def relu(self) -> "Tensor":
        out = Tensor(np.maximum(self.data, 0.0), (self,), "relu")

        def _backward():
            # Gradient flows only where the input was positive.
            self.grad += (self.data > 0.0) * out.grad

        out._backward = _backward
        return out

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,), "tanh")

        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> "Tensor":
        s = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(s, (self,), "sigmoid")

        def _backward():
            # d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> "Tensor":
        e = np.exp(self.data)
        out = Tensor(e, (self,), "exp")

        def _backward():
            # d/dx e^x = e^x
            self.grad += e * out.grad

        out._backward = _backward
        return out

    def log(self) -> "Tensor":
        out = Tensor(np.log(self.data), (self,), "log")

        def _backward():
            # d/dx log(x) = 1/x
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------ #
    # Shape ops
    # ------------------------------------------------------------------ #
    def reshape(self, *shape) -> "Tensor":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out = Tensor(self.data.reshape(shape), (self,), "reshape")

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def transpose(self) -> "Tensor":
        out = Tensor(self.data.T, (self,), "T")

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out

    @property
    def T(self):
        return self.transpose()

    # ------------------------------------------------------------------ #
    # The backward pass
    # ------------------------------------------------------------------ #
    def backward(self):
        """Populate ``.grad`` on every tensor that fed into ``self``.

        ``self`` is assumed to be a scalar (the loss). We seed its gradient with
        1 (d(loss)/d(loss) = 1) and then apply each node's ``_backward`` closure
        in reverse topological order so that, by the time we process a node,
        every consumer downstream of it has already contributed to its ``.grad``.
        """
        # Build a topological ordering of the graph via DFS.
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
