"""A minimal neural-network library built on top of the autograd engine.

The API intentionally mirrors PyTorch's ``nn.Module`` at a small scale: layers
are callables that return :class:`~tensorgrad.engine.Tensor` outputs, and each
module reports its learnable :meth:`parameters` so an optimiser can update them.
Because everything is expressed with engine ops, gradients flow automatically.
"""

from __future__ import annotations

import numpy as np

from .engine import Tensor


class Module:
    """Base class: something with parameters and a forward pass."""

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Linear(Module):
    """Fully-connected layer: ``y = x @ W + b``.

    Weights use Kaiming/He initialisation (variance ``2 / fan_in``), which keeps
    activation magnitudes stable through ReLU stacks; biases start at zero.
    """

    def __init__(self, in_features: int, out_features: int):
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(np.random.randn(in_features, out_features) * scale,
                        label="W")
        self.b = Tensor(np.zeros(out_features), label="b")

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


# --------------------------------------------------------------------------- #
# Activation layers (thin wrappers so they compose inside Sequential)
# --------------------------------------------------------------------------- #
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


_ACTIVATIONS = {"relu": ReLU, "tanh": Tanh, "sigmoid": Sigmoid}


class Sequential(Module):
    """Runs a list of modules in order and collects all their parameters."""

    def __init__(self, *layers: Module):
        self.layers = list(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class MLP(Sequential):
    """Convenience builder for a multi-layer perceptron.

    ``MLP([784, 128, 64, 10])`` builds Linear(784,128) -> act -> Linear(128,64)
    -> act -> Linear(64,10). The final layer is linear (it produces logits;
    the softmax lives inside the cross-entropy loss for numerical stability).
    """

    def __init__(self, sizes, activation: str = "relu"):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; "
                             f"choose from {list(_ACTIVATIONS)}")
        act_cls = _ACTIVATIONS[activation]
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:  # no activation after the output layer
                layers.append(act_cls())
        super().__init__(*layers)
