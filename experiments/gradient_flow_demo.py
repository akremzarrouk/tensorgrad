"""Visualise healthy vs vanishing gradients on the spirals dataset.

Trains nothing - this experiment takes a *single* backward pass through two
deep MLPs on the synthetic spirals data:

* a ReLU network, where gradient magnitudes stay roughly stable with depth;
* a sigmoid network, where each layer multiplies in a factor <= 0.25, so
  gradients shrink visibly toward the early layers - the classic vanishing-
  gradient effect that motivated ReLU.

    python experiments/gradient_flow_demo.py

Output: figures/gradient_flow_relu_vs_sigmoid.png
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorgrad import Tensor
from tensorgrad import nn
from tensorgrad.data import make_spirals
from tensorgrad.functional import cross_entropy

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")

DEPTH = 8          # number of hidden layers
WIDTH = 32


def layer_gradient_means(activation: str, X: np.ndarray, y: np.ndarray):
    """One backward pass; return mean |grad| of each Linear layer's W."""
    np.random.seed(0)
    sizes = [2] + [WIDTH] * DEPTH + [3]
    net = nn.MLP(sizes, activation=activation)
    loss = cross_entropy(net(Tensor(X)), y)
    loss.backward()
    linears = [l for l in net.layers if isinstance(l, nn.Linear)]
    return [float(np.mean(np.abs(l.W.grad))) for l in linears]


def main():
    X, y = make_spirals(points_per_class=100, num_classes=3, seed=0)

    relu_grads = layer_gradient_means("relu", X, y)
    sigmoid_grads = layer_gradient_means("sigmoid", X, y)
    layers = np.arange(1, len(relu_grads) + 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layers, relu_grads, marker="o", label="ReLU network")
    ax.plot(layers, sigmoid_grads, marker="s", label="Sigmoid network")
    ax.set_yscale("log")
    ax.set_xticks(layers)
    ax.set_xlabel("layer (1 = closest to input)")
    ax.set_ylabel("mean |dL/dW| (log scale)")
    ax.set_title(f"Vanishing gradients: mean weight-gradient magnitude per layer\n"
                 f"({DEPTH} hidden layers of width {WIDTH}, single backward pass)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = os.path.join(FIGURES_DIR, "gradient_flow_relu_vs_sigmoid.png")
    os.makedirs(FIGURES_DIR, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"saved {out}")
    print(f"ReLU    layer-1 vs layer-{DEPTH+1} mean|grad|: "
          f"{relu_grads[0]:.2e} vs {relu_grads[-1]:.2e}")
    print(f"Sigmoid layer-1 vs layer-{DEPTH+1} mean|grad|: "
          f"{sigmoid_grads[0]:.2e} vs {sigmoid_grads[-1]:.2e}")


if __name__ == "__main__":
    main()
