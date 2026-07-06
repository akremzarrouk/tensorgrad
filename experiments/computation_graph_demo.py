"""Render the computation graph of a tiny one-neuron network.

Builds loss = (tanh(w*x + b) - target)^2 - a single neuron with a squared
error - runs the backward pass, and draws the engine's actual recorded graph
with the forward value and gradient on every node. This is the picture that
makes reverse-mode autodiff concrete: you can follow the chain rule box by box.

    python experiments/computation_graph_demo.py

Output: figures/computation_graph.png
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorgrad import Tensor
from tensorgrad.graph import draw_graph

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")


def main():
    # A single neuron: pre-activation z = w*x + b, activation a = tanh(z),
    # loss = (a - target)^2.
    x = Tensor(2.0, label="x (input)")
    w = Tensor(-0.5, label="w (weight)")
    b = Tensor(1.5, label="b (bias)")
    target = Tensor(1.0, label="target")

    z = w * x + b
    z.label = "z = w*x + b"
    a = z.tanh()
    a.label = "a = tanh(z)"
    diff = a - target
    loss = diff * diff
    loss.label = "loss = (a-t)^2"

    loss.backward()

    out = draw_graph(
        loss, os.path.join(FIGURES_DIR, "computation_graph.png"),
        title="One neuron, one loss: the recorded computation graph "
              "after backward()")
    print(f"saved {out}")
    print(f"loss = {float(loss.data):.4f}")
    print(f"dloss/dw = {float(w.grad):.4f}   dloss/db = {float(b.grad):.4f}")


if __name__ == "__main__":
    main()
