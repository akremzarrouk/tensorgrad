"""Learn a non-linear decision boundary on the interleaved-spirals dataset.

The three spiral arms cannot be separated by any straight line, so this
experiment demonstrates the whole point of hidden layers: the network bends the
input space until the classes become separable. Runs fully offline in a few
seconds and produces one of the project's nicer figures.

    python experiments/spirals_decision_boundary.py

Output: figures/spirals_decision_boundary.png
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorgrad import Tensor
from tensorgrad import nn
from tensorgrad.data import make_spirals
from tensorgrad.functional import cross_entropy, accuracy, predict
from tensorgrad.optim import Adam
from tensorgrad.viz import plot_decision_boundary

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")

EPOCHS = 400


def main():
    np.random.seed(0)
    X, y = make_spirals(points_per_class=120, num_classes=3, noise=0.15)

    net = nn.MLP([2, 48, 48, 3], activation="relu")
    opt = Adam(net.parameters(), lr=5e-3)

    # Full-batch training: 360 points is tiny, so no minibatching needed.
    inputs = Tensor(X)
    for epoch in range(1, EPOCHS + 1):
        opt.zero_grad()
        loss = cross_entropy(net(inputs), y)
        loss.backward()
        opt.step()
        if epoch % 100 == 0 or epoch == 1:
            acc = accuracy(net(inputs), y)
            print(f"epoch {epoch:3d}/{EPOCHS} | loss {float(loss.data):.4f} "
                  f"| train acc {acc*100:.1f}%")

    final_acc = accuracy(net(inputs), y)
    out = plot_decision_boundary(
        lambda pts: predict(net(Tensor(pts))), X, y,
        os.path.join(FIGURES_DIR, "spirals_decision_boundary.png"),
        title=f"MLP [2, 48, 48, 3] decision regions on the spirals dataset "
              f"({final_acc*100:.1f}% train accuracy)")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
