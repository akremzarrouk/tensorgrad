"""Train an MLP on MNIST using only the tensorgrad engine.

This is the project's headline experiment: a 784-128-64-10 ReLU network trained
with Adam on the full 60k-image MNIST training set, evaluated on the held-out
10k test set. Run from the repository root:

    python experiments/train_mnist.py            # full run (~default 8 epochs)
    python experiments/train_mnist.py --epochs 3 # quicker run

Outputs:
    figures/mnist_training_curves.png
    figures/mnist_predictions.png
    figures/gradient_flow_mnist.png
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Allow running as a plain script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tensorgrad import Tensor
from tensorgrad import nn
from tensorgrad.data import load_mnist, iterate_minibatches
from tensorgrad.functional import cross_entropy, accuracy, predict
from tensorgrad.optim import Adam
from tensorgrad.viz import (plot_training_curves, plot_predictions_grid,
                            plot_gradient_flow, plot_confusion_matrix)

FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "figures")


def evaluate(net, X: np.ndarray, y: np.ndarray, batch_size: int = 1000):
    """Mean loss and accuracy over a dataset, in batches to bound memory."""
    losses, correct, total = [], 0, 0
    for xb, yb in iterate_minibatches(X, y, batch_size, shuffle=False):
        logits = net(Tensor(xb))
        losses.append(float(cross_entropy(logits, yb).data) * len(yb))
        correct += int((predict(logits) == yb).sum())
        total += len(yb)
    return sum(losses) / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64],
                        help="hidden layer sizes")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Loading MNIST (downloads ~11 MB on first run)...")
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"train: {X_train.shape}, test: {X_test.shape}")

    sizes = [784, *args.hidden, 10]
    net = nn.MLP(sizes, activation="relu")
    opt = Adam(net.parameters(), lr=args.lr)
    n_params = sum(p.data.size for p in net.parameters())
    print(f"model: MLP {sizes} (ReLU), {n_params:,} parameters; Adam lr={args.lr}")

    history = {"train_loss": [], "test_loss": [],
               "train_acc": [], "test_acc": []}

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        for xb, yb in iterate_minibatches(X_train, y_train, args.batch_size,
                                          seed=args.seed + epoch):
            opt.zero_grad()
            loss = cross_entropy(net(Tensor(xb)), yb)
            loss.backward()
            opt.step()

        train_loss, train_acc = evaluate(net, X_train, y_train)
        test_loss, test_acc = evaluate(net, X_test, y_test)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        print(f"epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc*100:.2f}% | "
              f"test loss {test_loss:.4f} acc {test_acc*100:.2f}% | "
              f"{time.time() - start:.1f}s")

    # ------------------------------------------------------------------ #
    # Figures
    # ------------------------------------------------------------------ #
    curves_path = plot_training_curves(
        history, os.path.join(FIGURES_DIR, "mnist_training_curves.png"),
        title=f"MLP {sizes} trained with Adam on MNIST")
    print(f"saved {curves_path}")

    # Grid of sample test predictions.
    logits = net(Tensor(X_test[:16]))
    preds_path = plot_predictions_grid(
        X_test[:16], y_test[:16], predict(logits),
        os.path.join(FIGURES_DIR, "mnist_predictions.png"))
    print(f"saved {preds_path}")

    # Confusion matrix over the full test set.
    all_preds = np.concatenate([
        predict(net(Tensor(xb)))
        for xb, _ in iterate_minibatches(X_test, y_test, 1000, shuffle=False)])
    cm_path = plot_confusion_matrix(
        y_test, all_preds, os.path.join(FIGURES_DIR, "mnist_confusion_matrix.png"),
        title=f"Confusion matrix on the 10,000-image test set")
    print(f"saved {cm_path}")

    # Gradient flow snapshot: one fresh backward pass on a batch.
    net.zero_grad()
    loss = cross_entropy(net(Tensor(X_train[:256])), y_train[:256])
    loss.backward()
    linear_layers = [l for l in net.layers if isinstance(l, nn.Linear)]
    named_grads = {}
    for i, layer in enumerate(linear_layers, start=1):
        named_grads[f"L{i}.W"] = layer.W.grad
        named_grads[f"L{i}.b"] = layer.b.grad
    flow_path = plot_gradient_flow(
        named_grads, os.path.join(FIGURES_DIR, "gradient_flow_mnist.png"),
        title="Per-layer gradient magnitudes after one backward pass")
    print(f"saved {flow_path}")

    final = history["test_acc"][-1] * 100
    print(f"\nFinal test accuracy: {final:.2f}%")


if __name__ == "__main__":
    main()
