"""Plotting utilities: training curves and gradient-flow diagnostics.

All functions save a figure to disk and return the path, so experiments can
print clickable output. matplotlib's non-interactive Agg backend is forced so
the plots render identically in scripts, CI, and headless environments.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# A consistent, readable style across all project figures.
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
})


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def plot_training_curves(history: dict, out_path: str,
                         title: str = "Training on MNIST") -> str:
    """Plot loss and accuracy over epochs side by side.

    ``history`` maps series names to lists of per-epoch values. Keys containing
    ``"loss"`` go on the left axis panel; keys containing ``"acc"`` on the right.
    """
    _ensure_dir(out_path)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    for name, values in history.items():
        epochs = np.arange(1, len(values) + 1)
        if "loss" in name:
            ax_loss.plot(epochs, values, marker="o", markersize=3, label=name)
        elif "acc" in name:
            ax_acc.plot(epochs, np.asarray(values) * 100.0,
                        marker="o", markersize=3, label=name)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy (%)")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()

    fig.suptitle(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_gradient_flow(named_grads: dict, out_path: str,
                       title: str = "Gradient flow through the network") -> str:
    """Bar chart of mean |gradient| per parameter tensor, in network order.

    A healthy network shows non-zero gradients at every layer; vanishing
    gradients appear as bars collapsing toward zero in the early layers.
    ``named_grads`` maps parameter names (in forward order) to gradient arrays.
    """
    _ensure_dir(out_path)
    names = list(named_grads.keys())
    means = [float(np.mean(np.abs(g))) for g in named_grads.values()]
    maxes = [float(np.max(np.abs(g))) for g in named_grads.values()]

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(names)), 4.5))
    x = np.arange(len(names))
    ax.bar(x, maxes, width=0.6, color="#aecbe8", label="max |grad|")
    ax.bar(x, means, width=0.6, color="#1f77b4", label="mean |grad|")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("|gradient| (log scale)")
    ax.set_title(title)
    ax.legend()

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_predictions_grid(images: np.ndarray, true_labels: np.ndarray,
                          pred_labels: np.ndarray, out_path: str,
                          n: int = 16,
                          title: str = "Sample test predictions") -> str:
    """Show a grid of test digits with predicted vs true labels.

    Correct predictions are titled in green, mistakes in red. ``images`` may be
    flat ``(N, 784)`` or square ``(N, 28, 28)``.
    """
    _ensure_dir(out_path)
    if images.ndim == 2:
        side = int(np.sqrt(images.shape[1]))
        images = images.reshape(-1, side, side)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.6 * cols, 1.9 * rows))

    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i], cmap="gray_r")
        ok = pred_labels[i] == true_labels[i]
        ax.set_title(f"pred {pred_labels[i]} / true {true_labels[i]}",
                     fontsize=8, color="green" if ok else "red")

    fig.suptitle(title)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
