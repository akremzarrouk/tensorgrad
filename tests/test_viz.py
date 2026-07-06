"""Smoke tests for the plotting utilities: each function must produce a file.

(Rendering correctness is judged by eye in the README figures; these tests
guard against API breakage and backend issues in headless environments.)
"""

import os

import numpy as np

from tensorgrad.viz import (plot_training_curves, plot_gradient_flow,
                            plot_predictions_grid, plot_decision_boundary,
                            plot_confusion_matrix)


def test_training_curves(tmp_path):
    history = {"train_loss": [1.0, 0.5], "test_loss": [1.1, 0.6],
               "train_acc": [0.5, 0.8], "test_acc": [0.45, 0.75]}
    out = plot_training_curves(history, str(tmp_path / "curves.png"))
    assert os.path.getsize(out) > 0


def test_gradient_flow(tmp_path):
    grads = {"L1.W": np.random.randn(4, 3), "L1.b": np.random.randn(3)}
    out = plot_gradient_flow(grads, str(tmp_path / "flow.png"))
    assert os.path.getsize(out) > 0


def test_predictions_grid(tmp_path):
    images = np.random.rand(4, 16)  # 4x4 "digits"
    labels = np.array([0, 1, 2, 3])
    out = plot_predictions_grid(images, labels, labels,
                                str(tmp_path / "grid.png"), n=4)
    assert os.path.getsize(out) > 0


def test_decision_boundary(tmp_path):
    X = np.random.randn(30, 2)
    y = (X[:, 0] > 0).astype(int)
    out = plot_decision_boundary(lambda pts: (pts[:, 0] > 0).astype(int),
                                 X, y, str(tmp_path / "boundary.png"),
                                 resolution=50)
    assert os.path.getsize(out) > 0


def test_confusion_matrix(tmp_path):
    true = np.array([0, 1, 2, 2, 1])
    pred = np.array([0, 1, 2, 1, 1])
    out = plot_confusion_matrix(true, pred, str(tmp_path / "cm.png"),
                                num_classes=3)
    assert os.path.getsize(out) > 0
