"""Computation-graph visualisation.

Because the engine records every operation (each Tensor knows its parents in
``_prev`` and the op that produced it), we can draw the *actual* graph that
``backward()`` walks — not an idealised diagram. Each node shows the tensor's
label/op, its forward value, and the gradient deposited by the backward pass,
making the chain rule visible end to end.

Nodes are laid out on a simple grid: column = depth in the graph (longest path
from a leaf), row = position within that column. For the small demonstration
graphs this is intended for, that layout is clear and fully deterministic.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import Tensor
from .viz import _ensure_dir


def _trace(root: Tensor):
    """Collect all nodes and edges reachable from ``root``.

    Nodes are returned in a stable (insertion) order so layouts are
    reproducible run to run.
    """
    nodes, edges = [], []
    seen = set()

    def walk(t: Tensor):
        if id(t) in seen:
            return
        seen.add(id(t))
        nodes.append(t)
        for parent in sorted(t._prev, key=id):
            edges.append((parent, t))
            walk(parent)

    walk(root)
    return nodes, edges


def _fmt(arr: np.ndarray) -> str:
    """Compact one-line description of an array for a node label."""
    if arr.size == 1:
        return f"{float(arr):.3g}"
    return f"shape {arr.shape}"


def _depth_of(nodes):
    """Column index for each node = its longest path from any leaf."""
    depth = {}

    def compute(t: Tensor) -> int:
        if id(t) in depth:
            return depth[id(t)]
        depth[id(t)] = 0 if not t._prev else 1 + max(compute(p) for p in t._prev)
        return depth[id(t)]

    for n in nodes:
        compute(n)
    return depth


def draw_graph(root: Tensor, out_path: str,
               title: str = "Computation graph (forward values + gradients)") -> str:
    """Render the computation graph rooted at ``root`` to ``out_path``.

    Call this *after* ``root.backward()`` so gradients are populated. Intended
    for small demonstration graphs (a handful of ops); a full MLP has far too
    many nodes to read.
    """
    _ensure_dir(out_path)
    nodes, edges = _trace(root)
    depth = _depth_of(nodes)

    # Grid layout: x = depth, y = slot within the column (centred vertically).
    columns = {}
    for n in nodes:
        columns.setdefault(depth[id(n)], []).append(n)
    max_rows = max(len(col) for col in columns.values())

    pos = {}
    for d, col in columns.items():
        k = len(col)
        # Even vertical spread, centred on 0; single node sits at 0.
        ys = np.linspace((k - 1) / 2.0, -(k - 1) / 2.0, k)
        for n, y in zip(col, ys):
            pos[id(n)] = (float(d), float(y))

    n_cols = len(columns)
    fig, ax = plt.subplots(figsize=(2.7 * n_cols + 1.0, 1.7 * max_rows + 1.2))
    ax.set_axis_off()
    ax.set_title(title, pad=14)
    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(-(max_rows - 1) / 2.0 - 0.6, (max_rows - 1) / 2.0 + 0.6)

    # Edges first (arrows), then node boxes on top.
    for parent, child in edges:
        x0, y0 = pos[id(parent)]
        x1, y1 = pos[id(child)]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#888888",
                                    lw=1.2, shrinkA=40, shrinkB=40))

    for n in nodes:
        x, y = pos[id(n)]
        is_leaf = not n._prev
        header = n.label if n.label else (n._op if n._op else "input")
        text = (f"{header}\n"
                f"value: {_fmt(n.data)}\n"
                f"grad: {_fmt(n.grad) if n.grad.size > 1 else f'{float(n.grad):.3g}'}")
        ax.text(x, y, text, ha="center", va="center", fontsize=9,
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor="#e8f0fe" if is_leaf else "#fde9d9",
                          edgecolor="#4472c4" if is_leaf else "#ed7d31",
                          linewidth=1.4))

    # Legend: leaves (inputs/parameters) vs computed nodes.
    leaf_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#e8f0fe",
                               edgecolor="#4472c4")
    op_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#fde9d9",
                             edgecolor="#ed7d31")
    ax.legend([leaf_patch, op_patch],
              ["leaf (input / parameter)", "operation result"],
              loc="upper left", fontsize=8, framealpha=0.9)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path
