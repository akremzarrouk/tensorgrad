# tensorgrad

**A tensor-based reverse-mode automatic differentiation engine and neural-network
library, built from scratch in Python — with only NumPy for array math.**

> Status: work in progress. This README is expanded with methodology, results,
> and figures as the project is built.

---

## What this is

`tensorgrad` implements the machinery that powers modern deep-learning frameworks
— reverse-mode automatic differentiation (backpropagation) — from first
principles, and then builds a small neural-network library on top of it that
trains on real handwritten-digit images (MNIST).

Crucially, **no deep-learning framework is used for the gradients**. NumPy is
used only to make array arithmetic fast enough to train on real data; every
derivative is derived and implemented by hand.

## Quickstart

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # run the test + gradient-check suite
```

More usage, methodology, and results to follow.

## License

MIT — see [LICENSE](LICENSE).
