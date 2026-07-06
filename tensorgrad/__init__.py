"""tensorgrad: a small, NumPy-backed reverse-mode automatic differentiation
engine and neural-network library, built from first principles.

The differentiation logic is implemented from scratch - NumPy is used only for
fast array arithmetic, not for gradients. See ``tensorgrad.engine`` for the core.
"""

from .engine import Tensor

__all__ = ["Tensor"]
__version__ = "0.1.0"
