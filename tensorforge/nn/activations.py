"""Activation function modules for TensorForge neural networks."""

from __future__ import annotations

from tensorforge.nn.module import Module
from tensorforge.tensor.operations import relu, sigmoid, softmax, tanh
from tensorforge.tensor.tensor import Tensor


class ReLU(Module):
    """Applies the rectified linear unit activation function element-wise: ReLU(x) = max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)

    def __repr__(self) -> str:
        return "ReLU()"


class Sigmoid(Module):
    """Applies the element-wise logistic sigmoid function: Sigmoid(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: Tensor) -> Tensor:
        return sigmoid(x)

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    """Applies the element-wise hyperbolic tangent function: Tanh(x) = tanh(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return tanh(x)

    def __repr__(self) -> str:
        return "Tanh()"


class Softmax(Module):
    """Applies the Softmax function to an n-dimensional input Tensor rescaling elements so that they sum to 1.

    Args:
        dim: A dimension along which Softmax will be computed (default: -1).
    """

    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim: int = int(dim)

    def forward(self, x: Tensor) -> Tensor:
        return softmax(x, dim=self.dim)

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"
