"""Loss function modules for TensorForge neural networks."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

from tensorforge.nn.module import Module
from tensorforge.tensor.operations import cross_entropy
from tensorforge.tensor.tensor import Tensor


class MSELoss(Module):
    """Measures the element-wise mean squared error (squared L2 norm) between predictions and targets.

    Args:
        reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum' (default: 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'; expected 'mean', 'sum', or 'none'")
        self.reduction: str = reduction

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Compute MSE loss.

        Args:
            prediction: Model predicted tensor.
            target: Ground truth target tensor of matching shape.

        Returns:
            Computed MSE loss tensor.
        """
        diff = prediction - target
        sq = diff * diff

        if self.reduction == "mean":
            return sq.mean()
        elif self.reduction == "sum":
            return sq.sum()
        else:
            return sq

    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class CrossEntropyLoss(Module):
    """Computes the cross-entropy loss between input logits and target class indices.

    Combines stable log-softmax and negative log-likelihood computation.

    Args:
        reduction: Specifies the reduction to apply: 'none' | 'mean' | 'sum' (default: 'mean').
    """

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}'; expected 'mean', 'sum', or 'none'")
        self.reduction: str = reduction

    def forward(
        self,
        logits: Tensor,
        targets: Union[Tensor, Sequence[int], np.ndarray],
    ) -> Tensor:
        """Compute Cross-Entropy loss.

        Args:
            logits: Unnormalized class scores of shape (N, C) or (C,).
            targets: Target class indices of shape (N,) or integer scalar.

        Returns:
            Computed Cross-Entropy loss tensor.
        """
        return cross_entropy(logits, targets, reduction=self.reduction)

    def __repr__(self) -> str:
        return f"CrossEntropyLoss(reduction='{self.reduction}')"
