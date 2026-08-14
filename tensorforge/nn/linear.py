"""Fully-connected (dense) linear layer for TensorForge."""

from __future__ import annotations

import math
from typing import Optional, Union

from tensorforge.nn.init import uniform_
from tensorforge.nn.module import Module
from tensorforge.nn.parameter import Parameter
from tensorforge.tensor.dtype import DType, float32, to_dtype
from tensorforge.tensor.tensor import Tensor, zeros


class Linear(Module):
    """Applies an affine linear transformation to the incoming data: y = xA^T + b.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        bias: If False, the layer will not learn an additive bias (default: True).
        dtype: Data type of layer parameters (default: float32).

    Attributes:
        weight: The learnable weights of the module of shape (out_features, in_features).
        bias: The learnable bias of the module of shape (out_features,).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Union[DType, str, type] = float32,
    ) -> None:
        super().__init__()
        self.in_features: int = int(in_features)
        self.out_features: int = int(out_features)
        self.dtype: DType = to_dtype(dtype)

        # Allocate weight parameter of shape (out_features, in_features)
        self.weight: Parameter = Parameter(
            zeros((self.out_features, self.in_features), dtype=self.dtype)
        )

        if bias:
            self.bias: Optional[Parameter] = Parameter(
                zeros((self.out_features,), dtype=self.dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize weights and biases uniformly in range [-1 / sqrt(in_features), 1 / sqrt(in_features)]."""
        bound = 1.0 / math.sqrt(self.in_features) if self.in_features > 0 else 0.0
        uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        """Execute linear forward transformation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None})"
        )
