"""Parameter abstraction for trainable neural network weights in TensorForge."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union
import numpy as np

from tensorforge.tensor.dtype import DType
from tensorforge.tensor.tensor import Tensor


class Parameter(Tensor):
    """A kind of Tensor that is to be considered a module parameter.

    Parameters are Tensor subclasses that have `requires_grad=True` by default.
    When assigned as Module attributes, they are automatically registered as
    trainable parameters.
    """

    def __init__(
        self,
        data: Any = None,
        dtype: Optional[Union[DType, str, np.dtype, type]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        copy: bool = True,
        requires_grad: bool = True,
    ) -> None:
        """Initialize a Parameter.

        Args:
            data: Initial parameter data (Tensor, array-like, or scalar).
            dtype: Optional target data type.
            shape: Optional explicit shape override.
            copy: Whether to copy input data.
            requires_grad: Whether to compute gradients (default: True).
        """
        if data is None:
            data = []
        super().__init__(
            data,
            dtype=dtype,
            shape=shape,
            copy=copy,
            requires_grad=requires_grad,
        )

    def __repr__(self) -> str:
        arr_str = np.array2string(
            self.numpy(),
            separator=", ",
            prefix="Parameter(",
        )
        return f"Parameter({arr_str}, dtype={self._dtype}, requires_grad={self._requires_grad})"
