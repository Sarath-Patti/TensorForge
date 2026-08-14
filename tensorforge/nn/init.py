"""Parameter initialization utilities for TensorForge neural network layers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
import numpy as np

from tensorforge.tensor.storage import NumPyStorage

if TYPE_CHECKING:
    from tensorforge.tensor.tensor import Tensor


def uniform_(tensor: Tensor, a: float = 0.0, b: float = 1.0) -> Tensor:
    """Fills the input Tensor with values drawn from the uniform distribution U(a, b).

    Args:
        tensor: An n-dimensional TensorForge Tensor.
        a: Lower bound of uniform distribution.
        b: Upper bound of uniform distribution.

    Returns:
        The mutated tensor.
    """
    arr = np.random.uniform(a, b, size=tensor.shape).astype(tensor.dtype.numpy_dtype)
    tensor._storage = NumPyStorage(arr, dtype=tensor.dtype)
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with zeros.

    Args:
        tensor: An n-dimensional TensorForge Tensor.

    Returns:
        The mutated tensor.
    """
    arr = np.zeros(tensor.shape, dtype=tensor.dtype.numpy_dtype)
    tensor._storage = NumPyStorage(arr, dtype=tensor.dtype)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fills the input Tensor with ones.

    Args:
        tensor: An n-dimensional TensorForge Tensor.

    Returns:
        The mutated tensor.
    """
    arr = np.ones(tensor.shape, dtype=tensor.dtype.numpy_dtype)
    tensor._storage = NumPyStorage(arr, dtype=tensor.dtype)
    return tensor


def kaiming_uniform_(tensor: Tensor, a: float = 0.0) -> Tensor:
    """Fills the input Tensor with values according to the He (Kaiming) uniform method.

    Args:
        tensor: An n-dimensional TensorForge Tensor.
        a: Negative slope of the rectifier used after this layer (default: 0 for ReLU).

    Returns:
        The mutated tensor.
    """
    fan_in = tensor.shape[1] if tensor.ndim > 1 else tensor.shape[0]
    gain = math.sqrt(2.0 / (1.0 + a ** 2))
    std = gain / math.sqrt(fan_in)
    bound = math.sqrt(3.0) * std
    return uniform_(tensor, -bound, bound)
