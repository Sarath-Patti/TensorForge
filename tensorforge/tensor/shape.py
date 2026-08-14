"""Shape and stride manipulation utilities for TensorForge.

Handles tensor dimension calculations, C-contiguous stride computation,
memory layout checks, and multi-dimensional broadcasting geometry.
In v0.1, all tensors operate in row-major (C-contiguous) layout.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from tensorforge.utils.validation import ShapeError, validate_broadcast_shapes, validate_shape


def compute_numel(shape: Tuple[int, ...]) -> int:
    """Compute the total number of elements for a given shape.

    Args:
        shape: A tuple of non-negative integer dimensions.

    Returns:
        Total number of elements (1 for 0-D scalar, product of dims otherwise, 0 if any dim is 0).
    """
    if not shape:
        return 1
    return math.prod(shape)


def compute_contiguous_strides(shape: Tuple[int, ...], itemsize: int = 1) -> Tuple[int, ...]:
    """Compute row-major (C-contiguous) strides in element counts or byte units.

    For shape (d0, d1, ..., dn-1), the contiguous stride in element count is:
        stride[i] = prod(shape[i+1:]) * itemsize
        stride[n-1] = 1 * itemsize

    Args:
        shape: Tensor dimensions tuple.
        itemsize: Size of a single element (1 for element strides, or dtype itemsize for byte strides).

    Returns:
        Tuple of strides corresponding to each dimension.
    """
    if not shape:
        return ()

    ndim = len(shape)
    strides = [0] * ndim
    current_stride = itemsize

    for i in range(ndim - 1, -1, -1):
        strides[i] = current_stride
        current_stride *= max(shape[i], 1)

    return tuple(strides)


def is_c_contiguous(shape: Tuple[int, ...], strides: Tuple[int, ...], itemsize: int = 1) -> bool:
    """Check if the given shape and strides correspond to a row-major (C-contiguous) memory layout.

    Args:
        shape: Tensor dimensions tuple.
        strides: Current strides tuple (in elements or bytes matching itemsize).
        itemsize: Size of element unit in which strides are measured.

    Returns:
        True if memory is contiguous, False otherwise.
    """
    if not shape or len(shape) <= 1:
        return True

    expected = compute_contiguous_strides(shape, itemsize=itemsize)
    return strides == expected


def broadcast_shapes(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute the broadcast target shape of two tensors.

    Args:
        shape_a: First shape tuple.
        shape_b: Second shape tuple.

    Returns:
        Broadcasted shape tuple.

    Raises:
        ShapeError: If shapes cannot be broadcast together.
    """
    return validate_broadcast_shapes(shape_a, shape_b)


def broadcast_strides(
    orig_shape: Tuple[int, ...],
    orig_strides: Tuple[int, ...],
    target_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Compute new strides when broadcasting an existing tensor to a target shape.

    Prepends 0-strides for added dimensions and sets stride to 0 for dimensions of size 1
    that are expanded to larger sizes.

    Args:
        orig_shape: Original tensor shape.
        orig_strides: Original tensor strides.
        target_shape: Target broadcast shape.

    Returns:
        New strides tuple of length len(target_shape).

    Raises:
        ShapeError: If orig_shape cannot be broadcast to target_shape.
    """
    orig_ndim = len(orig_shape)
    target_ndim = len(target_shape)

    if orig_ndim > target_ndim:
        raise ShapeError(
            f"Cannot broadcast shape {orig_shape} ({orig_ndim}D) to smaller target shape {target_shape} ({target_ndim}D)"
        )

    # Pad original shape and strides from the left
    pad_len = target_ndim - orig_ndim
    padded_orig_shape = (1,) * pad_len + orig_shape
    padded_orig_strides = (0,) * pad_len + orig_strides

    new_strides: list[int] = []
    for orig_dim, orig_stride, target_dim in zip(padded_orig_shape, padded_orig_strides, target_shape):
        if orig_dim == target_dim:
            new_strides.append(orig_stride)
        elif orig_dim == 1:
            new_strides.append(0)  # Striding across this dimension reads the same element
        else:
            raise ShapeError(
                f"Cannot broadcast dimension {orig_dim} into {target_dim} (shapes {orig_shape} -> {target_shape})"
            )

    return tuple(new_strides)
