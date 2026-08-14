"""Validation utilities and custom exception hierarchy for TensorForge.

Provides clear, actionable error messages for dimension mismatches,
unsupported dtypes, invalid reshape dimensions, and indexing errors.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple


class TensorForgeError(Exception):
    """Base exception for all TensorForge errors."""


class ShapeError(TensorForgeError, ValueError):
    """Raised when shapes are invalid or incompatible for an operation."""


class DimensionError(TensorForgeError, ValueError):
    """Raised when tensor dimensions or axes are incompatible or out of bounds."""


class DTypeError(TensorForgeError, TypeError):
    """Raised when an unsupported or incompatible data type is encountered."""


class IndexError_(TensorForgeError, IndexError):
    """Raised when an indexing operation is out of bounds or invalid."""


class StorageError(TensorForgeError, RuntimeError):
    """Raised when memory allocation or low-level storage operations fail."""


def validate_shape(shape: Any) -> Tuple[int, ...]:
    """Validate and normalize a shape specification.

    Args:
        shape: A sequence of dimension sizes or an integer for 1D.

    Returns:
        Normalized tuple of non-negative integer dimensions.

    Raises:
        ShapeError: If any dimension is negative or non-integer.
    """
    if isinstance(shape, int):
        if shape < 0:
            raise ShapeError(f"Dimension size must be non-negative, got {shape}")
        return (shape,)

    if not isinstance(shape, (tuple, list)):
        raise ShapeError(f"Shape must be a tuple, list, or int, got {type(shape).__name__}")

    normalized: list[int] = []
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise ShapeError(f"All dimensions must be integers, but dimension {i} is {type(dim).__name__} ({dim})")
        if dim < 0:
            raise ShapeError(f"Dimension sizes must be non-negative, but dimension {i} is {dim}")
        normalized.append(dim)

    return tuple(normalized)


def validate_reshape_shape(current_numel: int, new_shape: Sequence[int]) -> Tuple[int, ...]:
    """Validate a requested shape for a reshape operation and resolve any inferred dimension (-1).

    Args:
        current_numel: Total number of elements in the source tensor.
        new_shape: Requested target shape sequence.

    Returns:
        Concrete target shape tuple with all dimensions resolved.

    Raises:
        ShapeError: If the new shape cannot accommodate `current_numel` elements or has multiple -1s.
    """
    if not isinstance(new_shape, (tuple, list)):
        raise ShapeError(f"Target shape must be a tuple or list of ints, got {type(new_shape).__name__}")

    inferred_idx = -1
    known_product = 1
    resolved_dims: list[int] = []

    for i, dim in enumerate(new_shape):
        if not isinstance(dim, int):
            raise ShapeError(f"Reshape dimensions must be integers, got {type(dim).__name__} at index {i}")
        if dim == -1:
            if inferred_idx != -1:
                raise ShapeError("Can only specify one unknown dimension (-1) in reshape")
            inferred_idx = i
            resolved_dims.append(-1)
        elif dim < 0:
            raise ShapeError(f"Invalid dimension size {dim} at index {i} in reshape")
        else:
            known_product *= dim
            resolved_dims.append(dim)

    if inferred_idx != -1:
        if known_product == 0:
            if current_numel == 0:
                resolved_dims[inferred_idx] = 0
            else:
                raise ShapeError(f"Cannot infer dimension -1 when other dimensions multiply to 0 for size {current_numel}")
        else:
            if current_numel % known_product != 0:
                raise ShapeError(
                    f"Cannot reshape tensor of size {current_numel} into shape {tuple(new_shape)}: "
                    f"size {current_numel} is not divisible by {known_product}"
                )
            resolved_dims[inferred_idx] = current_numel // known_product
    else:
        if known_product != current_numel:
            raise ShapeError(
                f"Cannot reshape tensor of size {current_numel} into shape {tuple(new_shape)}: "
                f"element count mismatch ({current_numel} != {known_product})"
            )

    return tuple(resolved_dims)


def validate_broadcast_shapes(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute and validate the result shape of broadcasting two tensor shapes.

    Args:
        shape_a: Shape of the first operand.
        shape_b: Shape of the second operand.

    Returns:
        Resulting broadcast shape tuple.

    Raises:
        ShapeError: If shapes are not broadcastable according to standard broadcasting rules.
    """
    len_a = len(shape_a)
    len_b = len(shape_b)
    max_len = max(len_a, len_b)

    # Pad shapes with 1s on the left
    padded_a = (1,) * (max_len - len_a) + shape_a
    padded_b = (1,) * (max_len - len_b) + shape_b

    result_shape: list[int] = []
    for dim_a, dim_b in zip(padded_a, padded_b):
        if dim_a == dim_b:
            result_shape.append(dim_a)
        elif dim_a == 1:
            result_shape.append(dim_b)
        elif dim_b == 1:
            result_shape.append(dim_a)
        else:
            raise ShapeError(
                f"Cannot broadcast shapes {shape_a} and {shape_b}: "
                f"incompatible dimensions {dim_a} and {dim_b}"
            )

    return tuple(result_shape)


def validate_matmul_shapes(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> Tuple[int, ...]:
    """Validate shapes for matrix multiplication (@) and return the output shape.

    Supports:
    - 1D @ 1D: Dot product -> 0D scalar
    - 2D @ 2D: (M, K) @ (K, N) -> (M, N)
    - 1D @ 2D: (K,) @ (K, N) -> (N,)
    - 2D @ 1D: (M, K) @ (K,) -> (M,)
    - Batched: (..., M, K) @ (..., K, N) -> (..., M, N)

    Args:
        shape_a: Shape of the left operand.
        shape_b: Shape of the right operand.

    Returns:
        Output shape of the matrix multiplication.

    Raises:
        DimensionError: If shapes are incompatible for matrix multiplication.
    """
    ndim_a = len(shape_a)
    ndim_b = len(shape_b)

    if ndim_a == 0 or ndim_b == 0:
        raise DimensionError(
            f"Matrix multiplication requires tensors of at least 1 dimension, got {ndim_a}D and {ndim_b}D"
        )

    # 1D @ 1D: vector dot product
    if ndim_a == 1 and ndim_b == 1:
        if shape_a[0] != shape_b[0]:
            raise DimensionError(
                f"Incompatible vector dimensions for dot product: {shape_a[0]} vs {shape_b[0]}"
            )
        return ()

    # 1D @ 2D: prepends 1 to left shape, removes it from output
    if ndim_a == 1 and ndim_b == 2:
        k_a = shape_a[0]
        k_b, n = shape_b
        if k_a != k_b:
            raise DimensionError(
                f"Incompatible matrix multiplication dimensions: left vector size {k_a} does not match right matrix rows {k_b}"
            )
        return (n,)

    # 2D @ 1D: appends 1 to right shape, removes it from output
    if ndim_a == 2 and ndim_b == 1:
        m, k_a = shape_a
        k_b = shape_b[0]
        if k_a != k_b:
            raise DimensionError(
                f"Incompatible matrix multiplication dimensions: left matrix columns {k_a} does not match right vector size {k_b}"
            )
        return (m,)

    # 2D @ 2D or Batched matrix multiplication
    if shape_a[-1] != shape_b[-2]:
        raise DimensionError(
            f"Incompatible inner matrix dimensions for matmul: {shape_a[-1]} != {shape_b[-2]} "
            f"(shapes {shape_a} and {shape_b})"
        )

    # Validate and broadcast batch dimensions (all leading dimensions except last 2)
    batch_a = shape_a[:-2]
    batch_b = shape_b[:-2]
    broadcast_batch = validate_broadcast_shapes(batch_a, batch_b)

    return broadcast_batch + (shape_a[-2], shape_b[-1])


def validate_axis(axis: int, ndim: int) -> int:
    """Validate and normalize an axis index to [0, ndim - 1].

    Args:
        axis: Specified axis index (supports negative indexing).
        ndim: Number of dimensions in the tensor.

    Returns:
        Normalized non-negative axis index.

    Raises:
        DimensionError: If axis is out of the valid range [-ndim, ndim - 1].
    """
    if ndim == 0:
        raise DimensionError("Cannot specify axis on a 0-dimensional scalar tensor")

    orig_axis = axis
    if axis < 0:
        axis += ndim

    if axis < 0 or axis >= ndim:
        raise DimensionError(
            f"Axis {orig_axis} is out of bounds for tensor with {ndim} dimension(s) "
            f"(valid range: [{-ndim}, {ndim - 1}])"
        )

    return axis


def validate_transpose_axes(ndim: int, axes: Sequence[int] | None) -> Tuple[int, ...]:
    """Validate and normalize permutation axes for a transpose operation.

    Args:
        ndim: Number of dimensions of the tensor.
        axes: Permutation of axis indices, or None for reverse permutation.

    Returns:
        Validated permutation tuple of length `ndim`.

    Raises:
        DimensionError: If axes do not form a valid permutation of range(ndim).
    """
    if axes is None:
        return tuple(reversed(range(ndim)))

    if len(axes) != ndim:
        raise DimensionError(
            f"Axes length ({len(axes)}) must match tensor dimensions ({ndim})"
        )

    normalized_axes: list[int] = []
    seen: set[int] = set()

    for ax in axes:
        if not isinstance(ax, int):
            raise DimensionError(f"Axis entries must be integers, got {type(ax).__name__}")
        norm_ax = validate_axis(ax, ndim)
        if norm_ax in seen:
            raise DimensionError(f"Duplicate axis {ax} in transpose permutation {axes}")
        seen.add(norm_ax)
        normalized_axes.append(norm_ax)

    return tuple(normalized_axes)
