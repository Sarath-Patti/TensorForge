"""Core mathematical and structural operations for TensorForge tensors.

Implements element-wise arithmetic with broadcasting, matrix multiplication,
reductions (sum, mean), and structural transformations (reshape, transpose).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Tuple, Union
import numpy as np

from tensorforge.tensor.dtype import DType, float32, promote_dtypes, to_dtype
from tensorforge.utils.validation import (
    DimensionError,
    ShapeError,
    validate_axis,
    validate_broadcast_shapes,
    validate_matmul_shapes,
    validate_reshape_shape,
    validate_transpose_axes,
)

if TYPE_CHECKING:
    from tensorforge.tensor.tensor import Tensor


def _ensure_tensor(val: Any) -> Tensor:
    """Helper to convert scalar or array-like to Tensor if not already a Tensor."""
    from tensorforge.tensor.tensor import Tensor, tensor

    if isinstance(val, Tensor):
        return val
    return tensor(val)


def add(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise addition supporting broadcasting and scalar operands.

    Args:
        a: Left operand (Tensor, int, or float).
        b: Right operand (Tensor, int, or float).

    Returns:
        A new Tensor containing the element-wise sum.
    """
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    # Validate broadcastability
    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)

    result_arr = np.add(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    return Tensor(result_arr, dtype=out_dtype)


def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise subtraction supporting broadcasting and scalar operands.

    Args:
        a: Left operand (Tensor, int, or float).
        b: Right operand (Tensor, int, or float).

    Returns:
        A new Tensor containing the element-wise difference (a - b).
    """
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)

    result_arr = np.subtract(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    return Tensor(result_arr, dtype=out_dtype)


def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise multiplication supporting broadcasting and scalar operands.

    Args:
        a: Left operand (Tensor, int, or float).
        b: Right operand (Tensor, int, or float).

    Returns:
        A new Tensor containing the element-wise product.
    """
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)

    result_arr = np.multiply(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    return Tensor(result_arr, dtype=out_dtype)


def truediv(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise true division supporting broadcasting and scalar operands.

    Args:
        a: Numerator (Tensor, int, or float).
        b: Denominator (Tensor, int, or float).

    Returns:
        A new Tensor containing the quotient (a / b). Floating-point result.
    """
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)
    if not out_dtype.is_floating_point:
        out_dtype = float32

    result_arr = np.true_divide(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    return Tensor(result_arr, dtype=out_dtype)


def neg(a: Tensor) -> Tensor:
    """Element-wise negation (-a).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor with negated values.
    """
    from tensorforge.tensor.tensor import Tensor

    result_arr = np.negative(a.numpy())
    return Tensor(result_arr, dtype=a.dtype)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    """Matrix multiplication (@) of two tensors.

    Supports:
    - 1D @ 1D -> scalar dot product
    - 2D @ 2D -> (M, K) @ (K, N) -> (M, N)
    - 1D @ 2D -> (K,) @ (K, N) -> (N,)
    - 2D @ 1D -> (M, K) @ (K,) -> (M,)
    - Batched: (..., M, K) @ (..., K, N) -> (..., M, N)

    Args:
        a: Left tensor operand.
        b: Right tensor operand.

    Returns:
        Resulting tensor product.

    Raises:
        DimensionError: If matrix shapes are incompatible.
    """
    from tensorforge.tensor.tensor import Tensor

    # Validate shapes and compute expected output shape
    expected_shape = validate_matmul_shapes(a.shape, b.shape)
    out_dtype = promote_dtypes(a.dtype, b.dtype)

    arr_a = a.numpy().astype(out_dtype.numpy_dtype, copy=False)
    arr_b = b.numpy().astype(out_dtype.numpy_dtype, copy=False)

    result_arr = np.matmul(arr_a, arr_b)
    result_tensor = Tensor(result_arr, dtype=out_dtype)

    # Double check resulting shape matches validated shape
    if result_tensor.shape != expected_shape:
        result_tensor = result_tensor.reshape(expected_shape)

    return result_tensor


def reshape(a: Tensor, *shape: Union[int, Sequence[int]]) -> Tensor:
    """Return a tensor with the same data reshaped to target dimensions.

    Note (v0.1):
        In v0.1, TensorForge enforces a contiguous storage model. Reshaping
        materializes a new contiguous Tensor.

    Args:
        a: Input tensor.
        *shape: Target shape specified as separate integers or a single tuple/list.

    Returns:
        A new contiguous Tensor reshaped to target dimensions.

    Raises:
        ShapeError: If the target shape is incompatible with the number of elements.
    """
    from tensorforge.tensor.tensor import Tensor

    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)  # type: ignore

    resolved_shape = validate_reshape_shape(a.numel, target_shape)
    reshaped_arr = np.ascontiguousarray(a.numpy().reshape(resolved_shape))
    return Tensor(reshaped_arr, dtype=a.dtype)


def transpose(a: Tensor, *axes: Union[int, Sequence[int]]) -> Tensor:
    """Permute the dimensions of a tensor, returning a new contiguous Tensor.

    Note (v0.1):
        In v0.1, TensorForge operates exclusively on contiguous memory.
        This operation materializes a new contiguous Tensor rather than
        creating a non-contiguous view.

    Args:
        a: Input tensor.
        *axes: Desired dimension order. If empty, dimensions are reversed.

    Returns:
        A new contiguous Tensor with permuted dimensions.

    Raises:
        DimensionError: If axes do not form a valid permutation of dimensions.
    """
    from tensorforge.tensor.tensor import Tensor

    if len(axes) == 0:
        norm_axes = None
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        norm_axes = tuple(axes[0])
    else:
        norm_axes = tuple(axes)  # type: ignore

    validated_axes = validate_transpose_axes(a.ndim, norm_axes)
    transposed_arr = np.ascontiguousarray(np.transpose(a.numpy(), axes=validated_axes))
    return Tensor(transposed_arr, dtype=a.dtype)


def sum(
    a: Tensor,
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the sum of tensor elements over given axis/axes.

    Args:
        a: Input tensor.
        axis: Axis or axes along which to sum. If None, sums all elements.
        keepdims: If True, retained reduced dimensions with size 1.

    Returns:
        Reduced Tensor with sum values.

    Raises:
        DimensionError: If axis is out of valid bounds.
    """
    from tensorforge.tensor.tensor import Tensor

    if axis is not None:
        if isinstance(axis, int):
            validated_axis: Union[int, Tuple[int, ...]] = validate_axis(axis, a.ndim)
        else:
            validated_axis = tuple(validate_axis(ax, a.ndim) for ax in axis)
    else:
        validated_axis = None

    res_arr = np.sum(a.numpy(), axis=validated_axis, keepdims=keepdims)
    return Tensor(res_arr, dtype=a.dtype)


def mean(
    a: Tensor,
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False,
) -> Tensor:
    """Compute the arithmetic mean of tensor elements over given axis/axes.

    Args:
        a: Input tensor.
        axis: Axis or axes along which to compute the mean. If None, reduces all elements.
        keepdims: If True, retained reduced dimensions with size 1.

    Returns:
        Reduced Tensor with mean values (floating-point dtype).

    Raises:
        DimensionError: If axis is out of valid bounds.
    """
    from tensorforge.tensor.tensor import Tensor

    if a.numel == 0:
        raise DimensionError("Cannot compute mean of empty tensor with 0 elements")

    if axis is not None:
        if isinstance(axis, int):
            validated_axis: Union[int, Tuple[int, ...]] = validate_axis(axis, a.ndim)
        else:
            validated_axis = tuple(validate_axis(ax, a.ndim) for ax in axis)
    else:
        validated_axis = None

    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    res_arr = np.mean(a.numpy(), axis=validated_axis, keepdims=keepdims, dtype=out_dtype.numpy_dtype)
    return Tensor(res_arr, dtype=out_dtype)
