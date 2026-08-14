"""Core mathematical and structural operations for TensorForge tensors.

Implements element-wise arithmetic with broadcasting, matrix multiplication,
reductions (sum, mean), and structural transformations (reshape, transpose),
with integrated reverse-mode automatic differentiation graph construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Tuple, Union
import numpy as np

from tensorforge.tensor.dtype import DType, float32, promote_dtypes, to_dtype
from tensorforge.utils.validation import (
    DimensionError,
    IndexError_,
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
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import AddBackward
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
    result_tensor = Tensor(result_arr, dtype=out_dtype)

    if is_grad_enabled() and (t_a.requires_grad or t_b.requires_grad):
        result_tensor.grad_fn = AddBackward(t_a, t_b)
        result_tensor.requires_grad = True

    return result_tensor


def sub(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise subtraction supporting broadcasting and scalar operands.

    Args:
        a: Left operand (Tensor, int, or float).
        b: Right operand (Tensor, int, or float).

    Returns:
        A new Tensor containing the element-wise difference (a - b).
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import SubBackward
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)

    result_arr = np.subtract(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    result_tensor = Tensor(result_arr, dtype=out_dtype)

    if is_grad_enabled() and (t_a.requires_grad or t_b.requires_grad):
        result_tensor.grad_fn = SubBackward(t_a, t_b)
        result_tensor.requires_grad = True

    return result_tensor


def mul(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise multiplication supporting broadcasting and scalar operands.

    Args:
        a: Left operand (Tensor, int, or float).
        b: Right operand (Tensor, int, or float).

    Returns:
        A new Tensor containing the element-wise product.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import MulBackward
    from tensorforge.tensor.tensor import Tensor

    t_a = _ensure_tensor(a)
    t_b = _ensure_tensor(b)

    validate_broadcast_shapes(t_a.shape, t_b.shape)
    out_dtype = promote_dtypes(t_a.dtype, t_b.dtype)

    result_arr = np.multiply(
        t_a.numpy().astype(out_dtype.numpy_dtype, copy=False),
        t_b.numpy().astype(out_dtype.numpy_dtype, copy=False),
    )
    result_tensor = Tensor(result_arr, dtype=out_dtype)

    if is_grad_enabled() and (t_a.requires_grad or t_b.requires_grad):
        result_tensor.grad_fn = MulBackward(t_a, t_b)
        result_tensor.requires_grad = True

    return result_tensor


def truediv(a: Union[Tensor, float, int], b: Union[Tensor, float, int]) -> Tensor:
    """Element-wise true division supporting broadcasting and scalar operands.

    Args:
        a: Numerator (Tensor, int, or float).
        b: Denominator (Tensor, int, or float).

    Returns:
        A new Tensor containing the quotient (a / b). Floating-point result.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import DivBackward
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
    result_tensor = Tensor(result_arr, dtype=out_dtype)

    if is_grad_enabled() and (t_a.requires_grad or t_b.requires_grad):
        result_tensor.grad_fn = DivBackward(t_a, t_b)
        result_tensor.requires_grad = True

    return result_tensor


def neg(a: Tensor) -> Tensor:
    """Element-wise negation (-a).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor with negated values.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import NegBackward
    from tensorforge.tensor.tensor import Tensor

    result_arr = np.negative(a.numpy())
    result_tensor = Tensor(result_arr, dtype=a.dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = NegBackward(a)
        result_tensor.requires_grad = True

    return result_tensor


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
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import MatmulBackward
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

    if is_grad_enabled() and (a.requires_grad or b.requires_grad):
        result_tensor.grad_fn = MatmulBackward(a, b)
        result_tensor.requires_grad = True

    return result_tensor


def reshape(a: Tensor, *shape: Union[int, Sequence[int]]) -> Tensor:
    """Return a tensor with the same data reshaped to target dimensions.

    Note (v0.1+):
        In TensorForge, reshaping materializes a contiguous Tensor.

    Args:
        a: Input tensor.
        *shape: Target shape specified as separate integers or a single tuple/list.

    Returns:
        A new contiguous Tensor reshaped to target dimensions.

    Raises:
        ShapeError: If the target shape is incompatible with the number of elements.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import ReshapeBackward
    from tensorforge.tensor.tensor import Tensor

    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = tuple(shape[0])
    else:
        target_shape = tuple(shape)  # type: ignore

    resolved_shape = validate_reshape_shape(a.numel, target_shape)
    reshaped_arr = np.ascontiguousarray(a.numpy().reshape(resolved_shape))
    result_tensor = Tensor(reshaped_arr, dtype=a.dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = ReshapeBackward(a)
        result_tensor.requires_grad = True

    return result_tensor


def transpose(a: Tensor, *axes: Union[int, Sequence[int]]) -> Tensor:
    """Permute the dimensions of a tensor, returning a new contiguous Tensor.

    Args:
        a: Input tensor.
        *axes: Desired dimension order. If empty, dimensions are reversed.

    Returns:
        A new contiguous Tensor with permuted dimensions.

    Raises:
        DimensionError: If axes do not form a valid permutation of dimensions.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import TransposeBackward
    from tensorforge.tensor.tensor import Tensor

    if len(axes) == 0:
        norm_axes = None
    elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
        norm_axes = tuple(axes[0])
    else:
        norm_axes = tuple(axes)  # type: ignore

    validated_axes = validate_transpose_axes(a.ndim, norm_axes)
    transposed_arr = np.ascontiguousarray(np.transpose(a.numpy(), axes=validated_axes))
    result_tensor = Tensor(transposed_arr, dtype=a.dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = TransposeBackward(a, validated_axes)
        result_tensor.requires_grad = True

    return result_tensor


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
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import SumBackward
    from tensorforge.tensor.tensor import Tensor

    if axis is not None:
        if isinstance(axis, int):
            validated_axis: Union[int, Tuple[int, ...]] = validate_axis(axis, a.ndim)
        else:
            validated_axis = tuple(validate_axis(ax, a.ndim) for ax in axis)
    else:
        validated_axis = None

    res_arr = np.sum(a.numpy(), axis=validated_axis, keepdims=keepdims)
    result_tensor = Tensor(res_arr, dtype=a.dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = SumBackward(a, axis=validated_axis, keepdims=keepdims)
        result_tensor.requires_grad = True

    return result_tensor


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
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import MeanBackward
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
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = MeanBackward(a, axis=validated_axis, keepdims=keepdims)
        result_tensor.requires_grad = True

    return result_tensor


def exp(a: Tensor) -> Tensor:
    """Element-wise exponential: z = exp(a).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor containing exponential of elements.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import ExpBackward
    from tensorforge.tensor.tensor import Tensor

    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    res_arr = np.exp(a.numpy().astype(out_dtype.numpy_dtype, copy=False))
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = ExpBackward(a, result_tensor)
        result_tensor.requires_grad = True

    return result_tensor


def log(a: Tensor) -> Tensor:
    """Element-wise natural logarithm: z = log(a).

    Args:
        a: Input tensor (must contain positive numbers).

    Returns:
        A new Tensor containing natural log of elements.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import LogBackward
    from tensorforge.tensor.tensor import Tensor

    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    res_arr = np.log(a.numpy().astype(out_dtype.numpy_dtype, copy=False))
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = LogBackward(a)
        result_tensor.requires_grad = True

    return result_tensor


def relu(a: Tensor) -> Tensor:
    """Rectified Linear Unit activation: z = max(0, a).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor with negative values clamped to 0.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import ReluBackward
    from tensorforge.tensor.tensor import Tensor

    res_arr = np.maximum(a.numpy(), 0)
    result_tensor = Tensor(res_arr, dtype=a.dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = ReluBackward(a)
        result_tensor.requires_grad = True

    return result_tensor


def sigmoid(a: Tensor) -> Tensor:
    """Logistic sigmoid activation: z = 1 / (1 + exp(-a)).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor with sigmoid activated values in range (0, 1).
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import SigmoidBackward
    from tensorforge.tensor.tensor import Tensor

    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    arr = a.numpy().astype(out_dtype.numpy_dtype, copy=False)
    # Numerically stable logistic sigmoid
    res_arr = np.where(arr >= 0, 1.0 / (1.0 + np.exp(-arr)), np.exp(arr) / (1.0 + np.exp(arr)))
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = SigmoidBackward(a, result_tensor)
        result_tensor.requires_grad = True

    return result_tensor


def tanh(a: Tensor) -> Tensor:
    """Hyperbolic tangent activation: z = tanh(a).

    Args:
        a: Input tensor.

    Returns:
        A new Tensor with tanh activated values in range (-1, 1).
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import TanhBackward
    from tensorforge.tensor.tensor import Tensor

    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    res_arr = np.tanh(a.numpy().astype(out_dtype.numpy_dtype, copy=False))
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = TanhBackward(a, result_tensor)
        result_tensor.requires_grad = True

    return result_tensor


def softmax(a: Tensor, dim: int = -1) -> Tensor:
    """Softmax activation along specified dimension: S = exp(a_i) / sum(exp(a_j)).

    Args:
        a: Input tensor.
        dim: Dimension along which softmax is computed (default: -1).

    Returns:
        A new Tensor with normalized probabilities summing to 1 along `dim`.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import SoftmaxBackward
    from tensorforge.tensor.tensor import Tensor

    norm_dim = validate_axis(dim, a.ndim)
    out_dtype = a.dtype if a.dtype.is_floating_point else float32
    arr = a.numpy().astype(out_dtype.numpy_dtype, copy=False)

    # Subtract max for numerical stability
    max_val = np.max(arr, axis=norm_dim, keepdims=True)
    exp_arr = np.exp(arr - max_val)
    sum_exp = np.sum(exp_arr, axis=norm_dim, keepdims=True)
    res_arr = exp_arr / sum_exp
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = SoftmaxBackward(a, result_tensor, dim=norm_dim)
        result_tensor.requires_grad = True

    return result_tensor


def pow(a: Tensor, exponent: Union[float, int]) -> Tensor:
    """Element-wise power: z = a^exponent.

    Args:
        a: Input tensor.
        exponent: Scalar power.

    Returns:
        A new Tensor with powered elements.
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import PowBackward
    from tensorforge.tensor.tensor import Tensor

    out_dtype = a.dtype if isinstance(exponent, int) else float32
    res_arr = np.power(a.numpy(), exponent)
    result_tensor = Tensor(res_arr, dtype=out_dtype)

    if is_grad_enabled() and a.requires_grad:
        result_tensor.grad_fn = PowBackward(a, exponent)
        result_tensor.requires_grad = True

    return result_tensor


def cross_entropy(
    logits: Tensor,
    targets: Union[Tensor, Sequence[int], np.ndarray],
    reduction: str = "mean",
) -> Tensor:
    """Compute cross-entropy loss between unnormalized logits and class index targets.

    Args:
        logits: Unnormalized class scores of shape (N, C) or (C,).
        targets: Ground truth class integer indices of shape (N,) or scalar.
        reduction: 'mean', 'sum', or 'none'.

    Returns:
        Scalar loss Tensor (if reduction is 'mean' or 'sum') or 1D Tensor (if 'none').
    """
    from tensorforge.autograd.engine import is_grad_enabled
    from tensorforge.autograd.function import CrossEntropyBackward
    from tensorforge.tensor.tensor import Tensor

    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"Invalid reduction '{reduction}'; must be 'mean', 'sum', or 'none'")

    logits_np = logits.numpy()
    is_1d = (logits_np.ndim == 1)
    if is_1d:
        logits_np = logits_np.reshape(1, -1)

    if isinstance(targets, Tensor):
        targets_np = targets.numpy().astype(np.int64)
    elif isinstance(targets, np.ndarray):
        targets_np = targets.astype(np.int64)
    else:
        targets_np = np.array(targets, dtype=np.int64)

    if targets_np.ndim == 0:
        targets_np = targets_np.reshape(1)

    N, C = logits_np.shape
    if targets_np.shape[0] != N:
        raise DimensionError(
            f"CrossEntropy target batch size ({targets_np.shape[0]}) does not match logits batch size ({N})"
        )

    if np.any(targets_np < 0) or np.any(targets_np >= C):
        raise IndexError_(
            f"Target class index out of bounds: expected in range [0, {C - 1}], got {targets_np}"
        )

    # Stable log-softmax computation
    max_logits = np.max(logits_np, axis=-1, keepdims=True)
    shifted = logits_np - max_logits
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)
    probs = exp_shifted / sum_exp

    # NLL loss for target indices
    sample_losses = -log_probs[np.arange(N), targets_np]

    out_dtype = logits.dtype if logits.dtype.is_floating_point else float32

    if reduction == "mean":
        loss_val = np.mean(sample_losses)
        res_tensor = Tensor(loss_val, dtype=out_dtype)
    elif reduction == "sum":
        loss_val = np.sum(sample_losses)
        res_tensor = Tensor(loss_val, dtype=out_dtype)
    else:  # none
        if is_1d:
            sample_losses = sample_losses.reshape(())
        res_tensor = Tensor(sample_losses, dtype=out_dtype)

    if is_grad_enabled() and logits.requires_grad:
        res_tensor.grad_fn = CrossEntropyBackward(
            logits, probs, targets_np, reduction=reduction, is_1d=is_1d
        )
        res_tensor.requires_grad = True

    return res_tensor


def argmax(a: Tensor, axis: int = -1, keepdims: bool = False) -> Tensor:
    """Return indices of the maximum values along the specified axis.

    Args:
        a: Input tensor.
        axis: Axis along which to find maximum indices (default: -1).
        keepdims: If True, the reduced axis is retained with length 1.

    Returns:
        Integer Tensor containing argmax indices.
    """
    from tensorforge.tensor.dtype import int64
    from tensorforge.tensor.tensor import Tensor

    norm_axis = validate_axis(axis, a.ndim)
    res_arr = np.argmax(a.numpy(), axis=norm_axis, keepdims=keepdims)
    return Tensor(res_arr, dtype=int64, copy=False, requires_grad=False)


