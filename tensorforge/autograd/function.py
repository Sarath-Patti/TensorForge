"""Computation graph nodes and backward differentiation rules for TensorForge.

Defines the base Node abstraction for reverse-mode automatic differentiation,
gradient unbroadcasting utilities, and operation-specific backward rules.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.tensor.shape import compute_numel

if TYPE_CHECKING:
    from tensorforge.tensor.tensor import Tensor


def reduce_gradient_to_shape(grad: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    """Reduce a broadcasted gradient tensor back to the target operand's shape.

    Summates across dimensions that were expanded or prepended during forward broadcasting.

    Args:
        grad: Incoming gradient tensor of broadcasted shape.
        target_shape: Original shape of the operand that requires the gradient.

    Returns:
        Gradient tensor reduced along broadcast dimensions, matching target_shape.
    """
    from tensorforge.tensor.tensor import Tensor

    if grad.shape == target_shape:
        return grad

    # Scalar target ()
    if not target_shape:
        return grad.sum().reshape(())

    grad_shape = grad.shape
    ndim_grad = len(grad_shape)
    ndim_target = len(target_shape)

    current_grad = grad

    # 1. Sum across prepended leading dimensions (when grad has more dimensions than target)
    leading_dims = ndim_grad - ndim_target
    if leading_dims > 0:
        current_grad = current_grad.sum(axis=tuple(range(leading_dims)), keepdims=False)

    # 2. Sum along dimensions where target_shape is 1 but grad has expanded to > 1
    axes_to_keepdim_sum: list[int] = []
    for i, (g_dim, t_dim) in enumerate(zip(current_grad.shape, target_shape)):
        if t_dim == 1 and g_dim > 1:
            axes_to_keepdim_sum.append(i)

    if axes_to_keepdim_sum:
        current_grad = current_grad.sum(axis=tuple(axes_to_keepdim_sum), keepdims=True)

    if current_grad.shape != target_shape:
        current_grad = current_grad.reshape(target_shape)

    return current_grad


class Node(ABC):
    """Abstract base class for a backward operation node in the computational DAG.

    Attributes:
        name: Name of the backward operation (e.g. 'AddBackward', 'MulBackward').
        parents: Input tensors that produced the output of this node.
        needs_input_grad: Boolean flags indicating which parent tensors require gradients.
    """

    def __init__(self, name: str, parents: Sequence[Tensor]) -> None:
        self.name: str = name
        self.parents: Tuple[Tensor, ...] = tuple(parents)
        self.needs_input_grad: Tuple[bool, ...] = tuple(
            isinstance(p, object) and getattr(p, "requires_grad", False) for p in parents
        )

    @abstractmethod
    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        """Compute gradients with respect to each parent input tensor.

        Args:
            grad_output: Upstream gradient flowing into this node.

        Returns:
            Tuple of gradient tensors corresponding to each input in `self.parents`.
            Entries for inputs that do not require grad may be None.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"


# -----------------------------------------------------------------------------
# Concrete Backward Nodes
# -----------------------------------------------------------------------------

class AddBackward(Node):
    """Backward rule for addition: z = a + b -> dz/da = 1, dz/db = 1."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__("AddBackward", (a, b))
        self.shape_a: Tuple[int, ...] = a.shape
        self.shape_b: Tuple[int, ...] = b.shape

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_a = reduce_gradient_to_shape(grad_output, self.shape_a) if self.needs_input_grad[0] else None
        grad_b = reduce_gradient_to_shape(grad_output, self.shape_b) if self.needs_input_grad[1] else None
        return (grad_a, grad_b)


class SubBackward(Node):
    """Backward rule for subtraction: z = a - b -> dz/da = 1, dz/db = -1."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__("SubBackward", (a, b))
        self.shape_a: Tuple[int, ...] = a.shape
        self.shape_b: Tuple[int, ...] = b.shape

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_a = reduce_gradient_to_shape(grad_output, self.shape_a) if self.needs_input_grad[0] else None
        grad_b = reduce_gradient_to_shape(-grad_output, self.shape_b) if self.needs_input_grad[1] else None
        return (grad_a, grad_b)


class MulBackward(Node):
    """Backward rule for element-wise multiplication: z = a * b -> dz/da = b, dz/db = a."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__("MulBackward", (a, b))
        self.saved_a: Tensor = a.detach()
        self.saved_b: Tensor = b.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_a = (
            reduce_gradient_to_shape(grad_output * self.saved_b, self.saved_a.shape)
            if self.needs_input_grad[0]
            else None
        )
        grad_b = (
            reduce_gradient_to_shape(grad_output * self.saved_a, self.saved_b.shape)
            if self.needs_input_grad[1]
            else None
        )
        return (grad_a, grad_b)


class DivBackward(Node):
    """Backward rule for division: z = a / b -> dz/da = 1/b, dz/db = -a / b^2."""

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__("DivBackward", (a, b))
        self.saved_a: Tensor = a.detach()
        self.saved_b: Tensor = b.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_a = (
            reduce_gradient_to_shape(grad_output / self.saved_b, self.saved_a.shape)
            if self.needs_input_grad[0]
            else None
        )
        grad_b = (
            reduce_gradient_to_shape(-grad_output * self.saved_a / (self.saved_b * self.saved_b), self.saved_b.shape)
            if self.needs_input_grad[1]
            else None
        )
        return (grad_a, grad_b)


class NegBackward(Node):
    """Backward rule for negation: z = -a -> dz/da = -1."""

    def __init__(self, a: Tensor) -> None:
        super().__init__("NegBackward", (a,))

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        grad_a = -grad_output if self.needs_input_grad[0] else None
        return (grad_a,)


class MatmulBackward(Node):
    """Backward rule for matrix multiplication: C = A @ B.

    Supports 1D dot products, 2D matrix multiplication, and batched matrix multiplication.
    """

    def __init__(self, a: Tensor, b: Tensor) -> None:
        super().__init__("MatmulBackward", (a, b))
        self.saved_a: Tensor = a.detach()
        self.saved_b: Tensor = b.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        a = self.saved_a
        b = self.saved_b
        ndim_a = a.ndim
        ndim_b = b.ndim

        grad_a: Optional[Tensor] = None
        grad_b: Optional[Tensor] = None

        # Case 1: 1D @ 1D (dot product -> scalar output)
        if ndim_a == 1 and ndim_b == 1:
            if self.needs_input_grad[0]:
                grad_a = grad_output * b
            if self.needs_input_grad[1]:
                grad_b = grad_output * a
            return (grad_a, grad_b)

        # Case 2: 1D @ 2D: A (K,) @ B (K, N) -> C (N,)
        if ndim_a == 1 and ndim_b == 2:
            if self.needs_input_grad[0]:
                # grad_output (N,) -> (1, N) @ B.T (N, K) -> (1, K) -> (K,)
                g_reshaped = grad_output.reshape(1, -1)
                grad_a = (g_reshaped @ b.T).reshape(a.shape)
            if self.needs_input_grad[1]:
                # A (K,) -> (K, 1) @ grad_output (1, N) -> (K, N)
                a_reshaped = a.reshape(-1, 1)
                g_reshaped = grad_output.reshape(1, -1)
                grad_b = a_reshaped @ g_reshaped
            return (grad_a, grad_b)

        # Case 3: 2D @ 1D: A (M, K) @ B (K,) -> C (M,)
        if ndim_a == 2 and ndim_b == 1:
            if self.needs_input_grad[0]:
                # grad_output (M,) -> (M, 1) @ B (1, K) -> (M, K)
                g_reshaped = grad_output.reshape(-1, 1)
                b_reshaped = b.reshape(1, -1)
                grad_a = g_reshaped @ b_reshaped
            if self.needs_input_grad[1]:
                # A.T (K, M) @ grad_output (M, 1) -> (K, 1) -> (K,)
                g_reshaped = grad_output.reshape(-1, 1)
                grad_b = (a.T @ g_reshaped).reshape(b.shape)
            return (grad_a, grad_b)

        # Case 4: 2D @ 2D: A (M, K) @ B (K, N) -> C (M, N)
        if ndim_a == 2 and ndim_b == 2:
            if self.needs_input_grad[0]:
                grad_a = grad_output @ b.T
            if self.needs_input_grad[1]:
                grad_b = a.T @ grad_output
            return (grad_a, grad_b)

        # Case 5: Batched Matrix Multiplication (>= 3D)
        # Transpose the last two dimensions of a and b
        axes_a = list(range(ndim_a))
        axes_a[-1], axes_a[-2] = axes_a[-2], axes_a[-1]
        a_t = a.transpose(*axes_a)

        axes_b = list(range(ndim_b))
        axes_b[-1], axes_b[-2] = axes_b[-2], axes_b[-1]
        b_t = b.transpose(*axes_b)

        if self.needs_input_grad[0]:
            raw_grad_a = grad_output @ b_t
            grad_a = reduce_gradient_to_shape(raw_grad_a, a.shape)

        if self.needs_input_grad[1]:
            raw_grad_b = a_t @ grad_output
            grad_b = reduce_gradient_to_shape(raw_grad_b, b.shape)

        return (grad_a, grad_b)


class SumBackward(Node):
    """Backward rule for sum reduction: broadcasts upstream gradient across reduced axes."""

    def __init__(
        self,
        a: Tensor,
        axis: Union[int, Sequence[int], None] = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__("SumBackward", (a,))
        self.input_shape: Tuple[int, ...] = a.shape
        self.input_dtype = a.dtype
        self.axis: Union[int, Sequence[int], None] = axis
        self.keepdims: bool = keepdims

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)

        from tensorforge.tensor.tensor import ones

        if self.axis is None:
            # Full reduction to scalar
            ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
            grad_a = grad_output.reshape(()) * ones_tensor
            return (grad_a,)

        if self.keepdims:
            # Output already had matching rank with size 1 along reduced axes
            ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
            grad_a = grad_output * ones_tensor
            return (grad_a,)

        # If keepdims=False, restore reduced dimensions as 1 before broadcasting
        norm_axes: list[int]
        if isinstance(self.axis, int):
            norm_axis = self.axis if self.axis >= 0 else self.axis + len(self.input_shape)
            norm_axes = [norm_axis]
        else:
            norm_axes = [ax if ax >= 0 else ax + len(self.input_shape) for ax in self.axis]

        expanded_shape: list[int] = list(self.input_shape)
        for ax in norm_axes:
            expanded_shape[ax] = 1

        reshaped_grad = grad_output.reshape(tuple(expanded_shape))
        ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
        grad_a = reshaped_grad * ones_tensor
        return (grad_a,)


class MeanBackward(Node):
    """Backward rule for mean reduction: scales upstream gradient by 1/N and broadcasts."""

    def __init__(
        self,
        a: Tensor,
        axis: Union[int, Sequence[int], None] = None,
        keepdims: bool = False,
    ) -> None:
        super().__init__("MeanBackward", (a,))
        self.input_shape: Tuple[int, ...] = a.shape
        self.input_dtype = a.dtype
        self.axis: Union[int, Sequence[int], None] = axis
        self.keepdims: bool = keepdims

        if axis is None:
            self.count: float = float(a.numel)
        else:
            if isinstance(axis, int):
                norm_axes = [axis if axis >= 0 else axis + len(a.shape)]
            else:
                norm_axes = [ax if ax >= 0 else ax + len(a.shape) for ax in axis]
            count = 1
            for ax in norm_axes:
                count *= a.shape[ax]
            self.count = float(count)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)

        from tensorforge.tensor.tensor import ones

        scaled_grad = grad_output * (1.0 / self.count)

        if self.axis is None:
            ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
            grad_a = scaled_grad.reshape(()) * ones_tensor
            return (grad_a,)

        if self.keepdims:
            ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
            grad_a = scaled_grad * ones_tensor
            return (grad_a,)

        norm_axes: list[int]
        if isinstance(self.axis, int):
            norm_axis = self.axis if self.axis >= 0 else self.axis + len(self.input_shape)
            norm_axes = [norm_axis]
        else:
            norm_axes = [ax if ax >= 0 else ax + len(self.input_shape) for ax in self.axis]

        expanded_shape: list[int] = list(self.input_shape)
        for ax in norm_axes:
            expanded_shape[ax] = 1

        reshaped_grad = scaled_grad.reshape(tuple(expanded_shape))
        ones_tensor = ones(self.input_shape, dtype=self.input_dtype)
        grad_a = reshaped_grad * ones_tensor
        return (grad_a,)


class ReshapeBackward(Node):
    """Backward rule for reshape: reshapes upstream gradient back to input shape."""

    def __init__(self, a: Tensor) -> None:
        super().__init__("ReshapeBackward", (a,))
        self.input_shape: Tuple[int, ...] = a.shape

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output.reshape(self.input_shape),)


class TransposeBackward(Node):
    """Backward rule for transpose: applies the inverse axis permutation to upstream gradient."""

    def __init__(self, a: Tensor, axes: Optional[Sequence[int]] = None) -> None:
        super().__init__("TransposeBackward", (a,))
        self.ndim: int = a.ndim
        if axes is None:
            self.inv_axes: Optional[Tuple[int, ...]] = None
        else:
            norm_axes = [ax if ax >= 0 else ax + self.ndim for ax in axes]
            inv = [0] * len(norm_axes)
            for i, ax in enumerate(norm_axes):
                inv[ax] = i
            self.inv_axes = tuple(inv)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)

        if self.inv_axes is None:
            return (grad_output.transpose(),)

        return (grad_output.transpose(*self.inv_axes),)


class ExpBackward(Node):
    """Backward rule for exponential: z = exp(a) -> dz/da = exp(a)."""

    def __init__(self, a: Tensor, out: Tensor) -> None:
        super().__init__("ExpBackward", (a,))
        self.saved_out: Tensor = out.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output * self.saved_out,)


class LogBackward(Node):
    """Backward rule for natural logarithm: z = log(a) -> dz/da = 1 / a."""

    def __init__(self, a: Tensor) -> None:
        super().__init__("LogBackward", (a,))
        self.saved_a: Tensor = a.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        return (grad_output / self.saved_a,)


class ReluBackward(Node):
    """Backward rule for rectified linear unit: z = relu(a) -> dz/da = 1 if a > 0 else 0."""

    def __init__(self, a: Tensor) -> None:
        super().__init__("ReluBackward", (a,))
        self.saved_a: Tensor = a.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        from tensorforge.tensor.tensor import Tensor

        mask = (self.saved_a.numpy() > 0).astype(grad_output.dtype.numpy_dtype)
        res_arr = grad_output.numpy() * mask
        return (Tensor(res_arr, dtype=grad_output.dtype),)


class SigmoidBackward(Node):
    """Backward rule for sigmoid: z = sigmoid(a) -> dz/da = z * (1 - z)."""

    def __init__(self, a: Tensor, out: Tensor) -> None:
        super().__init__("SigmoidBackward", (a,))
        self.saved_out: Tensor = out.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        from tensorforge.tensor.tensor import Tensor

        s = self.saved_out.numpy()
        ds = s * (1.0 - s)
        res_arr = grad_output.numpy() * ds
        return (Tensor(res_arr, dtype=grad_output.dtype),)


class TanhBackward(Node):
    """Backward rule for hyperbolic tangent: z = tanh(a) -> dz/da = 1 - z^2."""

    def __init__(self, a: Tensor, out: Tensor) -> None:
        super().__init__("TanhBackward", (a,))
        self.saved_out: Tensor = out.detach()

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        from tensorforge.tensor.tensor import Tensor

        t = self.saved_out.numpy()
        dt = 1.0 - t * t
        res_arr = grad_output.numpy() * dt
        return (Tensor(res_arr, dtype=grad_output.dtype),)


class SoftmaxBackward(Node):
    """Backward rule for softmax: S = softmax(a) -> dz/da = S * (grad - sum(S * grad))."""

    def __init__(self, a: Tensor, out: Tensor, dim: int = -1) -> None:
        super().__init__("SoftmaxBackward", (a,))
        self.saved_out: Tensor = out.detach()
        self.dim: int = dim if dim >= 0 else dim + a.ndim

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        s = self.saved_out
        sum_term = (grad_output * s).sum(axis=self.dim, keepdims=True)
        return (s * (grad_output - sum_term),)


class PowBackward(Node):
    """Backward rule for scalar power: z = a^p -> dz/da = p * a^(p - 1)."""

    def __init__(self, a: Tensor, exponent: Union[float, int]) -> None:
        super().__init__("PowBackward", (a,))
        self.saved_a: Tensor = a.detach()
        self.exponent: float = float(exponent)

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        from tensorforge.tensor.tensor import Tensor

        arr_a = self.saved_a.numpy()
        d_power = self.exponent * np.power(arr_a, self.exponent - 1.0)
        res_arr = grad_output.numpy() * d_power
        return (Tensor(res_arr, dtype=grad_output.dtype),)


class CrossEntropyBackward(Node):
    """Backward rule for multiclass cross-entropy loss with class indices."""

    def __init__(
        self,
        logits: Tensor,
        probs: np.ndarray,
        targets: np.ndarray,
        reduction: str = "mean",
        is_1d: bool = False,
    ) -> None:
        super().__init__("CrossEntropyBackward", (logits,))
        self.probs: np.ndarray = probs  # (N, C)
        self.targets: np.ndarray = targets  # (N,)
        self.reduction: str = reduction
        self.is_1d: bool = is_1d
        self.orig_shape: Tuple[int, ...] = logits.shape
        self.orig_dtype = logits.dtype

    def backward(self, grad_output: Tensor) -> Tuple[Optional[Tensor], ...]:
        if not self.needs_input_grad[0]:
            return (None,)
        from tensorforge.tensor.tensor import Tensor

        num_samples = self.probs.shape[0]
        grad_np = self.probs.copy()  # (N, C)

        # Subtract 1 at target class index
        for i, target_cls in enumerate(self.targets):
            if 0 <= target_cls < grad_np.shape[1]:
                grad_np[i, target_cls] -= 1.0

        if self.reduction == "mean":
            grad_np /= float(num_samples)
        elif self.reduction == "none":
            # If reduction='none', grad_output has shape (N,)
            g_out_np = grad_output.numpy().reshape(-1, 1)
            grad_np *= g_out_np

        if self.reduction in ("mean", "sum"):
            # grad_output is scalar
            scale = grad_output.item()
            grad_np *= scale

        if self.is_1d:
            grad_np = grad_np.reshape(self.orig_shape)

        return (Tensor(grad_np, dtype=self.orig_dtype),)

