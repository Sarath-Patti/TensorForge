"""Numerical gradient checking utilities for autograd test suite.

Approximates gradients using central finite differences:
    df/dx ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
and compares with analytical autograd results.
"""

from __future__ import annotations

from typing import Callable, Sequence, Tuple
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, tensor


def numerical_gradient(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    target_idx: int,
    eps: float = 1e-4,
) -> np.ndarray:
    """Compute the numerical gradient of a scalar function with respect to inputs[target_idx].

    Args:
        func: A callable taking `inputs` as arguments and returning a 0-D scalar Tensor.
        inputs: Sequence of input Tensors.
        target_idx: Index of the input tensor with respect to which gradients are calculated.
        eps: Small finite-difference step size.

    Returns:
        NumPy array of numerical gradients matching inputs[target_idx].shape.
    """
    target = inputs[target_idx]
    orig_np = target.numpy().copy()
    grad_num = np.zeros_like(orig_np, dtype=np.float64)

    # Flatten index iteration
    it = np.nditer(orig_np, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index

        # Evaluate f(x + eps)
        pos_np = orig_np.copy()
        pos_np[idx] += eps
        pos_inputs = list(inputs)
        pos_inputs[target_idx] = tensor(pos_np, dtype=target.dtype, copy=True)
        out_pos = func(*pos_inputs).item()

        # Evaluate f(x - eps)
        neg_np = orig_np.copy()
        neg_np[idx] -= eps
        neg_inputs = list(inputs)
        neg_inputs[target_idx] = tensor(neg_np, dtype=target.dtype, copy=True)
        out_neg = func(*neg_inputs).item()

        # Central difference approximation
        grad_num[idx] = (out_pos - out_neg) / (2.0 * eps)
        it.iternext()

    return grad_num


def gradcheck(
    func: Callable[..., Tensor],
    inputs: Sequence[Tensor],
    eps: float = 1e-4,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Validate analytical autograd gradients of a scalar function against finite differences.

    Args:
        func: Scalar function func(*inputs) -> Tensor.
        inputs: Input tensors to differentiate with respect to.
        eps: Step size for central difference.
        rtol: Relative error tolerance.
        atol: Absolute error tolerance.

    Returns:
        True if all gradients match within tolerance.
    """
    # 1. Forward pass & analytical backward
    for inp in inputs:
        if inp.requires_grad:
            inp.zero_grad()

    out = func(*inputs)
    out.backward()

    # 2. Check each input requiring gradient
    for i, inp in enumerate(inputs):
        if not inp.requires_grad:
            continue

        num_grad = numerical_gradient(func, inputs, i, eps=eps)
        ana_grad = inp.grad.numpy() if inp.grad is not None else np.zeros_like(num_grad)

        np.testing.assert_allclose(
            ana_grad,
            num_grad,
            rtol=rtol,
            atol=atol,
            err_msg=f"Gradient check failed for input index {i}",
        )

    return True
