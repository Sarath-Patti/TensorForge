"""Quantization, dequantization, and INT8 compute operations."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from tensorforge.backend.dispatcher import get_backend, set_last_backend
from tensorforge.backend.native_backend import can_native_qmatmul, native_qmatmul
from tensorforge.quantization.calibration import calibrate_tensor
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import float32, int8
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import QuantizationError, ShapeError


def quantize_tensor(
    tensor: Union[Tensor, np.ndarray],
    scale: float,
    zero_point: int = 0,
    scheme: str = "symmetric",
) -> QuantizedTensor:
    """Quantize floating-point tensor data to INT8 using explicit scale and zero_point.

    Formula:
        q = clamp(round(x / scale) + zero_point, -128, 127)

    Args:
        tensor: Floating-point Tensor or array to quantize.
        scale: Positive quantization scaling factor.
        zero_point: Integer quantization zero-point offset.
        scheme: Quantization scheme ('symmetric' or 'asymmetric').

    Returns:
        QuantizedTensor instance holding contiguous physical INT8 storage.
    """
    if scale <= 0.0:
        raise QuantizationError(f"Scale must be strictly positive, got: {scale}")

    if isinstance(tensor, Tensor):
        fp_arr = tensor.numpy().astype(np.float32)
        orig_dtype = tensor.dtype
        orig_shape = tensor.shape
    else:
        fp_arr = np.asarray(tensor, dtype=np.float32)
        orig_dtype = float32
        orig_shape = fp_arr.shape

    inv_scale = 1.0 / scale
    q_unclamped = np.round(fp_arr * inv_scale) + float(zero_point)
    q_clamped = np.clip(q_unclamped, -128, 127).astype(np.int8)

    return QuantizedTensor(
        qdata=q_clamped,
        scale=scale,
        zero_point=zero_point,
        scheme=scheme,
        orig_dtype=orig_dtype,
        orig_shape=orig_shape,
    )


def quantize(
    tensor: Union[Tensor, np.ndarray],
    scheme: str = "symmetric",
) -> QuantizedTensor:
    """Quantize floating-point tensor to INT8 with automatic calibration.

    Args:
        tensor: Input floating-point Tensor or array.
        scheme: Quantization scheme ('symmetric' or 'asymmetric').

    Returns:
        Calibrated QuantizedTensor instance.
    """
    scale, zero_point = calibrate_tensor(tensor, scheme=scheme)
    return quantize_tensor(tensor, scale=scale, zero_point=zero_point, scheme=scheme)


def dequantize_tensor(qtensor: QuantizedTensor) -> Tensor:
    """Dequantize an INT8 QuantizedTensor back to an FP32 Tensor.

    Formula:
        x = (q - zero_point) * scale

    Args:
        qtensor: QuantizedTensor instance.

    Returns:
        FP32 Tensor with reconstructed values.
    """
    q_arr = qtensor.numpy().astype(np.float32)
    deq_arr = (q_arr - float(qtensor.zero_point)) * float(qtensor.scale)
    return Tensor(deq_arr, dtype=float32)


def dequantize(qtensor: QuantizedTensor) -> Tensor:
    """Alias for dequantize_tensor."""
    return dequantize_tensor(qtensor)


def qmatmul(
    a: Union[QuantizedTensor, Tensor],
    b: Union[QuantizedTensor, Tensor],
) -> Tensor:
    """Quantized matrix multiplication (@) of two 2D matrices producing an FP32 output.

    Accumulates in 32-bit integer arithmetic to prevent 8-bit overflow:
        C(i, j) = scale_a * scale_b * sum_k((A_q(i, k) - zp_a) * (B_q(k, j) - zp_b))

    Supports both NumPy and Native C++ compute backends via the dispatcher.

    Args:
        a: Left matrix operand (QuantizedTensor or Tensor).
        b: Right matrix operand (QuantizedTensor or Tensor).

    Returns:
        Dequantized FP32 Tensor resulting from the matrix multiplication.
    """
    # Auto-quantize standard Tensors if passed
    a_q = a if isinstance(a, QuantizedTensor) else quantize(a, scheme="symmetric")
    b_q = b if isinstance(b, QuantizedTensor) else quantize(b, scheme="symmetric")

    if a_q.ndim != 2 or b_q.ndim != 2:
        raise ShapeError(f"qmatmul requires 2D matrices, got a.ndim={a_q.ndim}, b.ndim={b_q.ndim}")

    if a_q.shape[1] != b_q.shape[0]:
        raise ShapeError(f"qmatmul inner dimension mismatch: {a_q.shape} x {b_q.shape}")

    backend = get_backend()

    if backend == "native" and can_native_qmatmul(a_q.ndim, a_q.shape, b_q.ndim, b_q.shape):
        c_arr = native_qmatmul(
            a_q.numpy(),
            b_q.numpy(),
            a_q.scale,
            a_q.zero_point,
            b_q.scale,
            b_q.zero_point,
        )
        set_last_backend("native")
    else:
        # NumPy integer accumulation path
        a_int = a_q.numpy().astype(np.int32) - a_q.zero_point
        b_int = b_q.numpy().astype(np.int32) - b_q.zero_point
        accum_int32 = np.matmul(a_int, b_int)
        c_arr = accum_int32.astype(np.float32) * (a_q.scale * b_q.scale)
        set_last_backend("native (fallback)" if backend == "native" else "numpy")

    return Tensor(c_arr, dtype=float32)
