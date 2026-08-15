"""Tests for INT8 quantized matrix multiplication."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.backend import backend_context, is_native_available
from tensorforge.quantization import (
    dequantize,
    mean_absolute_error,
    qmatmul,
    quantize,
)
from tensorforge.utils.validation import ShapeError


def test_qmatmul_numpy_backend():
    np.random.seed(42)
    A_np = np.random.randn(8, 16).astype(np.float32)
    B_np = np.random.randn(16, 4).astype(np.float32)
    FP32_ref = A_np @ B_np

    A_q = quantize(A_np, scheme="symmetric")
    B_q = quantize(B_np, scheme="symmetric")

    with backend_context("numpy"):
        C_q = qmatmul(A_q, B_q)

    assert isinstance(C_q, tf.Tensor)
    assert C_q.shape == (8, 4)
    assert C_q.dtype == tf.float32

    # MAE should be low (< 0.1 for normalized gaussian matrices)
    mae = mean_absolute_error(FP32_ref, C_q.numpy())
    assert mae < 0.1


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_qmatmul_native_backend():
    np.random.seed(42)
    A_np = np.random.randn(16, 32).astype(np.float32)
    B_np = np.random.randn(32, 8).astype(np.float32)

    A_q = quantize(A_np, scheme="symmetric")
    B_q = quantize(B_np, scheme="symmetric")

    with backend_context("numpy"):
        C_numpy = qmatmul(A_q, B_q)

    with backend_context("native"):
        C_native = qmatmul(A_q, B_q)

    assert C_native.shape == (16, 8)
    # Native and NumPy quantized matmul must match within float tolerance
    np.testing.assert_allclose(C_native.numpy(), C_numpy.numpy(), rtol=1e-5, atol=1e-5)


def test_qmatmul_operator_overload():
    A_q = quantize(np.random.randn(4, 4).astype(np.float32))
    B_q = quantize(np.random.randn(4, 4).astype(np.float32))

    C1 = A_q @ B_q
    C2 = qmatmul(A_q, B_q)
    np.testing.assert_array_equal(C1.numpy(), C2.numpy())


def test_qmatmul_dimension_mismatch():
    A_q = quantize(np.random.randn(4, 3).astype(np.float32))
    B_q = quantize(np.random.randn(5, 2).astype(np.float32))

    with pytest.raises(ShapeError):
        _ = qmatmul(A_q, B_q)
