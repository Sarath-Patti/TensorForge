"""Tests verifying numerical correctness and parity of parallel CPU compute kernels."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.backend import is_native_available, set_num_threads
from tensorforge.backend.native_backend import (
    native_fused_linear,
    native_fused_linear_relu,
    native_fused_linear_sigmoid,
    native_fused_linear_softmax,
    native_fused_linear_tanh,
    native_fused_qlinear_relu,
    native_matmul,
)
from tensorforge.quantization import quantize


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_parallel_matmul_parity():
    np.random.seed(42)
    m, k, n = 64, 128, 64
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    set_num_threads(1)
    out_st = native_matmul(a, b)

    set_num_threads(4)
    out_mt = native_matmul(a, b)

    np.testing.assert_allclose(out_mt, out_st, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_parallel_fused_linear_relu_parity():
    np.random.seed(42)
    m, k, n = 64, 128, 32
    x = np.random.randn(m, k).astype(np.float32)
    w = np.random.randn(n, k).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    set_num_threads(1)
    out_st = native_fused_linear_relu(x, w, b)

    set_num_threads(4)
    out_mt = native_fused_linear_relu(x, w, b)

    np.testing.assert_allclose(out_mt, out_st, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_parallel_fused_linear_activations_parity():
    np.random.seed(42)
    m, k, n = 32, 64, 16
    x = np.random.randn(m, k).astype(np.float32)
    w = np.random.randn(n, k).astype(np.float32)
    b = np.random.randn(n).astype(np.float32)

    # Sigmoid
    set_num_threads(1)
    sig_st = native_fused_linear_sigmoid(x, w, b)
    set_num_threads(4)
    sig_mt = native_fused_linear_sigmoid(x, w, b)
    np.testing.assert_allclose(sig_mt, sig_st, atol=1e-5, rtol=1e-5)

    # Tanh
    set_num_threads(1)
    tanh_st = native_fused_linear_tanh(x, w, b)
    set_num_threads(4)
    tanh_mt = native_fused_linear_tanh(x, w, b)
    np.testing.assert_allclose(tanh_mt, tanh_st, atol=1e-5, rtol=1e-5)

    # Softmax
    set_num_threads(1)
    sm_st = native_fused_linear_softmax(x, w, b, dim=-1)
    set_num_threads(4)
    sm_mt = native_fused_linear_softmax(x, w, b, dim=-1)
    np.testing.assert_allclose(sm_mt, sm_st, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_parallel_fused_qlinear_relu_parity():
    np.random.seed(42)
    m, k, n = 64, 128, 32
    x_t = tf.randn((m, k))
    w_t = tf.randn((n, k))
    b = np.random.randn(n).astype(np.float32)

    x_q = quantize(x_t, scheme="symmetric")
    w_q = quantize(w_t, scheme="symmetric")

    set_num_threads(1)
    out_st = native_fused_qlinear_relu(
        x_q.numpy(), w_q.numpy(), b, float(x_q.scale), int(x_q.zero_point), float(w_q.scale), int(w_q.zero_point)
    )

    set_num_threads(4)
    out_mt = native_fused_qlinear_relu(
        x_q.numpy(), w_q.numpy(), b, float(x_q.scale), int(x_q.zero_point), float(w_q.scale), int(w_q.zero_point)
    )

    np.testing.assert_allclose(out_mt, out_st, atol=1e-5, rtol=1e-5)
