"""Tests for native C++ operations and dispatcher fallback behavior."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.backend import (
    backend_context,
    get_last_backend,
    is_native_available,
    set_backend,
)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_native_elementwise_parity():
    np_a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np_b = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)

    a_tf = tf.tensor(np_a, dtype=tf.float32)
    b_tf = tf.tensor(np_b, dtype=tf.float32)

    with backend_context("native"):
        res_add = a_tf + b_tf
        assert get_last_backend() == "native"
        np.testing.assert_allclose(res_add.numpy(), np_a + np_b, rtol=1e-5)

        res_sub = a_tf - b_tf
        assert get_last_backend() == "native"
        np.testing.assert_allclose(res_sub.numpy(), np_a - np_b, rtol=1e-5)

        res_mul = a_tf * b_tf
        assert get_last_backend() == "native"
        np.testing.assert_allclose(res_mul.numpy(), np_a * np_b, rtol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_native_matmul_parity():
    np_a = np.random.randn(8, 12).astype(np.float32)
    np_b = np.random.randn(12, 6).astype(np.float32)

    a_tf = tf.tensor(np_a, dtype=tf.float32)
    b_tf = tf.tensor(np_b, dtype=tf.float32)

    with backend_context("native"):
        c_tf = a_tf @ b_tf
        assert get_last_backend() == "native"
        np.testing.assert_allclose(c_tf.numpy(), np_a @ np_b, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_broadcasting_fallback():
    # Native elementwise kernels only support identical shapes, so broadcasting falls back to NumPy
    a = tf.ones((4, 1), dtype=tf.float32)
    b = tf.ones((4, 4), dtype=tf.float32)

    with backend_context("native"):
        c = a + b
        assert get_last_backend() == "native (fallback)"
        assert c.shape == (4, 4)
        np.testing.assert_allclose(c.numpy(), 2.0)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_dtype_fallback():
    # Non-float32 tensors fall back to NumPy backend
    a = tf.ones((4,), dtype=tf.int32)
    b = tf.ones((4,), dtype=tf.int32)

    with backend_context("native"):
        c = a + b
        assert get_last_backend() == "native (fallback)"
        assert c.dtype == tf.int32
        np.testing.assert_allclose(c.numpy(), 2)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_native_autograd_compatibility():
    # Autograd backward must work seamlessly when forward pass uses native dispatch
    np_x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np_w = np.array([[0.5, -0.5], [1.5, 0.2]], dtype=np.float32)

    x = tf.tensor(np_x, dtype=tf.float32, requires_grad=True)
    w = tf.tensor(np_w, dtype=tf.float32, requires_grad=True)

    with backend_context("native"):
        y = x @ w
        loss = y.sum()
        loss.backward()

    # Expected gradients:
    # dLoss/dw = x^T @ 1 = sum_rows(x)
    # dLoss/dx = 1 @ w^T = sum_cols(w)
    expected_w_grad = np_x.T @ np.ones_like(np_x @ np_w)
    expected_x_grad = np.ones_like(np_x @ np_w) @ np_w.T

    np.testing.assert_allclose(w.grad.numpy(), expected_w_grad, rtol=1e-4)
    np.testing.assert_allclose(x.grad.numpy(), expected_x_grad, rtol=1e-4)
