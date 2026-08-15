"""Tests for inference backend integration and dispatch."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import backend_context, is_native_available
from tensorforge.backend.native_backend import (
    native_add,
    native_matmul,
    native_mul,
    native_qmatmul,
    native_sub,
)
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def test_inference_numpy_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path, backend="numpy")
        assert runtime.backend == "numpy"

        x = tf.randn((4, 8))
        out = runtime.predict(x)
        assert out.shape == (4, 4)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_inference_native_backend_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
        save_model(model, model_path)

        runtime_np = InferenceRuntime.load(model_path, backend="numpy")
        runtime_native = InferenceRuntime.load(model_path, backend="native")

        x = tf.randn((8, 8))
        out_np = runtime_np.predict(x)
        out_native = runtime_native.predict(x)

        np.testing.assert_allclose(out_np.numpy(), out_native.numpy(), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_native_low_level_kernels_pointer_conversion():
    # 1. Test native_add
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    res_add = native_add(a, b)
    np.testing.assert_allclose(res_add, a + b)

    # 2. Test native_sub
    res_sub = native_sub(a, b)
    np.testing.assert_allclose(res_sub, a - b)

    # 3. Test native_mul
    res_mul = native_mul(a, b)
    np.testing.assert_allclose(res_mul, a * b)

    # 4. Test native_matmul
    res_matmul = native_matmul(a, b)
    np.testing.assert_allclose(res_matmul, a @ b)

    # 5. Test native_qmatmul
    a_int8 = np.array([[10, 20], [30, 40]], dtype=np.int8)
    b_int8 = np.array([[1, 2], [3, 4]], dtype=np.int8)
    res_qmatmul = native_qmatmul(a_int8, b_int8, scale_a=0.5, zp_a=0, scale_b=0.5, zp_b=0)
    expected = (a_int8.astype(np.float32) * 0.5) @ (b_int8.astype(np.float32) * 0.5)
    np.testing.assert_allclose(res_qmatmul, expected, rtol=1e-5, atol=1e-5)


def test_inference_scoped_backend_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Linear(4, 2)
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path)
        x = tf.randn((2, 4))

        with backend_context("numpy"):
            out = runtime.predict(x)
            assert out.shape == (2, 2)
