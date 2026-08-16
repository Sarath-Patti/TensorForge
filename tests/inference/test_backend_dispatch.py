"""Tests for inference backend integration and dispatch."""

import os
import tempfile
import unittest
import numpy as np
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


class TestBackendDispatch(unittest.TestCase):

    def test_inference_numpy_backend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path, backend="numpy")
            self.assertEqual(runtime.backend, "numpy")

            x = tf.randn((4, 8))
            out = runtime.predict(x)
            self.assertEqual(out.shape, (4, 4))

    def test_inference_native_backend_parity(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

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

    def test_native_low_level_kernels_pointer_conversion(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

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
        c = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        d = np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float32)
        res_mm = native_matmul(c, d)
        np.testing.assert_allclose(res_mm, c @ d, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
