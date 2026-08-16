"""Tests for compiled InferenceRuntime execution, caching, and correctness."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


class TestCompiledRuntime(unittest.TestCase):

    def test_runtime_compile_and_predict_parity(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            self.assertFalse(runtime.is_compiled)

            runtime.compile(input_shape=(8, 16))
            self.assertTrue(runtime.is_compiled)
            self.assertIsNotNone(runtime.execution_plan)
            self.assertGreater(runtime.workspace_size, 0)

            # Predict with compiled runtime
            x = tf.randn((8, 16))
            with tf.no_grad():
                ref_out = model(x)
            compiled_out = runtime.predict(x)

            np.testing.assert_allclose(compiled_out.numpy(), ref_out.numpy(), atol=1e-5, rtol=1e-5)

    def test_runtime_compiled_dynamic_batch_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 2),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))

            # Predict with different batch sizes
            for b in [1, 4, 16, 32]:
                x = tf.randn((b, 8))
                with tf.no_grad():
                    ref_out = model(x)
                out = runtime.predict(x)
                np.testing.assert_allclose(out.numpy(), ref_out.numpy(), atol=1e-5, rtol=1e-5)

    def test_compiled_native_vs_numpy_parity(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.Sigmoid(),
            )
            save_model(model, model_path)

            runtime_np = InferenceRuntime.load(model_path, backend="numpy").compile(input_shape=(8, 16))
            runtime_native = InferenceRuntime.load(model_path, backend="native").compile(input_shape=(8, 16))

            x = tf.randn((8, 16))
            out_np = runtime_np.predict(x)
            out_native = runtime_native.predict(x)

            np.testing.assert_allclose(out_native.numpy(), out_np.numpy(), atol=1e-5, rtol=1e-5)

    def test_compiled_quantized_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "q_model.tfmodel")
            seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            seq.eval()

            state = {
                "0.weight": quantize(seq[0].weight, scheme="symmetric"),
                "0.bias": quantize(seq[0].bias, scheme="symmetric"),
                "2.weight": quantize(seq[2].weight, scheme="symmetric"),
                "2.bias": quantize(seq[2].bias, scheme="symmetric"),
            }
            meta = {
                "format_version": "1.0",
                "library_version": "1.3.0",
                "architecture": extract_module_architecture(seq),
                "is_quantized": True,
                "created_at": "2026-08-16T12:00:00Z",
                "custom_metadata": {},
            }
            write_tfmodel_container(model_path, state, meta)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
            self.assertTrue(runtime.is_quantized)
            self.assertTrue(runtime.is_compiled)

            x = tf.randn((4, 8))
            out = runtime.predict(x)
            self.assertEqual(out.shape, (4, 4))
            self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
