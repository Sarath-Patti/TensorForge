"""Tests for multi-backend execution, no-grad invariants, and INT8 fusion."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import backend_context, is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


class TestFusedBackendDispatch(unittest.TestCase):

    def test_fused_runtime_summary_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).optimize()
            summary = runtime.summary()

            self.assertTrue(summary["is_optimized"])
            self.assertEqual(summary["original_nodes"], 4)
            self.assertEqual(summary["optimized_nodes"], 2)
            self.assertEqual(summary["fused_count"], 2)
            self.assertEqual(summary["fused_patterns"], ["Linear+ReLU", "Linear+Softmax"])

    def test_fused_native_vs_numpy_parity(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
                nn.Sigmoid(),
            )
            save_model(model, model_path)

            runtime_np = InferenceRuntime.load(model_path, backend="numpy").optimize()
            runtime_native = InferenceRuntime.load(model_path, backend="native").optimize()

            x = tf.randn((10, 8))
            out_np = runtime_np.predict(x)
            out_native = runtime_native.predict(x)

            np.testing.assert_allclose(out_np.numpy(), out_native.numpy(), atol=1e-5, rtol=1e-5)

    def test_fused_no_grad_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).optimize()
            w_before = runtime.model[0].weight.numpy().copy()

            x = tf.randn((4, 8), requires_grad=True)
            out = runtime.predict(x)

            self.assertFalse(out.requires_grad)
            self.assertIsNone(out.grad_fn)

            w_after = runtime.model[0].weight.numpy()
            np.testing.assert_array_equal(w_before, w_after)

    def test_fused_quantized_inference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "quant_model.tfmodel")

            seq = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            seq.eval()

            state = {
                "0.weight": quantize(seq[0].weight, scheme="symmetric"),
                "0.bias": quantize(seq[0].bias, scheme="symmetric"),
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

            runtime = InferenceRuntime.load(model_path).optimize()
            self.assertTrue(runtime.is_quantized)
            self.assertTrue(runtime.is_optimized)

            x = tf.randn((4, 8))
            out = runtime.predict(x)
            self.assertEqual(out.shape, (4, 16))
            self.assertFalse(out.requires_grad)


if __name__ == "__main__":
    unittest.main()
