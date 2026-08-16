"""Tests for InferenceRuntime API and lifecycle."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestRuntime(unittest.TestCase):

    def test_inference_runtime_load_and_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "classifier.tfmodel")

            model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            self.assertIsInstance(runtime, InferenceRuntime)
            self.assertFalse(runtime.is_quantized)
            self.assertEqual(runtime.input_shape, (8,))
            self.assertEqual(runtime.output_shape, (3,))

            # Test single sample prediction (1D or 2D batch)
            x_np = np.random.randn(1, 8).astype(np.float32)
            out1 = runtime.predict(x_np)
            self.assertIsInstance(out1, tf.Tensor)
            self.assertEqual(out1.shape, (1, 3))

            # Test batch prediction
            x_batch = tf.randn((10, 8))
            out_batch = runtime.predict_batch(x_batch)
            self.assertEqual(out_batch.shape, (10, 3))

    def test_inference_runtime_summary_and_properties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "linear.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path, metadata={"author": "TensorForge Team"})

            runtime = InferenceRuntime.load(model_path)
            summary = runtime.summary()

            self.assertEqual(summary["model_type"], "Linear")
            self.assertEqual(summary["num_parameters"], 10)  # 4*2 + 2
            self.assertEqual(summary["input_shape"], (4,))
            self.assertEqual(summary["output_shape"], (2,))
            self.assertFalse(summary["is_quantized"])
            self.assertIn("backend", summary)
            self.assertIn("architecture", summary)


if __name__ == "__main__":
    unittest.main()
