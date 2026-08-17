"""Tests for scheduler integration with compiled plans, native/NumPy backends, and INT8 quantization."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model


class TestSchedulerBackend(unittest.TestCase):

    def test_scheduler_compiled_runtime_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(8, 16))
            with InferenceScheduler(runtime, max_batch_size=8, batch_timeout_ms=5.0) as scheduler:
                x = tf.randn((2, 16))
                out = scheduler.predict(x)
                self.assertEqual(out.shape, (2, 4))

            runtime.close()

    def test_scheduler_quantized_runtime_integration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            # Quantize linear weights
            linear_layer = runtime.model.layers[0] if hasattr(runtime.model, "layers") else runtime.model
            linear_layer.weight.data = quantize(linear_layer.weight.data, symmetric=True)

            with InferenceScheduler(runtime, max_batch_size=4, batch_timeout_ms=5.0) as scheduler:
                x = tf.randn((2, 8))
                out = scheduler.predict(x)
                self.assertEqual(out.shape, (2, 4))

            runtime.close()


if __name__ == "__main__":
    unittest.main()
