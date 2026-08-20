"""Tests for InferenceRuntime performance_snapshot(), metrics(), and memory analytics."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, PerformanceSnapshot
from tensorforge.serialization import save_model


class TestRuntimeObservability(unittest.TestCase):

    def test_runtime_performance_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))

            # Run 3 predictions
            for _ in range(3):
                _ = runtime.predict(tf.randn((4, 8)))

            snapshot = runtime.performance_snapshot()
            self.assertIsInstance(snapshot, PerformanceSnapshot)
            self.assertEqual(snapshot.requests.completed, 3)
            self.assertGreater(snapshot.throughput.requests_per_sec, 0.0)
            self.assertGreater(snapshot.memory.workspace_bytes, 0)
            self.assertGreater(snapshot.memory.parameter_bytes, 0)

            # Test alias
            metrics_obj = runtime.metrics()
            self.assertEqual(metrics_obj.requests.completed, 3)

            runtime.close()


if __name__ == "__main__":
    unittest.main()
