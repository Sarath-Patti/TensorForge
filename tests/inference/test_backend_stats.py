"""Tests for backend execution breakdown, native/NumPy statistics, and fallback counters."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestBackendStats(unittest.TestCase):

    def test_backend_statistics_numpy_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path, backend="numpy").compile(input_shape=(4, 8))
            runtime.enable_profiling(detailed=True)

            _ = runtime.predict(tf.randn((4, 8)))

            stats = runtime.backend_stats()
            self.assertIn("numpy", stats)
            self.assertIn("native", stats)
            self.assertGreater(stats["numpy"]["operations"], 0)
            self.assertGreater(stats["numpy"]["fused_operations"], 0)
            self.assertEqual(stats["native"]["operations"], 0)

    def test_backend_statistics_native_execution(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path, backend="native").compile(input_shape=(8, 16))
            runtime.enable_profiling(detailed=True)

            _ = runtime.predict(tf.randn((8, 16)))

            stats = runtime.backend_stats()
            self.assertGreater(stats["native"]["operations"], 0)
            self.assertGreater(stats["native"]["fused_operations"], 0)

            report = runtime.profile()
            bk_summary = report.backend_breakdown()
            self.assertIn("Native", bk_summary)


if __name__ == "__main__":
    unittest.main()
