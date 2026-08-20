"""Tests for runtime lifecycle management, context manager support, close handling, and health diagnostics."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeClosedError


class TestRuntimeLifecycle(unittest.TestCase):

    def test_runtime_context_manager(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            with InferenceRuntime.load(model_path) as runtime:
                self.assertFalse(runtime.is_closed)
                self.assertFalse(runtime.closed)
                out = runtime.predict(np.random.randn(2, 8).astype(np.float32))
                self.assertEqual(out.shape, (2, 16))

            self.assertTrue(runtime.is_closed)
            self.assertTrue(runtime.closed)

    def test_predict_after_close_raises_runtime_closed_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            runtime.close()

            self.assertTrue(runtime.is_closed)

            with self.assertRaises(RuntimeClosedError):
                runtime.predict(np.random.randn(2, 8).astype(np.float32))

    def test_config_and_compile_after_close_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            runtime.close()

            with self.assertRaises(RuntimeClosedError):
                runtime.set_num_threads(2)

            with self.assertRaises(RuntimeClosedError):
                runtime.compile(input_shape=(2, 8))

            with self.assertRaises(RuntimeClosedError):
                runtime.optimize()

    def test_runtime_idempotent_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            runtime.close()
            runtime.close()  # No error on repeat close
            self.assertTrue(runtime.is_closed)

    def test_runtime_health_and_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
            health = runtime.health()

            self.assertEqual(health["status"], "healthy")
            self.assertTrue(health["is_compiled"])
            self.assertEqual(health["prediction_count"], 0)
            self.assertEqual(health["error_count"], 0)
            self.assertEqual(health["active_contexts"], 0)

            _ = runtime.predict(np.random.randn(4, 8).astype(np.float32))

            health_after = runtime.health()
            self.assertEqual(health_after["prediction_count"], 1)
            self.assertEqual(health_after["error_count"], 0)

            stats = runtime.stats()
            self.assertIn("health", stats)
            self.assertEqual(stats["prediction_count"], 1)
            self.assertEqual(stats["tensorforge_version"], tf.__version__)

            runtime.close()
            health_closed = runtime.health()
            self.assertEqual(health_closed["status"], "closed")


if __name__ == "__main__":
    unittest.main()
