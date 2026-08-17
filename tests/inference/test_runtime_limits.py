"""Tests for RuntimeLimits configuration, validation, and enforcement."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeLimitError, RuntimeTimeoutError


class TestRuntimeLimits(unittest.TestCase):

    def test_runtime_limits_validation(self):
        # Valid limits
        limits = RuntimeLimits(
            max_batch_size=32,
            max_input_elements=1024,
            max_workspace_bytes=65536,
            max_prediction_time_ms=50.0,
            max_concurrent_requests=4,
        )
        self.assertEqual(limits.max_batch_size, 32)
        self.assertEqual(limits.max_input_elements, 1024)
        self.assertEqual(limits.max_workspace_bytes, 65536)
        self.assertEqual(limits.max_prediction_time_ms, 50.0)
        self.assertEqual(limits.max_concurrent_requests, 4)

        d = limits.to_dict()
        self.assertEqual(d["max_batch_size"], 32)

        # Invalid values
        with self.assertRaises(ValueError):
            RuntimeLimits(max_batch_size=0)

        with self.assertRaises(ValueError):
            RuntimeLimits(max_input_elements=-5)

        with self.assertRaises(ValueError):
            RuntimeLimits(max_workspace_bytes=-1)

        with self.assertRaises(ValueError):
            RuntimeLimits(max_prediction_time_ms=0)

        with self.assertRaises(ValueError):
            RuntimeLimits(max_concurrent_requests=0)

    def test_batch_size_limit_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            limits = RuntimeLimits(max_batch_size=4)
            runtime = InferenceRuntime.load(model_path, limits=limits)

            # Valid batch size
            x_valid = tf.randn((4, 8))
            out_valid = runtime.predict(x_valid)
            self.assertEqual(out_valid.shape, (4, 4))

            # Exceeding batch size
            x_invalid = tf.randn((8, 8))
            with self.assertRaises(RuntimeLimitError):
                runtime.predict(x_invalid)

            # Check rejection telemetry
            health = runtime.health()
            self.assertEqual(health["rejected_requests"], 1)
            self.assertEqual(health["resource_limit_violations"], 1)

    def test_input_elements_limit_enforcement(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(16, 4)
            save_model(model, model_path)

            limits = RuntimeLimits(max_input_elements=32)
            runtime = InferenceRuntime.load(model_path, limits=limits)

            # Valid: 2 * 16 = 32 elements
            _ = runtime.predict(tf.randn((2, 16)))

            # Exceeding: 3 * 16 = 48 elements > 32
            with self.assertRaises(RuntimeLimitError):
                runtime.predict(tf.randn((3, 16)))

    def test_prediction_timeout_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
            save_model(model, model_path)

            # Set an extremely small timeout (e.g. 0.000001 ms) to trigger timeout error
            limits = RuntimeLimits(max_prediction_time_ms=0.000001)
            runtime = InferenceRuntime.load(model_path, limits=limits)

            with self.assertRaises(RuntimeTimeoutError):
                runtime.predict(tf.randn((4, 32)))

            stats = runtime.stats()
            self.assertGreater(stats["timeout_count"], 0)


if __name__ == "__main__":
    unittest.main()
