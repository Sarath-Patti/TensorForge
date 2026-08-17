"""Tests verifying low overhead when profiling is disabled and proper lifecycle error handling."""

import os
import tempfile
import time
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeClosedError


class TestProfilingOverhead(unittest.TestCase):

    def test_profiling_disabled_zero_event_allocation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(8, 16))
            self.assertFalse(runtime.profiling_enabled)

            # Run 50 predictions
            x = tf.randn((8, 16))
            for _ in range(50):
                _ = runtime.predict(x)

            # Verify zero events stored
            self.assertEqual(len(runtime.profile_events()), 0)
            self.assertEqual(runtime.latency_stats()["prediction_count"], 0)

    def test_profiling_after_close_raises_runtime_closed_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            runtime.close()

            with self.assertRaises(RuntimeClosedError):
                runtime.enable_profiling()

            with self.assertRaises(RuntimeClosedError):
                runtime.profile()

            with self.assertRaises(RuntimeClosedError):
                with runtime.profile_session():
                    pass


if __name__ == "__main__":
    unittest.main()
