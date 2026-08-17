"""Tests for ProfileEvent, RuntimeProfiler modes, and basic profiler operations."""

import unittest
import time
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, ProfileEvent, RuntimeProfiler
from tensorforge.serialization import save_model
import tempfile
import os


class TestProfiler(unittest.TestCase):

    def test_profile_event_attributes_and_timing(self):
        t0 = 1_000_000_000
        t1 = 1_005_500_000
        event = ProfileEvent(
            name="step_0_FusedLinear",
            op_type="FusedLinear",
            backend="native_fused",
            mode="compiled",
            start_time_ns=t0,
            end_time_ns=t1,
            input_shape=(8, 16),
            output_shape=(8, 32),
            dtype="float32",
            batch_size=8,
            estimated_flops=16384,
            workspace_bytes=2048,
            num_threads=4,
            is_fused=True,
            is_compiled=True,
            context_id=1,
            extra={"activation": "relu"},
        )

        self.assertEqual(event.duration_ns, 5_500_000)
        self.assertAlmostEqual(event.duration_ms, 5.5, places=4)
        self.assertAlmostEqual(event.duration_us, 5500.0, places=2)
        self.assertAlmostEqual(event.duration_sec, 0.0055, places=5)
        self.assertEqual(event.input_shape, (8, 16))
        self.assertEqual(event.output_shape, (8, 32))
        self.assertEqual(event.estimated_flops, 16384)
        self.assertTrue(event.is_fused)
        self.assertTrue(event.is_compiled)

        d = event.to_dict()
        self.assertEqual(d["op_type"], "FusedLinear")
        self.assertEqual(d["duration_ms"], 5.5)

    def test_runtime_profiling_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            self.assertFalse(runtime.profiling_enabled)
            self.assertEqual(runtime.profiling_mode, "disabled")

            x = tf.randn((4, 8))
            _ = runtime.predict(x)

            # No events recorded when disabled
            self.assertEqual(len(runtime.profile_events()), 0)
            lat = runtime.latency_stats()
            self.assertEqual(lat["prediction_count"], 0)

    def test_runtime_enable_disable_profiling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))

            # Enable summary profiling
            runtime.enable_profiling(detailed=False)
            self.assertTrue(runtime.profiling_enabled)
            self.assertEqual(runtime.profiling_mode, "summary")

            x = tf.randn((4, 8))
            _ = runtime.predict(x)

            self.assertEqual(runtime.prediction_count, 1)
            lat = runtime.latency_stats()
            self.assertEqual(lat["prediction_count"], 1)
            self.assertGreater(lat["mean_ms"], 0.0)

            # In summary mode, step-level events are not recorded
            self.assertEqual(len(runtime.profile_events()), 0)

            # Enable detailed profiling
            runtime.enable_profiling(detailed=True)
            self.assertEqual(runtime.profiling_mode, "detailed")
            _ = runtime.predict(x)

            self.assertGreater(len(runtime.profile_events()), 0)

            # Disable profiling
            runtime.disable_profiling()
            self.assertFalse(runtime.profiling_enabled)
            self.assertEqual(runtime.profiling_mode, "disabled")

    def test_clear_and_reset_profiler(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).enable_profiling(detailed=True)
            _ = runtime.predict(tf.randn((2, 4)))

            self.assertEqual(runtime.latency_stats()["prediction_count"], 1)
            self.assertGreater(len(runtime.profile_events()), 0)

            runtime.clear_profiler()
            self.assertEqual(runtime.latency_stats()["prediction_count"], 0)
            self.assertEqual(len(runtime.profile_events()), 0)


if __name__ == "__main__":
    unittest.main()
