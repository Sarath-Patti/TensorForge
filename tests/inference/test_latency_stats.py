"""Tests for latency distribution statistics, percentiles (p50, p95, p99), and throughput calculations."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeProfiler
from tensorforge.serialization import save_model


class TestLatencyStats(unittest.TestCase):

    def test_profiler_percentiles_calculation(self):
        profiler = RuntimeProfiler(history_size=100)
        profiler.enable(detailed=False)

        # Record synthetic latencies: 1ms, 2ms, ..., 100ms
        for i in range(1, 101):
            profiler.record_prediction(duration_ns=i * 1_000_000, batch_size=1)

        stats = profiler.latency_stats()
        self.assertEqual(stats["prediction_count"], 100)
        self.assertEqual(stats["total_samples"], 100)
        self.assertAlmostEqual(stats["min_ms"], 1.0, places=2)
        self.assertAlmostEqual(stats["max_ms"], 100.0, places=2)
        self.assertAlmostEqual(stats["mean_ms"], 50.5, places=2)
        self.assertAlmostEqual(stats["p50_ms"], 50.0, delta=1.0)
        self.assertAlmostEqual(stats["p95_ms"], 95.0, delta=1.0)
        self.assertAlmostEqual(stats["p99_ms"], 99.0, delta=1.0)
        self.assertGreater(stats["throughput_samples_per_sec"], 0.0)

    def test_bounded_history_size(self):
        profiler = RuntimeProfiler(history_size=10)
        profiler.enable()

        for i in range(50):
            profiler.record_prediction(duration_ns=(i + 1) * 1_000_000)

        self.assertEqual(profiler._prediction_count, 50)
        self.assertEqual(len(profiler._latencies_ns), 10)

        stats = profiler.latency_stats()
        # Ring buffer contains last 10 elements: 41..50
        self.assertAlmostEqual(stats["min_ms"], 41.0, places=2)
        self.assertAlmostEqual(stats["max_ms"], 50.0, places=2)

    def test_runtime_latency_stats_and_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
            runtime.enable_profiling(detailed=True)

            for _ in range(10):
                _ = runtime.predict(tf.randn((4, 8)))

            lat = runtime.latency_stats()
            self.assertEqual(lat["prediction_count"], 10)
            self.assertEqual(lat["total_samples"], 40)
            self.assertGreater(lat["mean_ms"], 0.0)
            self.assertGreater(lat["p95_ms"], 0.0)

            report = runtime.profile()
            self.assertIn("TensorForge Performance Report", str(report))
            self.assertIn("Latency Distribution", report.latency_summary())


if __name__ == "__main__":
    unittest.main()
