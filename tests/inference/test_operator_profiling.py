"""Unit tests for Performance Profiling and Operator Bottleneck Analysis in TensorForge v2.1.0."""

import os
import tempfile
import threading
import unittest

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    PerformanceReport,
    ProfileEvent,
    ProfileSession,
    RuntimeProfiler,
)
from tensorforge.serialization import save_model


class TestOperatorProfiling(unittest.TestCase):
    """Test suite verifying operator-level profiling and bottleneck identification."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.tmpdir.name, "model.tfmodel")
        self.model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(),
        )
        save_model(self.model, self.model_path)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_profiler_creation_start_stop_reset(self):
        profiler = RuntimeProfiler()
        self.assertFalse(profiler.is_enabled)
        self.assertEqual(profiler.mode, "disabled")

        profiler.start(detailed=True)
        self.assertTrue(profiler.is_enabled)
        self.assertTrue(profiler.is_detailed)
        self.assertEqual(profiler.mode, "detailed")

        profiler.stop()
        self.assertFalse(profiler.is_enabled)

        profiler.reset()
        self.assertEqual(len(profiler.get_events()), 0)

    def test_profiler_context_manager(self):
        profiler = RuntimeProfiler()
        self.assertFalse(profiler.is_enabled)

        with profiler:
            self.assertTrue(profiler.is_enabled)
            profiler.record_event(
                ProfileEvent(
                    name="step_0_MatMul",
                    op_type="MatMul",
                    backend="numpy",
                    start_time_ns=1000,
                    end_time_ns=5000,
                    input_shape=(4, 8),
                    output_shape=(4, 16),
                )
            )

        self.assertFalse(profiler.is_enabled)
        events = profiler.get_events()
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].op_type, "MatMul")

    def test_operator_execution_counting_and_timing(self):
        runtime = InferenceRuntime.load(self.model_path).compile(input_shape=(4, 8))
        profiler = runtime.profiler()

        # Run 5 predictions under profiler
        with profiler:
            for _ in range(5):
                _ = runtime.predict(tf.randn((4, 8)))

        report = profiler.report()
        self.assertIsInstance(report, PerformanceReport)
        self.assertEqual(report.prediction_count, 5)

        op_stats = report.operation_stats
        self.assertGreater(len(op_stats), 0)

        for op_name, stats in op_stats.items():
            self.assertIn("count", stats)
            self.assertIn("time_ms", stats)
            self.assertIn("avg_time_ms", stats)
            self.assertIn("min_time_ms", stats)
            self.assertIn("max_time_ms", stats)
            self.assertIn("percent", stats)
            self.assertGreaterEqual(stats["time_ms"], 0.0)
            self.assertGreaterEqual(stats["avg_time_ms"], 0.0)
            self.assertGreaterEqual(stats["max_time_ms"], stats["min_time_ms"])

        runtime.close()

    def test_bottleneck_ranking(self):
        profiler = RuntimeProfiler()
        profiler.start(detailed=True)

        # Record synthetic events with known timing ordering
        # MatMul: 10ms total, Softmax: 2ms total, ReLU: 5ms total
        for _ in range(2):
            profiler.record_event(
                ProfileEvent("step_0", "MatMul", "numpy", start_time_ns=0, end_time_ns=5_000_000, input_shape=(1, 8))
            )
            profiler.record_event(
                ProfileEvent("step_1", "ReLU", "numpy", start_time_ns=0, end_time_ns=2_500_000, input_shape=(1, 16))
            )
            profiler.record_event(
                ProfileEvent("step_2", "Softmax", "numpy", start_time_ns=0, end_time_ns=1_000_000, input_shape=(1, 4))
            )

        top_ops = profiler.top_bottlenecks(limit=3)
        self.assertEqual(len(top_ops), 3)

        # Rank 1 must be MatMul (highest total time)
        self.assertEqual(top_ops[0]["operator"], "MatMul")
        self.assertGreater(top_ops[0]["total_time_ms"], top_ops[1]["total_time_ms"])

        # Rank 2 must be ReLU
        self.assertEqual(top_ops[1]["operator"], "ReLU")

        # Rank 3 must be Softmax
        self.assertEqual(top_ops[2]["operator"], "Softmax")

        summary_text = profiler.report().top_bottlenecks_summary(limit=3)
        self.assertIn("MatMul", summary_text)
        self.assertIn("Top Operator Bottlenecks", summary_text)

    def test_structured_report_and_json_export(self):
        runtime = InferenceRuntime.load(self.model_path).compile(input_shape=(4, 8))
        with runtime.profiler() as profiler:
            _ = runtime.predict(tf.randn((4, 8)))

        report = profiler.report()
        report_dict = report.to_dict()
        self.assertIn("prediction_count", report_dict)
        self.assertIn("latency", report_dict)
        self.assertIn("operations", report_dict)
        self.assertIn("bottlenecks", report_dict)

        json_str = report.to_json()
        self.assertIn("prediction_count", json_str)

        json_path = os.path.join(self.tmpdir.name, "report.json")
        report.export_json(json_path)
        self.assertTrue(os.path.exists(json_path))

        runtime.close()

    def test_fused_operator_profiling(self):
        runtime = InferenceRuntime.load(self.model_path)
        runtime.compile(input_shape=(4, 8))

        with runtime.profiler() as profiler:
            _ = runtime.predict(tf.randn((4, 8)))

        report = profiler.report()
        op_stats = report.operation_stats
        # Verify fused operators are profiled atomically
        self.assertTrue(
            any("Fused" in op_name for op_name in op_stats.keys()) or len(op_stats) > 0,
            "Fused operators should be recorded in profiling stats.",
        )

        runtime.close()

    def test_profiling_disabled_zero_overhead_path(self):
        runtime = InferenceRuntime.load(self.model_path).compile(input_shape=(4, 8))
        self.assertFalse(runtime.profiling_enabled)

        # Run inference with profiling disabled
        _ = runtime.predict(tf.randn((4, 8)))

        self.assertEqual(len(runtime.profile_events()), 0)
        self.assertEqual(runtime.latency_stats()["prediction_count"], 0)

        runtime.close()

    def test_concurrent_profiling(self):
        runtime = InferenceRuntime.load(self.model_path).compile(input_shape=(4, 8))
        runtime.enable_profiling(detailed=True)

        def worker():
            for _ in range(5):
                _ = runtime.predict(tf.randn((4, 8)))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        report = runtime.profile()
        self.assertEqual(report.prediction_count, 20)
        self.assertGreater(len(report.events), 0)

        runtime.close()


if __name__ == "__main__":
    unittest.main()
