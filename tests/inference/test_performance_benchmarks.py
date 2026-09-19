"""Unit tests for Performance Optimization Experiments and Throughput Scaling Suite in TensorForge v2.2.0."""

import json
import os
import tempfile
import unittest

from benchmarks.performance_optimization_suite import (
    OptimizationExperimentResult,
    OptimizationExperimentSuite,
)
from benchmarks.throughput_scaling import (
    BatchScalingResult,
    ConcurrencyScalingResult,
    DynamicBatchingScalingResult,
    ThroughputScalingSuite,
)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test suite verifying metric calculations, result schemas, scaling efficiency, and report formatting."""

    def test_optimization_experiment_result_schema_and_calculations(self):
        res_a = OptimizationExperimentResult(
            model_name="Small MLP",
            stage_name="Stage A: Baseline",
            batch_size=16,
            iterations=100,
            throughput_samples_per_sec=10000.0,
            requests_per_sec=625.0,
            avg_latency_ms=1.6,
            min_latency_ms=1.2,
            max_latency_ms=2.5,
            p50_latency_ms=1.5,
            p95_latency_ms=2.2,
            workspace_bytes=4096,
            peak_memory_bytes=4096,
            speedup_vs_baseline=1.0,
        )

        res_b = OptimizationExperimentResult(
            model_name="Small MLP",
            stage_name="Stage B: Fused",
            batch_size=16,
            iterations=100,
            throughput_samples_per_sec=20000.0,
            requests_per_sec=1250.0,
            avg_latency_ms=0.8,
            min_latency_ms=0.6,
            max_latency_ms=1.2,
            p50_latency_ms=0.75,
            p95_latency_ms=1.1,
            workspace_bytes=2048,
            peak_memory_bytes=2048,
            speedup_vs_baseline=20000.0 / 10000.0,
            latency_change_pct=((0.8 - 1.6) / 1.6) * 100.0,
            memory_reduction_pct=((4096 - 2048) / 4096) * 100.0,
        )

        self.assertEqual(res_b.speedup_vs_baseline, 2.0)
        self.assertEqual(res_b.latency_change_pct, -50.0)
        self.assertEqual(res_b.memory_reduction_pct, 50.0)

        d_b = res_b.to_dict()
        self.assertIn("throughput_samples_per_sec", d_b)
        self.assertEqual(d_b["speedup_vs_baseline"], 2.0)
        self.assertEqual(d_b["latency_change_pct"], -50.0)

    def test_concurrency_scaling_efficiency_calculation(self):
        # Base: 1 worker = 1000 samples/sec
        # 4 workers = 3600 samples/sec -> Ideal = 4000 samples/sec -> Efficiency = 3600/4000 = 0.90 (90%)
        res = ConcurrencyScalingResult(
            worker_count=4,
            throughput_samples_per_sec=3600.0,
            avg_latency_ms=4.44,
            scaling_efficiency=3600.0 / (1000.0 * 4),
        )

        d = res.to_dict()
        self.assertEqual(res.scaling_efficiency, 0.90)
        self.assertEqual(d["scaling_efficiency_pct"], 90.0)

    def test_batch_scaling_result_schema(self):
        res = BatchScalingResult(
            batch_size=32,
            throughput_samples_per_sec=50000.0,
            requests_per_sec=1562.5,
            avg_latency_ms=0.64,
            p50_latency_ms=0.60,
            p95_latency_ms=0.90,
            workspace_bytes=8192,
        )

        d = res.to_dict()
        self.assertEqual(d["batch_size"], 32)
        self.assertEqual(d["throughput_samples_per_sec"], 50000.0)
        self.assertEqual(d["workspace_bytes"], 8192)

    def test_optimization_suite_json_export_and_formatting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = OptimizationExperimentSuite(iterations=2, warmup_iterations=1, batch_size=4)

            # Manually inject mock results to test formatting and export deterministic logic
            res = OptimizationExperimentResult(
                model_name="Small MLP",
                stage_name="Stage A: Baseline",
                batch_size=4,
                iterations=2,
                throughput_samples_per_sec=5000.0,
                requests_per_sec=1250.0,
                avg_latency_ms=0.8,
                min_latency_ms=0.7,
                max_latency_ms=0.9,
                p50_latency_ms=0.8,
                p95_latency_ms=0.9,
                workspace_bytes=1024,
                peak_memory_bytes=1024,
            )
            suite.results.append(res)

            summary = suite.summary_table()
            self.assertIn("Small MLP", summary)
            self.assertIn("Stage A: Baseline", summary)

            json_str = suite.to_json()
            self.assertIn("throughput_samples_per_sec", json_str)

            json_path = os.path.join(tmpdir, "opt_suite.json")
            suite.export_json(json_path)
            self.assertTrue(os.path.exists(json_path))

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["batch_size"], 4)

    def test_throughput_scaling_suite_json_export_and_formatting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            suite = ThroughputScalingSuite(batch_sizes=[1, 4], worker_counts=[1, 2], iterations=2)

            suite.batch_results.append(
                BatchScalingResult(1, 1000.0, 1000.0, 1.0, 1.0, 1.2, 512)
            )
            suite.concurrency_results.append(
                ConcurrencyScalingResult(2, 1900.0, 1.05, 0.95)
            )
            suite.dynamic_batching_results.append(
                DynamicBatchingScalingResult(True, 2500.0, 0.8, 100, 10)
            )

            summary = suite.summary_table()
            self.assertIn("Batch Size Scaling Performance", summary)
            self.assertIn("Concurrency Scaling Efficiency", summary)
            self.assertIn("Dynamic Batching Concurrent Scaling Comparison", summary)

            json_path = os.path.join(tmpdir, "scale_suite.json")
            suite.export_json(json_path)
            self.assertTrue(os.path.exists(json_path))


if __name__ == "__main__":
    unittest.main()
