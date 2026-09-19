"""TensorForge v2.2.0 Performance Optimization Experiments & Throughput Scaling Demo.

Demonstrates:
1. Baseline vs Fused vs Compiled vs INT8 controlled optimization stage comparisons.
2. Speedup multipliers, latency change %, and workspace memory reduction metrics.
3. Batch size scaling across sizes 1, 2, 4, 8, 16, 32, 64, 128.
4. Worker thread concurrency scaling & scaling efficiency calculation.
5. Dynamic batching impact under multi-threaded load.
6. Machine-readable JSON export and human-readable ASCII report formatting.
"""

import os
import sys
import tempfile

# Ensure repository root is in sys.path when running script directly from any location
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tensorforge as tf
from benchmarks.performance_optimization_suite import OptimizationExperimentSuite
from benchmarks.throughput_scaling import ThroughputScalingSuite


def main() -> None:
    print("=" * 85)
    print("TensorForge v2.2.0 Performance Optimization Experiments & Scaling Analysis")
    print(f"TensorForge Version: {tf.__version__}")
    print("=" * 85)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Controlled Optimization Experiments Suite
        print("\n1. Running Controlled Optimization Experiments Suite (Baseline vs Fused vs Compiled vs INT8)...")
        opt_suite = OptimizationExperimentSuite(iterations=50, warmup_iterations=5, batch_size=16)
        opt_suite.run_suite()
        print(opt_suite.summary_table())

        opt_json_path = os.path.join(tmpdir, "optimization_suite_report.json")
        opt_suite.export_json(opt_json_path)
        print(f"   ✓ Exported machine-readable JSON to: '{opt_json_path}'")

        # 2. Throughput & Concurrency Scaling Suite
        print("\n2. Running Throughput & Concurrency Scaling Suite...")
        scale_suite = ThroughputScalingSuite(
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128],
            worker_counts=[1, 2, 4, 8],
            iterations=50,
        )
        print("   ✓ Benchmarking batch size scaling...")
        scale_suite.run_batch_size_scaling()

        print("   ✓ Benchmarking thread concurrency scaling & efficiency...")
        scale_suite.run_concurrency_scaling()

        print("   ✓ Benchmarking dynamic batching under concurrent load...")
        scale_suite.run_dynamic_batching_scaling(num_producers=4, requests_per_producer=20)

        print(scale_suite.summary_table())

        scale_json_path = os.path.join(tmpdir, "throughput_scaling_report.json")
        scale_suite.export_json(scale_json_path)
        print(f"   ✓ Exported machine-readable JSON to: '{scale_json_path}'")

    print("\n" + "=" * 85)
    print("TensorForge v2.2.0 Performance Analysis Demo Completed Successfully!")
    print("=" * 85)


if __name__ == "__main__":
    main()
