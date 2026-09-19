"""TensorForge v2.1.0 Performance Profiling & Operator Bottleneck Analysis Demo.

Demonstrates:
1. Model construction and compilation.
2. Enabling operator-level profiling.
3. Executing a representative inference workload.
4. Printing the structured PerformanceReport summary.
5. Identifying top operator bottlenecks.
6. Inspecting backend, execution count, latency distribution, and shape information.
"""

import os
import tempfile
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def main() -> None:
    print("=" * 70)
    print("TensorForge v2.1.0 Performance Profiling & Operator Bottleneck Analysis")
    print(f"TensorForge Version: {tf.__version__}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Define and Save Model Architecture
        print("\n1. Constructing Neural Network Architecture...")
        model = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(),
        )
        model_path = os.path.join(tmpdir, "demo_model.tfmodel")
        save_model(model, model_path)
        print(f"   ✓ Model saved to: {model_path}")

        # 2. Compile InferenceRuntime with Operator Fusion
        print("\n2. Compiling InferenceRuntime with Operator Fusion...")
        runtime = InferenceRuntime.load(model_path)
        runtime.compile(input_shape=(8, 8))
        print(f"   ✓ Compiled runtime backend: {runtime.backend}")
        print(f"   ✓ Num threads: {runtime.num_threads}")

        # 3. Baseline Unprofiled Prediction (Zero Overhead)
        print("\n3. Executing Unprofiled Baseline Predictions...")
        input_tensor = tf.randn((8, 8))
        baseline_out = runtime.predict(input_tensor)
        print(f"   ✓ Baseline output shape: {baseline_out.shape}")
        print(f"   ✓ Baseline profiling enabled: {runtime.profiling_enabled}")

        # 4. Enable Profiling & Execute Workload
        print("\n4. Enabling Detailed Operator Profiler...")
        profiler = runtime.profiler(detailed=True)

        print("   ✓ Running 50 predictions under profiler...")
        with profiler:
            for _ in range(50):
                _ = runtime.predict(input_tensor)

        # 5. Retrieve PerformanceReport
        print("\n5. Generating PerformanceReport & Telemetry Breakdown...")
        report = profiler.report()
        print(report.summary())

        # 6. Identify Top Operator Bottlenecks
        print("\n6. Identifying Top Operator Bottlenecks...")
        bottlenecks = profiler.top_bottlenecks(limit=5)
        print(report.top_bottlenecks_summary(limit=5))

        print("\n   Top Bottleneck Details:")
        for idx, b in enumerate(bottlenecks, 1):
            print(
                f"   {idx}. Operator: {b['operator']:<20} | Calls: {b['calls']:<4} | "
                f"Total: {b['total_time_ms']:>7.3f} ms | Avg: {b['avg_time_ms']:>6.3f} ms | "
                f"Share: {b['percent']:>5.1f}% | Backend: {b['backend']} | Shape: {b['input_shape']} -> {b['output_shape']}"
            )

        # 7. Clean Shutdown
        runtime.close()

    print("\n" + "=" * 70)
    print("TensorForge v2.1.0 Performance Profiling Demo Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
