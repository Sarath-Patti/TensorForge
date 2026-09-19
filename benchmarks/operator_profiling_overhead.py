"""TensorForge Benchmark: Operator Profiling Overhead Measurement.

Measures:
1. Baseline inference latency and throughput (profiling disabled).
2. Profiled inference latency and throughput (detailed profiling enabled).
3. Profiling overhead percentage calculation.
"""

import os
import tempfile
import time
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def benchmark_profiling_overhead(iterations: int = 1000, batch_size: int = 16) -> None:
    print("=" * 70)
    print("TensorForge Benchmark: Operator Profiling Overhead Measurement")
    print(f"TensorForge Version: {tf.__version__}")
    print(f"Benchmark Config: {iterations} iterations, batch_size={batch_size}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.Softmax(),
        )
        model_path = os.path.join(tmpdir, "bench_model.tfmodel")
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).compile(input_shape=(batch_size, 32))
        inputs = tf.randn((batch_size, 32))

        # Warmup
        for _ in range(50):
            _ = runtime.predict(inputs)

        # 1. Baseline Benchmark (Profiling Disabled)
        runtime.disable_profiling()
        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            _ = runtime.predict(inputs)
        t1 = time.perf_counter_ns()

        baseline_time_sec = (t1 - t0) / 1_000_000_000.0
        baseline_latency_ms = ((t1 - t0) / 1_000_000.0) / iterations
        baseline_throughput = (iterations * batch_size) / baseline_time_sec

        # 2. Profiled Benchmark (Profiling Enabled)
        profiler = runtime.profiler(detailed=True)
        t2 = time.perf_counter_ns()
        with profiler:
            for _ in range(iterations):
                _ = runtime.predict(inputs)
        t3 = time.perf_counter_ns()

        profiled_time_sec = (t3 - t2) / 1_000_000_000.0
        profiled_latency_ms = ((t3 - t2) / 1_000_000.0) / iterations
        profiled_throughput = (iterations * batch_size) / profiled_time_sec

        # 3. Overhead Calculation
        overhead_pct = ((profiled_time_sec - baseline_time_sec) / baseline_time_sec) * 100.0

        print("\nResults Summary:")
        print("-" * 70)
        print(f"  Baseline Latency (No Profiling): {baseline_latency_ms:.4f} ms/batch")
        print(f"  Baseline Throughput:             {baseline_throughput:.1f} samples/sec")
        print(f"  Profiled Latency (Detailed):     {profiled_latency_ms:.4f} ms/batch")
        print(f"  Profiled Throughput:             {profiled_throughput:.1f} samples/sec")
        print(f"  Profiling Overhead:              {overhead_pct:+.2f}%")
        print("-" * 70)

        # Report Bottlenecks recorded during profiled benchmark
        print("\nTop Operator Bottlenecks Recorded:")
        print(profiler.report().top_bottlenecks_summary(limit=5))

        runtime.close()


if __name__ == "__main__":
    benchmark_profiling_overhead()
