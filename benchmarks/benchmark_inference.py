"""TensorForge v1.0 Production Inference Benchmark.

Measures:
  - NumPy Unfused vs Native Unfused vs Native Fused
  - Latency per batch (ms)
  - Per-sample latency (µs)
  - Throughput (samples/sec)
  - Speedup of Fused Native relative to Unfused Native
"""

import os
import tempfile
import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def benchmark_inference(batch_sizes=[1, 8, 32, 128], num_warmup=15, num_repeats=50):
    print("=" * 105)
    print("TensorForge v1.0: Production Inference & Operator Fusion Benchmark")
    native_avail = is_native_available()
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 105)

    np.random.seed(42)
    in_features, hidden_dim, out_features = 64, 128, 10

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "bench_model.tfmodel")

        model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        configurations = [
            ("NumPy Unfused", "numpy", False),
            ("NumPy Fused", "numpy", True),
        ]
        if native_avail:
            configurations.append(("Native Unfused", "native", False))
            configurations.append(("Native Fused", "native", True))

        results = {}

        for config_name, backend_name, optimize_flag in configurations:
            print(f"\n--- Mode: {config_name} (Backend={backend_name}, Optimized={optimize_flag}) ---")
            print(
                f"{'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Per-Sample (µs)':<18} | {'Throughput (samples/s)':<24}"
            )
            print("-" * 85)

            runtime = InferenceRuntime.load(model_path, backend=backend_name)
            if optimize_flag:
                runtime.optimize()

            results[config_name] = {}

            for b in batch_sizes:
                x_test = tf.randn((b, in_features))

                # Warmup
                for _ in range(num_warmup):
                    _ = runtime.predict(x_test)

                # Benchmark
                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = runtime.predict(x_test)
                total_time = time.perf_counter() - start

                avg_batch_ms = (total_time / num_repeats) * 1000.0
                per_sample_us = (avg_batch_ms / b) * 1000.0
                throughput = (b * num_repeats) / total_time

                results[config_name][b] = avg_batch_ms

                print(
                    f"{b:<12} | {avg_batch_ms:<20.3f} | {per_sample_us:<18.2f} | {throughput:<24.1f}"
                )

        # Comparative speedup summary
        if "Native Fused" in results and "Native Unfused" in results:
            print("\n" + "=" * 105)
            print("Operator Fusion Speedup Summary (Native Fused vs Native Unfused):")
            print(f"{'Batch Size':<12} | {'Unfused Native (ms)':<22} | {'Fused Native (ms)':<20} | {'Speedup':<15}")
            print("-" * 75)
            for b in batch_sizes:
                unfused_t = results["Native Unfused"][b]
                fused_t = results["Native Fused"][b]
                speedup = unfused_t / fused_t if fused_t > 0 else 1.0
                print(f"{b:<12} | {unfused_t:<22.3f} | {fused_t:<20.3f} | {speedup:<15.2f}x")

    print("=" * 105)


if __name__ == "__main__":
    benchmark_inference()
