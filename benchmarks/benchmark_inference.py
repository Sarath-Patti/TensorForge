"""TensorForge v1.1 Production Inference & Compiler Benchmark.

Measures:
  - Eager NumPy vs Eager Native vs Fused Native vs Compiled Native
  - First-call compilation latency vs steady-state latency
  - Latency per batch (ms)
  - Per-sample latency (µs)
  - Throughput (samples/sec)
  - Workspace memory consumption
  - Speedup of Compiled Native relative to Eager Native
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
    print("=" * 115)
    print("TensorForge v1.1: Production Inference Compiler & Execution Planning Benchmark")
    native_avail = is_native_available()
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 115)

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
            ("Eager NumPy", "numpy", False, False),
            ("Fused NumPy", "numpy", True, False),
            ("Compiled NumPy", "numpy", True, True),
        ]
        if native_avail:
            configurations.append(("Eager Native", "native", False, False))
            configurations.append(("Fused Native", "native", True, False))
            configurations.append(("Compiled Native", "native", True, True))

        results = {}

        for config_name, backend_name, optimize_flag, compile_flag in configurations:
            print(f"\n--- Mode: {config_name} (Backend={backend_name}, Optimized={optimize_flag}, Compiled={compile_flag}) ---")
            print(
                f"{'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Per-Sample (µs)':<18} | {'Throughput (samples/s)':<24} | {'Workspace (B)':<15}"
            )
            print("-" * 95)

            results[config_name] = {}

            for b in batch_sizes:
                x_test = tf.randn((b, in_features))

                # Load runtime
                runtime = InferenceRuntime.load(model_path, backend=backend_name)
                if optimize_flag:
                    runtime.optimize()

                compilation_ms = 0.0
                if compile_flag:
                    t_compile_start = time.perf_counter()
                    runtime.compile(input_shape=(b, in_features))
                    compilation_ms = (time.perf_counter() - t_compile_start) * 1000.0

                # Warmup
                for _ in range(num_warmup):
                    _ = runtime.predict(x_test)

                # Benchmark steady-state
                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = runtime.predict(x_test)
                total_time = time.perf_counter() - start

                avg_batch_ms = (total_time / num_repeats) * 1000.0
                per_sample_us = (avg_batch_ms / b) * 1000.0
                throughput = (b * num_repeats) / total_time
                ws_bytes = runtime.workspace_size

                results[config_name][b] = avg_batch_ms

                ws_desc = f"{ws_bytes} B" if ws_bytes > 0 else "N/A"
                print(
                    f"{b:<12} | {avg_batch_ms:<20.3f} | {per_sample_us:<18.2f} | {throughput:<24.1f} | {ws_desc:<15}"
                )

        # Comparative speedup summary
        if "Compiled Native" in results and "Eager Native" in results:
            print("\n" + "=" * 115)
            print("Speedup Summary (Compiled Native vs Eager Native):")
            print(f"{'Batch Size':<12} | {'Eager Native (ms)':<20} | {'Compiled Native (ms)':<22} | {'Speedup':<15}")
            print("-" * 75)
            for b in batch_sizes:
                eager_t = results["Eager Native"][b]
                compiled_t = results["Compiled Native"][b]
                speedup = eager_t / compiled_t if compiled_t > 0 else 1.0
                print(f"{b:<12} | {eager_t:<20.3f} | {compiled_t:<22.3f} | {speedup:<15.2f}x")

    print("=" * 115)


if __name__ == "__main__":
    benchmark_inference()
