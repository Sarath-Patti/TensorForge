"""TensorForge v1.2 Production Inference, Compiler & Multi-Threaded Parallel Benchmark.

Measures:
  - Native Single-Thread Eager vs Native Multi-Thread Eager
  - Native Fused Single-Thread vs Native Fused Multi-Thread
  - Native Compiled Single-Thread vs Native Compiled Multi-Thread
  - Batch sizes: 1, 8, 32, 128, 512
  - Thread scaling: 1, 2, 4, 8 threads
  - First-call compilation latency vs steady-state latency
  - Latency per batch (ms), per-sample latency (µs), throughput (samples/sec)
  - Memory workspace consumption
"""

import os
import tempfile
import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import get_num_threads, is_native_available, set_num_threads
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def benchmark_inference(
    batch_sizes=[1, 8, 32, 128, 512],
    thread_counts=[1, 2, 4, 8],
    num_warmup=15,
    num_repeats=50,
    enable_profiling=False,
):
    print("=" * 125)
    print(f"TensorForge v1.4: Production Inference, Parallel CPU & Telemetry Benchmark (Profiling={enable_profiling})")
    native_avail = is_native_available()
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 125)

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

        # ---------------------------------------------------------------------
        # 1. Comparative Backend & Compilation Benchmark
        # ---------------------------------------------------------------------
        configurations = [
            ("Eager NumPy", "numpy", False, False, 1),
            ("Fused NumPy", "numpy", True, False, 1),
            ("Compiled NumPy", "numpy", True, True, 1),
        ]
        if native_avail:
            configurations.append(("Native Eager (1 Thread)", "native", False, False, 1))
            configurations.append(("Native Eager (4 Threads)", "native", False, False, 4))
            configurations.append(("Native Fused (1 Thread)", "native", True, False, 1))
            configurations.append(("Native Fused (4 Threads)", "native", True, False, 4))
            configurations.append(("Native Compiled (1 Thread)", "native", True, True, 1))
            configurations.append(("Native Compiled (4 Threads)", "native", True, True, 4))

        results = {}

        for config_name, backend_name, optimize_flag, compile_flag, threads in configurations:
            print(f"\n--- Mode: {config_name} (Backend={backend_name}, Optimized={optimize_flag}, Compiled={compile_flag}, Threads={threads}) ---")
            print(
                f"{'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Per-Sample (µs)':<18} | {'Throughput (samples/s)':<24} | {'Workspace (B)':<15}"
            )
            print("-" * 105)

            results[config_name] = {}

            for b in batch_sizes:
                x_test = tf.randn((b, in_features))

                # Load runtime
                runtime = InferenceRuntime.load(model_path, backend=backend_name, num_threads=threads)
                if optimize_flag:
                    runtime.optimize()

                if compile_flag:
                    runtime.compile(input_shape=(b, in_features), num_threads=threads)

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

        # ---------------------------------------------------------------------
        # 2. Multi-Thread Scaling Benchmark (Native Compiled)
        # ---------------------------------------------------------------------
        if native_avail:
            print("\n" + "=" * 125)
            print("Parallel CPU Thread Scaling (Native Compiled Inference, Batch Size = 128):")
            print(f"{'Thread Count':<14} | {'Batch Latency (ms)':<22} | {'Throughput (samples/s)':<26} | {'Speedup vs 1-Thread':<20}")
            print("-" * 95)

            x_large = tf.randn((128, in_features))
            st_time = 1.0

            for tc in thread_counts:
                runtime_scaled = InferenceRuntime.load(model_path, backend="native", num_threads=tc).compile(
                    input_shape=(128, in_features), num_threads=tc
                )

                for _ in range(num_warmup):
                    _ = runtime_scaled.predict(x_large)

                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = runtime_scaled.predict(x_large)
                t_total = time.perf_counter() - start

                avg_ms = (t_total / num_repeats) * 1000.0
                thru = (128 * num_repeats) / t_total

                if tc == 1:
                    st_time = avg_ms
                speedup = st_time / avg_ms if avg_ms > 0 else 1.0

                print(f"{tc:<14} | {avg_ms:<22.3f} | {thru:<26.1f} | {speedup:<20.2f}x")

    print("=" * 125)


if __name__ == "__main__":
    benchmark_inference()
