"""Profiling Overhead Benchmark for TensorForge Inference Runtime.

Measures inference latency and throughput across:
1. Profiling Disabled (Zero-overhead baseline)
2. Summary Profiling (Latency and backend tracking)
3. Detailed Profiling (Per-operator event instrumentation)

Evaluates over batch sizes: 1, 8, 32, 128.
"""

from __future__ import annotations

import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def run_benchmark() -> None:
    print("=" * 110)
    print("TensorForge v1.4 Profiling Overhead & Telemetry Benchmark")
    print("=" * 110)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "benchmark_model.tfmodel")

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        batch_sizes = [1, 8, 32, 128]
        num_iterations = 200
        warmup_iterations = 30

        print(f"\n{'Batch':<8} {'Mode':<18} {'Latency (ms)':<16} {'Throughput (s/s)':<22} {'Overhead vs Disabled':<24}")
        print("-" * 110)

        for batch_size in batch_sizes:
            x_test = tf.randn((batch_size, 64))

            # -------------------------------------------------------------
            # 1. Disabled (Baseline)
            # -------------------------------------------------------------
            runtime = InferenceRuntime.load(model_path).compile(input_shape=(batch_size, 64))
            runtime.disable_profiling()

            for _ in range(warmup_iterations):
                _ = runtime.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_disabled = (t1 - t0) / 1e9
            mean_latency_ms_disabled = (total_sec_disabled / num_iterations) * 1000.0
            throughput_disabled = (num_iterations * batch_size) / total_sec_disabled

            print(
                f"{batch_size:<8} {'Disabled':<18} {mean_latency_ms_disabled:<16.4f} "
                f"{throughput_disabled:<22.1f} {'0.00% (baseline)':<24}"
            )

            # -------------------------------------------------------------
            # 2. Summary Mode
            # -------------------------------------------------------------
            runtime.clear_profiler()
            runtime.enable_profiling(detailed=False)

            for _ in range(warmup_iterations):
                _ = runtime.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_summary = (t1 - t0) / 1e9
            mean_latency_ms_summary = (total_sec_summary / num_iterations) * 1000.0
            throughput_summary = (num_iterations * batch_size) / total_sec_summary
            overhead_summary = ((mean_latency_ms_summary - mean_latency_ms_disabled) / mean_latency_ms_disabled) * 100.0

            print(
                f"{batch_size:<8} {'Summary':<18} {mean_latency_ms_summary:<16.4f} "
                f"{throughput_summary:<22.1f} {f'{overhead_summary:+.2f}%':<24}"
            )

            # -------------------------------------------------------------
            # 3. Detailed Mode
            # -------------------------------------------------------------
            runtime.clear_profiler()
            runtime.enable_profiling(detailed=True)

            for _ in range(warmup_iterations):
                _ = runtime.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_detailed = (t1 - t0) / 1e9
            mean_latency_ms_detailed = (total_sec_detailed / num_iterations) * 1000.0
            throughput_detailed = (num_iterations * batch_size) / total_sec_detailed
            overhead_detailed = ((mean_latency_ms_detailed - mean_latency_ms_disabled) / mean_latency_ms_disabled) * 100.0

            print(
                f"{batch_size:<8} {'Detailed':<18} {mean_latency_ms_detailed:<16.4f} "
                f"{throughput_detailed:<22.1f} {f'{overhead_detailed:+.2f}%':<24}"
            )
            print("-" * 110)

            runtime.close()

    print("=" * 110)
    print("Profiling Overhead Benchmark Completed.")
    print("=" * 110)


if __name__ == "__main__":
    run_benchmark()
