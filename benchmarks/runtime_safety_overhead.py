"""Runtime Safety & Admission Control Overhead Benchmark for TensorForge.

Measures inference latency and throughput across:
1. Baseline Runtime (Limits disabled/default, Profiling disabled)
2. Safety-Hardened Runtime (Input validation + RuntimeLimits active, Profiling disabled)
3. Full Diagnostics Runtime (Input validation + RuntimeLimits + Profiling enabled)

Evaluates over batch sizes: 1, 8, 32, 128.
"""

from __future__ import annotations

import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.serialization import save_model


def run_benchmark() -> None:
    print("=" * 115)
    print("TensorForge v1.5 Runtime Safety & Admission Control Overhead Benchmark")
    print("=" * 115)
    print("Note: Admission control, input validation, and telemetry tracking introduce necessary safety safeguards.")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "benchmark_safety_model.tfmodel")

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        batch_sizes = [1, 8, 32, 128]
        num_iterations = 200
        warmup_iterations = 25

        print(f"\n{'Batch':<8} {'Configuration':<30} {'Latency (ms)':<16} {'Throughput (s/s)':<22} {'Overhead vs Default':<22}")
        print("-" * 115)

        for batch_size in batch_sizes:
            x_test = tf.randn((batch_size, 64))

            # -------------------------------------------------------------
            # 1. Baseline (Limits Disabled/Default)
            # -------------------------------------------------------------
            runtime_base = InferenceRuntime.load(model_path).compile(input_shape=(batch_size, 64))

            for _ in range(warmup_iterations):
                _ = runtime_base.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_base.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_base = (t1 - t0) / 1e9
            mean_latency_base = (total_sec_base / num_iterations) * 1000.0
            throughput_base = (num_iterations * batch_size) / total_sec_base

            print(
                f"{batch_size:<8} {'Default / Unconstrained':<30} {mean_latency_base:<16.4f} "
                f"{throughput_base:<22.1f} {'0.00% (baseline)':<22}"
            )

            # -------------------------------------------------------------
            # 2. Safety-Hardened (RuntimeLimits Active)
            # -------------------------------------------------------------
            limits = RuntimeLimits(
                max_batch_size=256,
                max_input_elements=16384,
                max_workspace_bytes=10 * 1024 * 1024,
                max_concurrent_requests=16,
            )
            runtime_safe = InferenceRuntime.load(model_path, limits=limits).compile(input_shape=(batch_size, 64))

            for _ in range(warmup_iterations):
                _ = runtime_safe.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_safe.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_safe = (t1 - t0) / 1e9
            mean_latency_safe = (total_sec_safe / num_iterations) * 1000.0
            throughput_safe = (num_iterations * batch_size) / total_sec_safe
            overhead_safe = ((mean_latency_safe - mean_latency_base) / mean_latency_base) * 100.0

            print(
                f"{batch_size:<8} {'Safety Limits Active':<30} {mean_latency_safe:<16.4f} "
                f"{throughput_safe:<22.1f} {f'{overhead_safe:+.2f}%':<22}"
            )

            # -------------------------------------------------------------
            # 3. Full Diagnostics (Limits + Profiling Active)
            # -------------------------------------------------------------
            runtime_diag = InferenceRuntime.load(model_path, limits=limits).compile(input_shape=(batch_size, 64))
            runtime_diag.enable_profiling(detailed=False)

            for _ in range(warmup_iterations):
                _ = runtime_diag.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_diag.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_diag = (t1 - t0) / 1e9
            mean_latency_diag = (total_sec_diag / num_iterations) * 1000.0
            throughput_diag = (num_iterations * batch_size) / total_sec_diag
            overhead_diag = ((mean_latency_diag - mean_latency_base) / mean_latency_base) * 100.0

            print(
                f"{batch_size:<8} {'Limits + Profiling Active':<30} {mean_latency_diag:<16.4f} "
                f"{throughput_diag:<22.1f} {f'{overhead_diag:+.2f}%':<22}"
            )
            print("-" * 115)

            runtime_base.close()
            runtime_safe.close()
            runtime_diag.close()

    print("=" * 115)
    print("Runtime Safety Overhead Benchmark Completed.")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
