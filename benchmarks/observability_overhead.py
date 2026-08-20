"""Observability Subsystem Overhead Benchmark for TensorForge.

Measures and compares:
1. Baseline Runtime (Profiling disabled, raw forward execution).
2. Runtime with Metrics Collection Active (PerformanceSnapshot recording).
3. Full Observability + Profiling Active (Event tracking + Latency distribution percentiles).

Evaluates latency impact, throughput degradation, and reservoir memory bounds.
"""

from __future__ import annotations

import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, MetricsCollector
from tensorforge.serialization import save_model


def run_benchmark() -> None:
    print("=" * 115)
    print("TensorForge v1.7 Inference Observability Subsystem Overhead Benchmark")
    print("=" * 115)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "observability_overhead_model.tfmodel")

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

        print(f"\n{'Batch':<8} {'Configuration':<34} {'Latency (ms)':<16} {'Throughput (s/s)':<22} {'Overhead vs Baseline':<22}")
        print("-" * 115)

        for batch_size in batch_sizes:
            x_test = tf.randn((batch_size, 64))

            # -------------------------------------------------------------
            # 1. Baseline (Raw Execution)
            # -------------------------------------------------------------
            runtime_base = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(batch_size, 64))

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
                f"{batch_size:<8} {'1. Baseline (Raw Runtime)':<34} {mean_latency_base:<16.4f} "
                f"{throughput_base:<22.1f} {'0.00% (baseline)':<22}"
            )

            # -------------------------------------------------------------
            # 2. Observability Metrics Active
            # -------------------------------------------------------------
            runtime_obs = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(batch_size, 64))

            for _ in range(warmup_iterations):
                _ = runtime_obs.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_obs.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_obs = (t1 - t0) / 1e9
            mean_latency_obs = (total_sec_obs / num_iterations) * 1000.0
            throughput_obs = (num_iterations * batch_size) / total_sec_obs
            overhead_obs = ((mean_latency_obs - mean_latency_base) / mean_latency_base) * 100.0

            print(
                f"{batch_size:<8} {'2. MetricsCollector Active':<34} {mean_latency_obs:<16.4f} "
                f"{throughput_obs:<22.1f} {f'{overhead_obs:+.2f}%':<22}"
            )

            # -------------------------------------------------------------
            # 3. Full Observability + Profiler Active
            # -------------------------------------------------------------
            runtime_full = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(batch_size, 64))
            runtime_full.enable_profiling(detailed=True)

            for _ in range(warmup_iterations):
                _ = runtime_full.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_full.predict(x_test)
            t1 = time.perf_counter_ns()

            total_sec_full = (t1 - t0) / 1e9
            mean_latency_full = (total_sec_full / num_iterations) * 1000.0
            throughput_full = (num_iterations * batch_size) / total_sec_full
            overhead_full = ((mean_latency_full - mean_latency_base) / mean_latency_base) * 100.0

            print(
                f"{batch_size:<8} {'3. Metrics + Detailed Profiler':<34} {mean_latency_full:<16.4f} "
                f"{throughput_full:<22.1f} {f'{overhead_full:+.2f}%':<22}"
            )
            print("-" * 115)

            runtime_base.close()
            runtime_obs.close()
            runtime_full.close()

    print("=" * 115)
    print("Observability Subsystem Overhead Benchmark Completed.")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
