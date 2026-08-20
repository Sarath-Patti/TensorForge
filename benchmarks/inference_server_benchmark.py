"""Inference Serving Overhead Benchmark for TensorForge.

Measures and compares:
1. Raw InferenceRuntime execution.
2. InferenceScheduler dynamic batching overhead.
3. InferenceServer single-model request routing overhead.
4. InferenceServer multi-model version resolution overhead.
"""

from __future__ import annotations

import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    InferenceServer,
    SchedulerConfig,
)
from tensorforge.serialization import save_model


def run_benchmark() -> None:
    print("=" * 115)
    print("TensorForge v1.8 Production Inference Serving Layer Overhead Benchmark")
    print("=" * 115)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "server_benchmark_model.tfmodel")

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        batch_sizes = [1, 8, 32]
        num_iterations = 200
        warmup_iterations = 25

        print(f"\n{'Batch':<8} {'Layer Configuration':<36} {'Latency (ms)':<16} {'Throughput (s/s)':<22} {'Overhead vs Baseline':<22}")
        print("-" * 115)

        for batch_size in batch_sizes:
            x_test = tf.randn((batch_size, 64))

            # -------------------------------------------------------------
            # 1. Direct InferenceRuntime Baseline
            # -------------------------------------------------------------
            runtime_base = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(batch_size, 64))

            for _ in range(warmup_iterations):
                _ = runtime_base.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = runtime_base.predict(x_test)
            t1 = time.perf_counter_ns()

            sec_base = (t1 - t0) / 1e9
            latency_base = (sec_base / num_iterations) * 1000.0
            throughput_base = (num_iterations * batch_size) / sec_base

            print(
                f"{batch_size:<8} {'1. Direct InferenceRuntime':<36} {latency_base:<16.4f} "
                f"{throughput_base:<22.1f} {'0.00% (baseline)':<22}"
            )

            # -------------------------------------------------------------
            # 2. InferenceScheduler Layer
            # -------------------------------------------------------------
            runtime_sched = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(batch_size, 64))
            scheduler = InferenceScheduler(runtime_sched, config=SchedulerConfig(max_batch_size=batch_size))

            for _ in range(warmup_iterations):
                _ = scheduler.predict(x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = scheduler.predict(x_test)
            t1 = time.perf_counter_ns()

            sec_sched = (t1 - t0) / 1e9
            latency_sched = (sec_sched / num_iterations) * 1000.0
            throughput_sched = (num_iterations * batch_size) / sec_sched
            overhead_sched = ((latency_sched - latency_base) / latency_base) * 100.0

            print(
                f"{batch_size:<8} {'2. InferenceScheduler Layer':<36} {latency_sched:<16.4f} "
                f"{throughput_sched:<22.1f} {f'{overhead_sched:+.2f}%':<22}"
            )

            # -------------------------------------------------------------
            # 3. InferenceServer Layer (Single Model)
            # -------------------------------------------------------------
            server_single = InferenceServer()
            server_single.load_model(
                name="classifier",
                path=model_path,
                version="1",
                scheduler_config=SchedulerConfig(max_batch_size=batch_size),
                compile_input_shape=(batch_size, 64),
            )

            for _ in range(warmup_iterations):
                _ = server_single.predict("classifier", x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = server_single.predict("classifier", x_test)
            t1 = time.perf_counter_ns()

            sec_srv = (t1 - t0) / 1e9
            latency_srv = (sec_srv / num_iterations) * 1000.0
            throughput_srv = (num_iterations * batch_size) / sec_srv
            overhead_srv = ((latency_srv - latency_base) / latency_base) * 100.0

            print(
                f"{batch_size:<8} {'3. InferenceServer (Single Model)':<36} {latency_srv:<16.4f} "
                f"{throughput_srv:<22.1f} {f'{overhead_srv:+.2f}%':<22}"
            )

            # -------------------------------------------------------------
            # 4. InferenceServer Layer (Multi-Model Registry)
            # -------------------------------------------------------------
            server_multi = InferenceServer()
            for i in range(5):
                server_multi.load_model(
                    name=f"model_{i}",
                    path=model_path,
                    version="1",
                    scheduler_config=SchedulerConfig(max_batch_size=batch_size),
                    compile_input_shape=(batch_size, 64),
                )

            for _ in range(warmup_iterations):
                _ = server_multi.predict("model_2", x_test)

            t0 = time.perf_counter_ns()
            for _ in range(num_iterations):
                _ = server_multi.predict("model_2", x_test)
            t1 = time.perf_counter_ns()

            sec_multi = (t1 - t0) / 1e9
            latency_multi = (sec_multi / num_iterations) * 1000.0
            throughput_multi = (num_iterations * batch_size) / sec_multi
            overhead_multi = ((latency_multi - latency_base) / latency_base) * 100.0

            print(
                f"{batch_size:<8} {'4. InferenceServer (5 Models Registered)':<36} {latency_multi:<16.4f} "
                f"{throughput_multi:<22.1f} {f'{overhead_multi:+.2f}%':<22}"
            )
            print("-" * 115)

            runtime_base.close()
            scheduler.close()
            runtime_sched.close()
            server_single.close()
            server_multi.close()

    print("=" * 115)
    print("Inference Serving Layer Overhead Benchmark Completed.")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
