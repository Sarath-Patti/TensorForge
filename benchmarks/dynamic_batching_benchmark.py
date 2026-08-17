"""Inference Scheduling & Dynamic Batching Performance Benchmark for TensorForge.

Compares:
1. Direct InferenceRuntime.predict() baseline.
2. InferenceScheduler with batching disabled (effective max_batch_size = 1).
3. InferenceScheduler with Dynamic Batching enabled (max_batch_size = 32, timeout = 2.0 ms).

Evaluates latency, throughput, queue wait time, execution time, and average batch sizes under concurrent load.
"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    SchedulerConfig,
    SchedulingPolicy,
)
from tensorforge.serialization import save_model


def run_benchmark() -> None:
    print("=" * 115)
    print("TensorForge v1.6 Inference Scheduling & Dynamic Batching Performance Benchmark")
    print("=" * 115)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "dynamic_batching_bench_model.tfmodel")

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

        num_requests = 100
        concurrency = 8
        single_input = tf.randn((1, 64))

        print(f"\n{'Mode':<36} {'Latency (ms)':<16} {'Throughput (s/s)':<22} {'Avg Batch Size':<18} {'Queue Wait (ms)':<18}")
        print("-" * 115)

        # -------------------------------------------------------------
        # 1. Direct InferenceRuntime.predict() (No Scheduler)
        # -------------------------------------------------------------
        runtime_direct = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(1, 64))

        def worker_direct():
            return runtime_direct.predict(single_input)

        # Warmup
        for _ in range(10):
            worker_direct()

        t0 = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker_direct) for _ in range(num_requests)]
            _ = [f.result() for f in futures]
        t1 = time.perf_counter_ns()

        total_sec_direct = (t1 - t0) / 1e9
        mean_lat_direct = (total_sec_direct / num_requests) * 1000.0
        throughput_direct = num_requests / total_sec_direct

        print(f"{'1. Direct Runtime (Baseline)':<36} {mean_lat_direct:<16.4f} {throughput_direct:<22.1f} {'1.00':<18} {'0.000':<18}")
        runtime_direct.close()

        # -------------------------------------------------------------
        # 2. Scheduler with Batching Disabled (max_batch_size = 1)
        # -------------------------------------------------------------
        runtime_nobatch = InferenceRuntime.load(model_path, num_threads=4)
        config_nobatch = SchedulerConfig(max_batch_size=1, max_queue_size=256, batch_timeout_ms=0.0)
        scheduler_nobatch = InferenceScheduler(runtime_nobatch, config=config_nobatch)

        def worker_nobatch():
            return scheduler_nobatch.predict(single_input)

        # Warmup
        for _ in range(10):
            worker_nobatch()

        t0 = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker_nobatch) for _ in range(num_requests)]
            _ = [f.result() for f in futures]
        t1 = time.perf_counter_ns()

        total_sec_nobatch = (t1 - t0) / 1e9
        mean_lat_nobatch = (total_sec_nobatch / num_requests) * 1000.0
        throughput_nobatch = num_requests / total_sec_nobatch
        stats_nobatch = scheduler_nobatch.stats()

        print(
            f"{'2. Scheduler (Batch Size = 1)':<36} {mean_lat_nobatch:<16.4f} {throughput_nobatch:<22.1f} "
            f"{f'{stats_nobatch[\"average_batch_size\"]:.2f}':<18} {f'{stats_nobatch[\"avg_queue_wait_ms\"]:.3f}':<18}"
        )
        scheduler_nobatch.close()
        runtime_nobatch.close()

        # -------------------------------------------------------------
        # 3. Dynamic Batching Scheduler (max_batch_size = 32, timeout = 2.0ms)
        # -------------------------------------------------------------
        runtime_batched = InferenceRuntime.load(model_path, num_threads=4)
        config_batched = SchedulerConfig(max_batch_size=32, max_queue_size=256, batch_timeout_ms=2.0)
        scheduler_batched = InferenceScheduler(runtime_batched, config=config_batched)

        def worker_batched():
            return scheduler_batched.predict(single_input)

        # Warmup
        for _ in range(10):
            worker_batched()

        t0 = time.perf_counter_ns()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(worker_batched) for _ in range(num_requests)]
            _ = [f.result() for f in futures]
        t1 = time.perf_counter_ns()

        total_sec_batched = (t1 - t0) / 1e9
        mean_lat_batched = (total_sec_batched / num_requests) * 1000.0
        throughput_batched = num_requests / total_sec_batched
        stats_batched = scheduler_batched.stats()

        print(
            f"{'3. Dynamic Batching (Max Batch = 32)':<36} {mean_lat_batched:<16.4f} {throughput_batched:<22.1f} "
            f"{f'{stats_batched[\"average_batch_size\"]:.2f}':<18} {f'{stats_batched[\"avg_queue_wait_ms\"]:.3f}':<18}"
        )
        scheduler_batched.close()
        runtime_batched.close()

    print("-" * 115)
    print("Dynamic Batching Performance Benchmark Completed.")
    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
