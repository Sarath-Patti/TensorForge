"""TensorForge v1.7 – Production Inference Observability & Performance Analytics Demonstration.

Demonstrates:
1. Exporting a neural network model archive.
2. Initializing an InferenceRuntime and InferenceScheduler with dynamic batching.
3. Executing diverse inference predictions.
4. Generating a comprehensive PerformanceSnapshot.
5. Printing human-readable metrics breakdowns:
   - Request counts and lifecycle outcomes
   - Dynamic batching statistics and capacity utilization
   - Latency distributions (min, max, mean, p50, p90, p95, p99)
   - Real-time throughput rates (requests/sec, samples/sec, batches/sec)
   - Backend execution breakdown & fallback telemetry
   - Compiler cache efficiency metrics
   - Memory workspace telemetry
6. Exporting analytics snapshot to a JSON file.
7. Demonstrating metrics reset.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    PerformanceSnapshot,
    SchedulerConfig,
    SchedulingPolicy,
)
from tensorforge.serialization import save_model


def main() -> None:
    print("=" * 105)
    print("TensorForge v1.7 – Production Inference Observability & Analytics Demonstration")
    print("=" * 105)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "deep_classifier_observability.tfmodel")
        metrics_json_path = os.path.join(tmpdir, "performance_snapshot.json")

        # --- [Step 1] Construct and Export Model ---
        print("\n--- [Step 1] Constructing and Exporting Neural Network ---")
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 16),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path, metadata={"milestone": "v1.7"})
        print(f"  Exported model archive to: '{os.path.basename(model_path)}'")

        # --- [Step 2] Initializing Runtime & Scheduler ---
        print("\n--- [Step 2] Initializing Runtime with Dynamic Batching Scheduler ---")
        runtime = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(16, 32))
        config = SchedulerConfig(
            max_batch_size=16,
            max_queue_size=64,
            batch_timeout_ms=3.0,
            policy=SchedulingPolicy.FIFO,
        )
        scheduler = InferenceScheduler(runtime, config=config)

        # --- [Step 3] Executing Diverse Workloads ---
        print("\n--- [Step 3] Executing Diverse Workload Invocations ---")
        workload_batches = [
            tf.randn((2, 32)),
            tf.randn((4, 32)),
            tf.randn((1, 32)),
            tf.randn((8, 32)),
            tf.randn((2, 32)),
            tf.randn((4, 32)),
        ]

        futures = [scheduler.submit(x) for x in workload_batches]
        for i, fut in enumerate(futures):
            _ = fut.result(timeout=2.0)
            print(f"  ✓ Processed request #{i+1} successfully (ID: {fut.request_id})")

        # --- [Step 4] Generating Unified PerformanceSnapshot ---
        print("\n--- [Step 4] Unified Performance Analytics Snapshot ---")
        snapshot = scheduler.performance_snapshot()
        print(snapshot.summary())

        # --- [Step 5] Exporting Snapshot to JSON ---
        print("\n--- [Step 5] Exporting Analytics Snapshot to JSON ---")
        scheduler.export_metrics(metrics_json_path, indent=2)
        print(f"  Successfully exported performance metrics to: '{metrics_json_path}'")

        # Inspect exported JSON
        with open(metrics_json_path, "r", encoding="utf-8") as f:
            exported_data = json.load(f)
        print(f"  Exported JSON Sections: {list(exported_data.keys())}")
        print(f"  Captured Requests: {exported_data['requests']['completed']}, Batches: {exported_data['batches']['batches_formed']}")

        # --- [Step 6] Demonstrating Metrics Reset ---
        print("\n--- [Step 6] Demonstrating Thread-Safe Metrics Reset ---")
        scheduler.reset_metrics()
        reset_snapshot = scheduler.performance_snapshot()
        print(f"  Completed Requests after reset: {reset_snapshot.requests.completed}")
        print(f"  Batches Formed after reset:     {reset_snapshot.batches.batches_formed}")
        print(f"  Latency Sample Count:           {reset_snapshot.latency.execution.sample_count}")

        scheduler.close()
        runtime.close()

    print("\n" + "=" * 105)
    print("TensorForge v1.7 Observability & Performance Analytics Demo Finished Successfully!")
    print("=" * 105)


if __name__ == "__main__":
    main()
