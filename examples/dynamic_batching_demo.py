"""TensorForge v1.6 – Production Inference Scheduling & Dynamic Batching Demonstration.

Demonstrates:
1. Loading a model archive into an InferenceRuntime.
2. Initializing an InferenceScheduler with configured dynamic batching and timeout bounds.
3. Submitting concurrent asynchronous requests with varied sub-batch sizes.
4. Inspecting dynamic batch aggregation and result demultiplexing.
5. Demonstrating bounded queue backpressure (SchedulerQueueFullError).
6. Inspecting operational health and statistical reports.
7. Graceful scheduler draining and shutdown.
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
from tensorforge.utils.validation import SchedulerClosedError, SchedulerQueueFullError


def main() -> None:
    print("=" * 105)
    print("TensorForge v1.6 – Production Inference Scheduling & Dynamic Batching Demonstration")
    print("=" * 105)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "classifier_scheduler_demo.tfmodel")

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
        save_model(model, model_path, metadata={"stage": "v1.6-demo"})
        print(f"  Exported model archive to: '{os.path.basename(model_path)}'")

        # --- [Step 2] Initializing InferenceRuntime and InferenceScheduler ---
        print("\n--- [Step 2] Initializing Runtime and Dynamic Batching Scheduler ---")
        runtime = InferenceRuntime.load(model_path, num_threads=4)

        config = SchedulerConfig(
            max_batch_size=16,
            max_queue_size=64,
            batch_timeout_ms=5.0,
            policy=SchedulingPolicy.FIFO,
            drain_on_close=True,
        )
        scheduler = InferenceScheduler(runtime, config=config)
        print(f"  Active Scheduler Configuration: {scheduler.config}")
        print(f"  Scheduler State: {scheduler.lifecycle_state} (is_running={scheduler.is_running})")

        # --- [Step 3] Submitting Concurrent Asynchronous Requests ---
        print("\n--- [Step 3] Dispatching Multiple Concurrent Client Requests ---")
        client_inputs = [
            tf.randn((2, 32)),
            tf.randn((4, 32)),
            tf.randn((1, 32)),
            tf.randn((8, 32)),
            tf.randn((1, 32)),
        ]

        futures = []
        for i, x in enumerate(client_inputs):
            fut = scheduler.submit(x)
            print(f"  Enqueued client request #{i+1}: shape={x.shape}, id='{fut.request_id}'")
            futures.append(fut)

        # Collect and verify demultiplexed outputs
        print("\n--- [Step 4] Demultiplexing and Verifying Individual Results ---")
        for i, (fut, inp) in enumerate(zip(futures, client_inputs)):
            out = fut.result(timeout=2.0)
            expected_shape = (inp.shape[0], 16)
            print(f"  ✓ Request #{i+1} completed: input {inp.shape} -> output {out.shape}")
            assert out.shape == expected_shape

        # --- [Step 5] Demonstrating Bounded Queue Backpressure ---
        print("\n--- [Step 5] Demonstrating Bounded Queue Backpressure Enforcement ---")
        tight_config = SchedulerConfig(max_batch_size=4, max_queue_size=2, batch_timeout_ms=100.0)
        tight_scheduler = InferenceScheduler(runtime, config=tight_config)

        f1 = tight_scheduler.submit(tf.randn((1, 32)))
        f2 = tight_scheduler.submit(tf.randn((1, 32)))

        try:
            tight_scheduler.submit(tf.randn((1, 32)))
            print("  ❌ ERROR: Queue capacity violation was not rejected!")
        except SchedulerQueueFullError as e:
            print(f"  ✓ Successfully caught expected SchedulerQueueFullError: {e}")

        tight_scheduler.close(drain=True)

        # --- [Step 6] Operational Diagnostics and Telemetry ---
        print("\n--- [Step 6] Scheduler Health Diagnostics ---")
        health = scheduler.health()
        for k, v in health.items():
            if k != "runtime_health":
                print(f"    - {k:<26}: {v}")

        print("\n--- [Step 7] Scheduler Telemetry Statistics ---")
        stats = scheduler.stats()
        print(f"    - Submitted Requests:      {stats['submitted_requests']}")
        print(f"    - Completed Requests:      {stats['completed_requests']}")
        print(f"    - Failed Requests:         {stats['failed_requests']}")
        print(f"    - Rejected Requests:       {stats['rejected_requests']}")
        print(f"    - Batches Formed:          {stats['batches_formed']}")
        print(f"    - Average Batch Size:      {stats['average_batch_size']:.2f}")
        print(f"    - Max Batch Size Observed: {stats['max_batch_size_observed']}")
        print(f"    - Avg Queue Wait Time:     {stats['avg_queue_wait_ms']:.3f} ms")
        print(f"    - Avg Execution Time:      {stats['avg_batch_execution_ms']:.3f} ms")

        # --- [Step 8] Graceful Shutdown ---
        print("\n--- [Step 8] Graceful Scheduler Shutdown ---")
        scheduler.close()
        print(f"  Scheduler Closed: {scheduler.is_closed} (State: {scheduler.lifecycle_state})")

        try:
            scheduler.predict(tf.randn((1, 32)))
            print("  ❌ ERROR: Predict after shutdown was not rejected!")
        except SchedulerClosedError as e:
            print(f"  ✓ Successfully caught expected SchedulerClosedError: {e}")

        runtime.close()

    print("\n" + "=" * 105)
    print("TensorForge v1.6 Dynamic Batching Demo Completed Successfully!")
    print("=" * 105)


if __name__ == "__main__":
    main()
