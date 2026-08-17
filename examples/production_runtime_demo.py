"""TensorForge v1.5 – Production Reliability, Resource Management & Runtime Safety Demonstration.

Demonstrates:
1. Loading a serialized model with configurable RuntimeLimits.
2. Inspecting active runtime configuration and limits.
3. Executing valid inference predictions with input validation safeguards.
4. Attempting oversized batches and verifying clean admission rejection (RuntimeLimitError).
5. Demonstrating concurrent request capacity limits (RuntimeBusyError).
6. Demonstrating robust failure recovery (failed requests never corrupt subsequent requests).
7. Inspecting comprehensive operational health() and stats() diagnostics.
8. Scoped profiling integration with request-level metadata.
9. Graceful runtime shutdown and verifying post-closure rejection (RuntimeClosedError).
"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits, RuntimeState
from tensorforge.serialization import save_model
from tensorforge.utils.validation import (
    RuntimeBusyError,
    RuntimeClosedError,
    RuntimeLimitError,
    TensorForgeInputError,
)


def main() -> None:
    print("=" * 105)
    print("TensorForge v1.5 – Production Reliability, Resource Management & Runtime Safety Demo")
    print("=" * 105)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "production_classifier.tfmodel")

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
        save_model(model, model_path, metadata={"environment": "production"})
        print(f"  Exported production model archive to: '{os.path.basename(model_path)}'")

        # --- [Step 2] Configuring RuntimeLimits and Initializing Runtime ---
        print("\n--- [Step 2] Configuring Production Resource Limits ---")
        limits = RuntimeLimits(
            max_batch_size=16,
            max_input_elements=1024,
            max_workspace_bytes=1024 * 1024,  # 1 MB
            max_concurrent_requests=4,
            max_prediction_time_ms=500.0,
        )
        runtime = InferenceRuntime.load(model_path, num_threads=4, limits=limits)
        print("  Configured Runtime Limits:")
        for k, v in runtime.limits().items():
            print(f"    - {k:<26}: {v}")
        print(f"  Lifecycle State: {runtime.lifecycle_state} (is_ready={runtime.is_ready})")

        # --- [Step 3] Executing Valid Prediction with Input Validation ---
        print("\n--- [Step 3] Executing Valid Inference Predictions ---")
        x_valid = tf.randn((8, 32))
        out_valid = runtime.predict(x_valid)
        print(f"  ✓ Prediction completed successfully: input={x_valid.shape} -> output={out_valid.shape}")

        # --- [Step 4] Batch Size & Input Element Limits Protection ---
        print("\n--- [Step 4] Testing Admission Control & Oversized Batch Rejection ---")
        x_oversized = tf.randn((32, 32))  # 32 > max_batch_size (16)
        try:
            runtime.predict(x_oversized)
            print("  ❌ ERROR: Oversized batch was not rejected!")
        except RuntimeLimitError as e:
            print(f"  ✓ Successfully caught expected RuntimeLimitError: {e}")

        # --- [Step 5] Input Validation Protection (Invalid rank / NaNs) ---
        print("\n--- [Step 5] Testing Input Validation Safeguards ---")
        x_nan = tf.from_numpy(np.full((4, 32), np.nan, dtype=np.float32))
        try:
            runtime.predict(x_nan)
            print("  ❌ ERROR: Non-finite input was not rejected!")
        except TensorForgeInputError as e:
            print(f"  ✓ Successfully caught expected TensorForgeInputError: {e}")

        # --- [Step 6] Concurrent Request Capacity Protection ---
        print("\n--- [Step 6] Testing Concurrent Request Capacity Protection ---")
        busy_rejections = 0
        successful_requests = 0

        def concurrent_client(client_id: int):
            nonlocal busy_rejections, successful_requests
            try:
                # Prediction with valid batch
                _ = runtime.predict(tf.randn((2, 32)))
                successful_requests += 1
            except RuntimeBusyError:
                busy_rejections += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(concurrent_client, i) for i in range(25)]
            concurrent.futures.wait(futures)

        print(f"  Dispatched 25 concurrent requests:")
        print(f"    - Successful predictions: {successful_requests}")
        print(f"    - Capacity busy rejections: {busy_rejections}")
        print(f"    - Active requests after drain: {runtime.active_contexts}")

        # --- [Step 7] Robust Failure Recovery ---
        print("\n--- [Step 7] Demonstrating Seamless Failure Recovery ---")
        # Request A: Intentionally invalid feature dimension
        try:
            runtime.predict(tf.randn((4, 10)))  # 10 != 32
        except TensorForgeInputError:
            pass

        # Request B: Valid input immediately after failure
        out_recovered = runtime.predict(tf.randn((4, 32)))
        print(f"  ✓ Request after failure succeeded cleanly: output shape {out_recovered.shape}")
        print(f"  ✓ Active execution contexts: {runtime.active_contexts}")

        # --- [Step 8] Operational Health Diagnostics ---
        print("\n--- [Step 8] Operational Health Diagnostics ---")
        health = runtime.health()
        for k, v in health.items():
            print(f"    - {k:<28}: {v}")

        # --- [Step 9] Extended Runtime Statistics ---
        print("\n--- [Step 9] Extended Runtime Reliability Statistics ---")
        stats = runtime.stats()
        print(f"    - Accepted Requests:           {stats['accepted_requests']}")
        print(f"    - Completed Requests:          {stats['completed_requests']}")
        print(f"    - Failed Requests:             {stats['failed_requests']}")
        print(f"    - Rejected Requests:           {stats['rejected_requests']}")
        print(f"    - Peak Active Requests:        {stats['peak_active_requests']}")
        print(f"    - Input Validation Failures:   {stats['input_validation_failures']}")
        print(f"    - Resource Limit Failures:     {stats['resource_limit_failures']}")
        print(f"    - Last Error Message:          {health['last_error']}")

        # --- [Step 10] Graceful Shutdown & Post-Shutdown Protection ---
        print("\n--- [Step 10] Graceful Shutdown & Lifecycle Enforcement ---")
        runtime.close()
        print(f"  Runtime is_closed: {runtime.is_closed} (State: {runtime.lifecycle_state})")

        try:
            runtime.predict(tf.randn((2, 32)))
            print("  ❌ ERROR: Predict after shutdown was not rejected!")
        except RuntimeClosedError as e:
            print(f"  ✓ Successfully caught expected RuntimeClosedError: {e}")

    print("\n" + "=" * 105)
    print("TensorForge v1.5 Production Runtime Safety Demo Finished Successfully!")
    print("=" * 105)


if __name__ == "__main__":
    main()
