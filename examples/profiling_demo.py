"""TensorForge v1.4 – Runtime Observability & Performance Diagnostics Demonstration.

Demonstrates:
1. Constructing, optimizing, and compiling an inference model.
2. Running predictions with profiling disabled (zero overhead baseline).
3. Enabling runtime profiling and collecting telemetry.
4. Generating and displaying rich PerformanceReport summaries, operation breakdowns, and backend distributions.
5. Using scoped ProfileSession context managers.
6. Concurrent multi-threaded prediction observability.
7. Inspecting compiler cache efficiency and workspace memory telemetry.
8. Clean lifecycle shutdown.
"""

from __future__ import annotations

import concurrent.futures
import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def main() -> None:
    print("=" * 100)
    print("TensorForge v1.4 – Runtime Observability & Performance Diagnostics Demonstration")
    print("=" * 100)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "observability_model.tfmodel")

        # --- [Step 1] Construct and Export Model ---
        print("\n--- [Step 1] Constructing and Exporting Neural Network ---")
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path, metadata={"author": "TensorForge Observability Team"})
        print(f"  Exported deep classifier model to: '{os.path.basename(model_path)}'")

        # --- [Step 2] Initializing Runtime & Baseline Execution ---
        print("\n--- [Step 2] Initializing Runtime with Profiling Disabled ---")
        runtime = InferenceRuntime.load(model_path, num_threads=4).compile(input_shape=(16, 32))
        print(f"  Profiling Status: {runtime.profiling_enabled} (Mode: {runtime.profiling_mode})")

        # Run warm-up and unprofiled baseline predictions
        x_warmup = tf.randn((16, 32))
        for _ in range(5):
            _ = runtime.predict(x_warmup)
        print("  ✓ Completed 5 baseline predictions with ZERO profiling telemetry overhead.")
        print(f"    Stored Profile Events: {len(runtime.profile_events())}")

        # --- [Step 3] Enabling Profiling & Collecting Steady-State Telemetry ---
        print("\n--- [Step 3] Enabling Detailed Runtime Profiling ---")
        runtime.enable_profiling(detailed=True)
        print(f"  Profiling Status: {runtime.profiling_enabled} (Mode: {runtime.profiling_mode})")

        # Run batch predictions across multiple sizes
        for batch_sz in [1, 8, 16, 32]:
            x_in = tf.randn((batch_sz, 32))
            for _ in range(10):
                _ = runtime.predict(x_in)
        print(f"  ✓ Executed 40 profiled inference predictions across various batch sizes.")

        # --- [Step 4] Performance Report and Breakdown Diagnostics ---
        print("\n--- [Step 4] Full Performance Report ---")
        report = runtime.profile()
        print(report.summary())

        # --- [Step 5] Scoped ProfileSession Demonstration ---
        print("\n--- [Step 5] Demonstrating Scoped ProfileSession ---")
        runtime.disable_profiling()
        self_disabled = not runtime.profiling_enabled
        print(f"  Global Profiler Disabled: {self_disabled}")

        test_x = tf.randn((8, 32))
        with runtime.profile_session(detailed=True) as session:
            _ = runtime.predict(test_x)

        print(f"  Scoped Session Completed in {session.duration_ms:.3f} ms")
        print(f"  Global Profiler Restored: {not runtime.profiling_enabled}")
        print("  Session Breakdown:")
        print(session.summary())

        # --- [Step 6] Concurrent Predictions with Profiling ---
        print("\n--- [Step 6] Concurrent Multi-Threaded Profiling ---")
        runtime.clear_profiler()
        runtime.enable_profiling(detailed=True)

        num_workers = 6
        requests_per_worker = 15
        x_thread = tf.randn((16, 32))

        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    lambda: [runtime.predict(x_thread) for _ in range(requests_per_worker)]
                )
                for _ in range(num_workers)
            ]
            concurrent.futures.wait(futures)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0

        print(f"  Completed {num_workers * requests_per_worker} concurrent predictions in {t_elapsed:.2f} ms")
        lat_stats = runtime.latency_stats()
        print("  Concurrent Latency Metrics:")
        print(f"    - Prediction Count:    {lat_stats['prediction_count']}")
        print(f"    - Mean Latency:        {lat_stats['mean_ms']:.4f} ms")
        print(f"    - P50 Latency:         {lat_stats['p50_ms']:.4f} ms")
        print(f"    - P95 Latency:         {lat_stats['p95_ms']:.4f} ms")
        print(f"    - P99 Latency:         {lat_stats['p99_ms']:.4f} ms")
        print(f"    - Throughput:          {lat_stats['throughput_samples_per_sec']:.1f} samples/sec")

        # --- [Step 7] Inspecting Diagnostics and Health Report ---
        print("\n--- [Step 7] Runtime Health & Observability Metrics ---")
        health = runtime.health()
        for k, v in health.items():
            print(f"    - {k:<26}: {v}")

        # --- [Step 8] Clean Shutdown ---
        print("\n--- [Step 8] Clean Runtime Shutdown ---")
        runtime.disable_profiling()
        runtime.close()
        print(f"  Runtime is_closed: {runtime.is_closed}")
        print("  ✓ All resources and telemetry stores cleanly closed.")

    print("\n" + "=" * 100)
    print("TensorForge v1.4 Observability Demonstration Finished Successfully!")
    print("=" * 100)


if __name__ == "__main__":
    main()
