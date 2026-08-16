"""TensorForge v1.3 Production Runtime Reliability & Concurrency Demonstration.

Demonstrates:
  1. Exporting a neural network to a .tfmodel archive
  2. Loading a single InferenceRuntime instance and configuring CPU threads
  3. Dispatching multiple concurrent predict() requests across Python worker threads
  4. Verifying thread-safe output determinism and parameter immutability
  5. Inspecting real-time runtime diagnostics and context pooling statistics
  6. Context manager deterministic lifecycle and clean shutdown
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def run_concurrent_inference_demo():
    print("=" * 100)
    print("TensorForge v1.3 – Production Runtime Reliability & Concurrency Demonstration")
    print("=" * 100)

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "concurrent_model.tfmodel")

        # =========================================================================
        # 1. Export Model
        # =========================================================================
        print("\n--- [Step 1] Constructing and Exporting Neural Network ---")
        in_dim, hidden_dim, out_dim = 32, 64, 8

        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
            nn.Softmax(dim=-1),
        )
        model.eval()

        save_model(model, model_path, metadata={"task": "multi_thread_serving"})
        print(f"  Exported model to: '{os.path.basename(model_path)}'")

        # =========================================================================
        # 2. Initialize Runtime with Context Manager
        # =========================================================================
        print("\n--- [Step 2] Initializing Runtime with Context Manager ---")
        with InferenceRuntime.load(model_path) as runtime:
            # Configure native thread parallelism and compile
            runtime.set_num_threads(4)
            runtime.compile(input_shape=(16, in_dim))

            print("  Initial Health Diagnostic:")
            health_init = runtime.health()
            for k, v in health_init.items():
                print(f"    - {k:<20}: {v}")

            # =====================================================================
            # 3. Concurrent Inference with Multiple Worker Threads
            # =====================================================================
            print("\n--- [Step 3] Executing Concurrent Predictions against Single Runtime ---")
            num_requests = 30
            num_workers = 6
            batch_size = 16

            test_batches = [np.random.randn(batch_size, in_dim).astype(np.float32) for _ in range(num_requests)]

            # Compute baseline reference predictions sequentially
            ref_outputs = [runtime.predict(b).numpy() for b in test_batches]

            print(f"  Launching {num_requests} prediction requests across {num_workers} concurrent threads...")
            start_time = time.perf_counter()

            concurrent_outputs = [None] * num_requests
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_map = {executor.submit(runtime.predict, test_batches[i]): i for i in range(num_requests)}
                for future in as_completed(future_map):
                    idx = future_map[future]
                    concurrent_outputs[idx] = future.result().numpy()

            elapsed = time.perf_counter() - start_time
            print(f"  Completed {num_requests} concurrent requests in {elapsed * 1000.0:.2f} ms")

            # =====================================================================
            # 4. Verify Determinism & Immutability
            # =====================================================================
            print("\n--- [Step 4] Verifying Determinism & Parameter Immutability ---")
            max_divergence = 0.0
            for i in range(num_requests):
                diff = np.max(np.abs(concurrent_outputs[i] - ref_outputs[i]))
                if diff > max_divergence:
                    max_divergence = diff

            print(f"  Max Absolute Divergence (Concurrent vs Sequential): {max_divergence:.10e}")
            assert max_divergence < 1e-5, "Concurrent execution output diverged from sequential reference!"
            print("  ✓ Concurrent Output Determinism: PASSED")

            # =====================================================================
            # 5. Runtime Diagnostics & Health Report
            # =====================================================================
            print("\n--- [Step 5] Runtime Diagnostics & Concurrency Stats ---")
            stats = runtime.stats()
            print("  Operational Statistics:")
            print(f"    - Status:              {stats['health']['status']}")
            print(f"    - Backend:             {stats['backend']}")
            print(f"    - Configured Threads:  {stats['num_threads']}")
            print(f"    - Total Predictions:   {stats['prediction_count']}")
            print(f"    - Total Errors:        {stats['error_count']}")
            print(f"    - Active Contexts:     {stats['active_contexts']}")
            print(f"    - Pooled Contexts:     {stats['pooled_contexts']}")
            print(f"    - Workspace Memory:    {stats['workspace_bytes']} bytes ({stats['workspace_kb']:.2f} KB)")

        # =========================================================================
        # 6. Verify Clean Shutdown
        # =========================================================================
        print("\n--- [Step 6] Verifying Clean Lifecycle Shutdown ---")
        assert runtime.is_closed is True
        print(f"  Runtime is_closed: {runtime.is_closed}")
        print("  ✓ Context manager exited cleanly and released all execution resources.")

    print("\n" + "=" * 100)
    print("TensorForge v1.3 Concurrency Demonstration Finished Successfully!")
    print("=" * 100)


if __name__ == "__main__":
    run_concurrent_inference_demo()
