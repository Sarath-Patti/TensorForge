"""TensorForge v1.3 Multi-Threaded Concurrent Inference & Scalability Benchmark.

Compares:
  - Sequential inference vs Concurrent multi-worker inference
  - Python worker thread counts: 1, 2, 4, 8 workers
  - TensorForge native CPU thread counts: 1, 2, 4 threads
  - Batch sizes: 1, 8, 32, 64
  - Measures total latency, per-request latency, throughput (samples/sec), and actual concurrency speedup
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import get_num_threads, is_native_available, set_num_threads
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def benchmark_concurrent_inference(
    batch_sizes=[1, 8, 32, 64],
    worker_counts=[1, 2, 4, 8],
    native_thread_counts=[1, 2, 4],
    num_requests_per_worker=25,
):
    print("=" * 125)
    print("TensorForge v1.3: Multi-Threaded Concurrent Inference & Serving Benchmark")
    native_avail = is_native_available()
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 125)

    np.random.seed(42)
    in_features, hidden_dim, out_features = 32, 64, 10

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "bench_concurrent.tfmodel")

        model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_features),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        for b in batch_sizes:
            print(f"\n==================== Batch Size = {b} ====================")

            for nt in native_thread_counts:
                print(f"\n--- TensorForge Native Threads = {nt} ---")
                print(
                    f"{'Workers':<10} | {'Requests':<10} | {'Total Time (ms)':<18} | {'Avg Latency (ms)':<18} | {'Throughput (samples/s)':<24} | {'Speedup vs 1 Worker':<22}"
                )
                print("-" * 115)

                runtime = InferenceRuntime.load(model_path, num_threads=nt).compile(input_shape=(b, in_features))
                
                # Warmup
                warmup_x = np.random.randn(b, in_features).astype(np.float32)
                for _ in range(10):
                    _ = runtime.predict(warmup_x)

                baseline_time = 1.0

                for workers in worker_counts:
                    total_requests = workers * num_requests_per_worker
                    requests_data = [np.random.randn(b, in_features).astype(np.float32) for _ in range(total_requests)]

                    start_time = time.perf_counter()

                    if workers == 1:
                        for req in requests_data:
                            _ = runtime.predict(req)
                    else:
                        with ThreadPoolExecutor(max_workers=workers) as executor:
                            futures = [executor.submit(runtime.predict, req) for req in requests_data]
                            for f in as_completed(futures):
                                _ = f.result()

                    total_elapsed = time.perf_counter() - start_time
                    total_ms = total_elapsed * 1000.0
                    avg_lat_ms = total_ms / total_requests
                    total_samples = total_requests * b
                    throughput = total_samples / total_elapsed

                    if workers == 1:
                        baseline_time = total_elapsed
                        speedup = 1.0
                    else:
                        expected_seq_time = baseline_time * (total_requests / (1 * num_requests_per_worker))
                        speedup = expected_seq_time / total_elapsed if total_elapsed > 0 else 1.0

                    print(
                        f"{workers:<10} | {total_requests:<10} | {total_ms:<18.2f} | {avg_lat_ms:<18.3f} | {throughput:<24.1f} | {speedup:<22.2f}x"
                    )

                runtime.close()

    print("\n" + "=" * 125)
    print("Concurrent Inference Benchmark Completed.")
    print("=" * 125)


if __name__ == "__main__":
    benchmark_concurrent_inference()
