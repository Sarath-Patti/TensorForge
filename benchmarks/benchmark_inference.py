"""Inference performance benchmark measuring latency, batching efficiency, and throughput."""

import os
import tempfile
import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def benchmark_inference(batch_sizes=[1, 8, 32, 128], num_warmup=10, num_repeats=50):
    print("=" * 95)
    print("TensorForge Inference Runtime Benchmark")
    native_avail = is_native_available()
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 95)

    np.random.seed(42)
    in_features, hidden_dim, out_features = 64, 128, 10

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "bench_model.tfmodel")

        model = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        backends = ["numpy"]
        if native_avail:
            backends.append("native")

        for backend_name in backends:
            print(f"\n--- Backend: {backend_name.upper()} ---")
            print(
                f"{'Batch Size':<12} | {'Batch Latency (ms)':<20} | {'Per-Sample (µs)':<18} | {'Throughput (samples/s)':<24}"
            )
            print("-" * 80)

            runtime = InferenceRuntime.load(model_path, backend=backend_name)

            for b in batch_sizes:
                x_test = tf.randn((b, in_features))

                # Warmup
                for _ in range(num_warmup):
                    _ = runtime.predict(x_test)

                # Benchmark
                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = runtime.predict(x_test)
                total_time = time.perf_counter() - start

                avg_batch_ms = (total_time / num_repeats) * 1000.0
                per_sample_us = (avg_batch_ms / b) * 1000.0
                throughput = (b * num_repeats) / total_time

                print(
                    f"{b:<12} | {avg_batch_ms:<20.3f} | {per_sample_us:<18.2f} | {throughput:<24.1f}"
                )

    print("=" * 95)


if __name__ == "__main__":
    benchmark_inference()
