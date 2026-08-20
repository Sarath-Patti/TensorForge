"""Overhead Benchmark comparing direct InferenceServer vs high-level InferenceClient invocation in TensorForge v2.0."""

import os
import tempfile
import time
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceClient, InferenceServer
from tensorforge.serialization import save_model


def run_client_overhead_benchmark(num_iterations: int = 1000) -> None:
    print("=" * 70)
    print(f"TensorForge v2.0 InferenceClient Overhead Benchmark ({num_iterations} iterations)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        save_model(nn.Linear(8, 4), model_path)

        with InferenceServer() as server:
            server.load_model("benchmark_model", model_path)

            input_data = np.ones((1, 8), dtype=np.float32)

            # Warmup
            for _ in range(50):
                _ = server.predict("benchmark_model", input_data)

            # Direct Server Predict
            start_server = time.perf_counter()
            for _ in range(num_iterations):
                _ = server.predict("benchmark_model", input_data)
            server_duration_ms = (time.perf_counter() - start_server) * 1000.0

            # High-Level Client Predict
            with InferenceClient(server) as client:
                # Warmup
                for _ in range(50):
                    _ = client.predict("benchmark_model", input_data)

                start_client = time.perf_counter()
                for _ in range(num_iterations):
                    _ = client.predict("benchmark_model", input_data)
                client_duration_ms = (time.perf_counter() - start_client) * 1000.0

            server_avg_us = (server_duration_ms / num_iterations) * 1000.0
            client_avg_us = (client_duration_ms / num_iterations) * 1000.0
            overhead_us = client_avg_us - server_avg_us

            print(f"  Direct Server Latency (avg): {server_avg_us:.2f} µs / request")
            print(f"  InferenceClient Latency (avg): {client_avg_us:.2f} µs / request")
            print(f"  Client Wrapper Overhead: {overhead_us:.2f} µs / request")
            print("=" * 70)


if __name__ == "__main__":
    run_client_overhead_benchmark()
