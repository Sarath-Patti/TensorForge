"""Matrix multiplication benchmark comparing NumPy, TensorForge Python, and Native C++."""

import time
import numpy as np
import tensorforge as tf
from tensorforge.native import is_native_available


def benchmark_matmul(sizes=[128, 256, 512], num_warmup=3, num_repeats=10):
    print("=" * 70)
    print("TensorForge Matrix Multiplication Benchmark (Float32)")
    print(f"Native C++ Available: {is_native_available()}")
    print("=" * 70)
    print(f"{'Matrix Size':<15} | {'NumPy (ms)':<15} | {'TensorForge (ms)':<18} | {'GFLOP/s (TF)':<12}")
    print("-" * 70)

    for n in sizes:
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)

        a_tf = tf.tensor(a_np)
        b_tf = tf.tensor(b_np)

        # 1. NumPy Warmup & Benchmark
        for _ in range(num_warmup):
            _ = a_np @ b_np
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = a_np @ b_np
        numpy_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        # 2. TensorForge Warmup & Benchmark
        for _ in range(num_warmup):
            _ = a_tf @ b_tf
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = a_tf @ b_tf
        tf_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        # Compute GFLOP/s: 2 * N^3 operations / (time in seconds) / 1e9
        flops = 2.0 * (n ** 3)
        gflops = (flops / (tf_time_ms / 1000.0)) / 1e9

        print(f"{f'{n}x{n}':<15} | {numpy_time_ms:<15.3f} | {tf_time_ms:<18.3f} | {gflops:<12.2f}")

    print("=" * 70)


if __name__ == "__main__":
    benchmark_matmul()
