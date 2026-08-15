"""Matrix multiplication benchmark comparing NumPy, TensorForge NumPy Backend, and Native C++ Backend."""

import time
import numpy as np
import tensorforge as tf
from tensorforge.backend import backend_context, is_native_available


def benchmark_matmul(sizes=[128, 256, 512], num_warmup=3, num_repeats=10):
    native_avail = is_native_available()
    print("=" * 85)
    print("TensorForge Matrix Multiplication Benchmark (Float32)")
    print(f"Native C++ Available: {native_avail}")
    print("=" * 85)
    print(f"{'Matrix Size':<12} | {'NumPy (ms)':<14} | {'TF-NumPy (ms)':<16} | {'TF-Native (ms)':<16} | {'Speedup (Native/TF-NP)':<22}")
    print("-" * 85)

    for n in sizes:
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)

        a_tf = tf.tensor(a_np)
        b_tf = tf.tensor(b_np)

        # 1. NumPy Baseline
        for _ in range(num_warmup):
            _ = a_np @ b_np
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = a_np @ b_np
        numpy_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        # 2. TensorForge (NumPy Backend)
        with backend_context("numpy"):
            for _ in range(num_warmup):
                _ = a_tf @ b_tf
            start = time.perf_counter()
            for _ in range(num_repeats):
                _ = a_tf @ b_tf
            tf_np_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        # 3. TensorForge (Native Backend)
        tf_native_time_ms = float("nan")
        speedup_str = "N/A"
        if native_avail:
            with backend_context("native"):
                for _ in range(num_warmup):
                    _ = a_tf @ b_tf
                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = a_tf @ b_tf
                tf_native_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0
                speedup = tf_np_time_ms / tf_native_time_ms
                speedup_str = f"{speedup:.2f}x"

        native_ms_str = f"{tf_native_time_ms:<16.3f}" if native_avail else f"{'N/A':<16}"
        print(f"{f'{n}x{n}':<12} | {numpy_time_ms:<14.3f} | {tf_np_time_ms:<16.3f} | {native_ms_str} | {speedup_str:<22}")

    print("=" * 85)


if __name__ == "__main__":
    benchmark_matmul()
