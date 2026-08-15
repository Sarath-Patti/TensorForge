"""Element-wise operations benchmark comparing NumPy, TensorForge NumPy Backend, and Native Backend."""

import time
import numpy as np
import tensorforge as tf
from tensorforge.backend import backend_context, is_native_available


def benchmark_elementwise(sizes=[10_000, 100_000, 1_000_000], num_warmup=5, num_repeats=20):
    native_avail = is_native_available()
    print("=" * 90)
    print("TensorForge Element-wise Operations Benchmark (Float32)")
    print(f"Native C++ Available: {native_avail}")
    print("=" * 90)
    print(f"{'Operation':<10} | {'Elements':<10} | {'NumPy (ms)':<12} | {'TF-NP (ms)':<12} | {'TF-Native (ms)':<15} | {'Throughput (M-Elem/s)':<22}")
    print("-" * 90)

    for op_name, np_op, tf_op in [
        ("Add (a+b)", lambda a, b: a + b, lambda a, b: a + b),
        ("Mul (a*b)", lambda a, b: a * b, lambda a, b: a * b),
    ]:
        for n in sizes:
            a_np = np.random.randn(n).astype(np.float32)
            b_np = np.random.randn(n).astype(np.float32)

            a_tf = tf.tensor(a_np)
            b_tf = tf.tensor(b_np)

            # 1. NumPy Baseline
            for _ in range(num_warmup):
                _ = np_op(a_np, b_np)
            start = time.perf_counter()
            for _ in range(num_repeats):
                _ = np_op(a_np, b_np)
            np_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

            # 2. TensorForge (NumPy Backend)
            with backend_context("numpy"):
                for _ in range(num_warmup):
                    _ = tf_op(a_tf, b_tf)
                start = time.perf_counter()
                for _ in range(num_repeats):
                    _ = tf_op(a_tf, b_tf)
                tf_np_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

            # 3. TensorForge (Native Backend)
            tf_native_time_ms = float("nan")
            if native_avail:
                with backend_context("native"):
                    for _ in range(num_warmup):
                        _ = tf_op(a_tf, b_tf)
                    start = time.perf_counter()
                    for _ in range(num_repeats):
                        _ = tf_op(a_tf, b_tf)
                    tf_native_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

            active_ms = tf_native_time_ms if native_avail else tf_np_time_ms
            throughput = (n / (active_ms / 1000.0)) / 1e6

            native_ms_str = f"{tf_native_time_ms:<15.4f}" if native_avail else f"{'N/A':<15}"
            print(f"{op_name:<10} | {n:<10,d} | {np_time_ms:<12.4f} | {tf_np_time_ms:<12.4f} | {native_ms_str} | {throughput:<22.2f}")

    print("=" * 90)


if __name__ == "__main__":
    benchmark_elementwise()
