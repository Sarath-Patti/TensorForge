"""Element-wise operations benchmark comparing NumPy and TensorForge."""

import time
import numpy as np
import tensorforge as tf


def benchmark_elementwise(sizes=[10_000, 100_000, 1_000_000], num_warmup=5, num_repeats=20):
    print("=" * 80)
    print("TensorForge Element-wise Operations Benchmark (Float32)")
    print("=" * 80)
    print(f"{'Operation':<10} | {'Num Elements':<14} | {'NumPy (ms)':<14} | {'TF (ms)':<14} | {'TF M-Elem/s':<14}")
    print("-" * 80)

    for op_name, np_op, tf_op in [
        ("Add (a+b)", lambda a, b: a + b, lambda a, b: a + b),
        ("Mul (a*b)", lambda a, b: a * b, lambda a, b: a * b),
    ]:
        for n in sizes:
            a_np = np.random.randn(n).astype(np.float32)
            b_np = np.random.randn(n).astype(np.float32)

            a_tf = tf.tensor(a_np)
            b_tf = tf.tensor(b_np)

            # NumPy
            for _ in range(num_warmup):
                _ = np_op(a_np, b_np)
            start = time.perf_counter()
            for _ in range(num_repeats):
                _ = np_op(a_np, b_np)
            np_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

            # TensorForge
            for _ in range(num_warmup):
                _ = tf_op(a_tf, b_tf)
            start = time.perf_counter()
            for _ in range(num_repeats):
                _ = tf_op(a_tf, b_tf)
            tf_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

            throughput = (n / (tf_time_ms / 1000.0)) / 1e6

            print(f"{op_name:<10} | {n:<14,d} | {np_time_ms:<14.4f} | {tf_time_ms:<14.4f} | {throughput:<14.2f}")

    print("=" * 80)


if __name__ == "__main__":
    benchmark_elementwise()
