"""Quantization benchmark comparing FP32 and INT8 memory, accuracy, and inference latency."""

import time
import numpy as np
import tensorforge as tf
from tensorforge.backend import is_native_available
from tensorforge.quantization import compare_tensors, qmatmul, quantize


def benchmark_quantization(sizes=[128, 256, 512, 1024], num_warmup=3, num_repeats=10):
    np.random.seed(42)
    native_avail = is_native_available()

    print("=" * 95)
    print("TensorForge Quantization & INT8 Inference Benchmark")
    print(f"Native C++ Backend Available: {native_avail}")
    print("=" * 95)
    print(
        f"{'Matrix Size':<12} | {'FP32 Mem (KB)':<14} | {'INT8 Mem (KB)':<14} | {'Max Abs Err':<12} | {'MAE':<10} | {'FP32 (ms)':<10} | {'INT8 (ms)':<10}"
    )
    print("-" * 95)

    for n in sizes:
        a_np = np.random.randn(n, n).astype(np.float32)
        b_np = np.random.randn(n, n).astype(np.float32)

        a_fp = tf.tensor(a_np)
        b_fp = tf.tensor(b_np)

        # 1. Measure Memory
        fp32_mem_kb = (a_fp.nbytes + b_fp.nbytes) / 1024.0

        a_q = quantize(a_fp, scheme="symmetric")
        b_q = quantize(b_fp, scheme="symmetric")
        int8_mem_kb = (a_q.nbytes + b_q.nbytes) / 1024.0

        # 2. Measure Quantization Error
        c_fp = a_fp @ b_fp
        c_int8 = qmatmul(a_q, b_q)
        metrics = compare_tensors(c_fp, c_int8)

        # 3. Measure FP32 Latency
        for _ in range(num_warmup):
            _ = a_fp @ b_fp
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = a_fp @ b_fp
        fp32_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        # 4. Measure INT8 Latency
        for _ in range(num_warmup):
            _ = qmatmul(a_q, b_q)
        start = time.perf_counter()
        for _ in range(num_repeats):
            _ = qmatmul(a_q, b_q)
        int8_time_ms = ((time.perf_counter() - start) / num_repeats) * 1000.0

        print(
            f"{f'{n}x{n}':<12} | {fp32_mem_kb:<14.2f} | {int8_mem_kb:<14.2f} | {metrics['max_abs_error']:<12.4f} | {metrics['mean_abs_error']:<10.4f} | {fp32_time_ms:<10.3f} | {int8_time_ms:<10.3f}"
        )

    print("=" * 95)


if __name__ == "__main__":
    benchmark_quantization()
