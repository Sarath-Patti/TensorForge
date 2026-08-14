"""Memory allocation and introspection benchmark for TensorForge."""

import time
import tensorforge as tf
from tensorforge import float32, float64, int32, int64, int8
from tensorforge.tensor.native_storage import NativeStorage, is_native_available


def benchmark_memory():
    print("=" * 80)
    print("TensorForge Memory Introspection & Storage Benchmark")
    print(f"Native Storage Available: {is_native_available()}")
    print("=" * 80)

    tensors = [
        ("Scalar", tf.tensor(3.14, dtype=float32)),
        ("Small 1D (64)", tf.zeros((64,), dtype=float32)),
        ("Medium 2D (512x512)", tf.zeros((512, 512), dtype=float32)),
        ("Large 2D (2048x2048)", tf.zeros((2048, 2048), dtype=float32)),
        ("Int64 3D (32x32x32)", tf.ones((32, 32, 32), dtype=int64)),
        ("Int8 Buffer (1M)", tf.ones((1_000_000,), dtype=int8)),
    ]

    print(f"{'Description':<22} | {'Shape':<16} | {'DType':<8} | {'Numel':<10} | {'Bytes (KB)':<12} | {'Storage Type':<16}")
    print("-" * 80)

    for desc, t in tensors:
        kb = t.nbytes / 1024.0
        storage_type = t.storage.__class__.__name__
        print(f"{desc:<22} | {str(t.shape):<16} | {t.dtype.name:<8} | {t.numel:<10,d} | {kb:<12.2f} | {storage_type:<16}")

    print("=" * 80)

    # NativeStorage check
    print("\nAllocating NativeStorage (100,000 float32 elements)...")
    start = time.perf_counter()
    native_st = NativeStorage(100_000, dtype=float32)
    alloc_time_us = (time.perf_counter() - start) * 1e6
    print(f"Allocated {native_st.nbytes / 1024.0:.2f} KB in {alloc_time_us:.2f} microseconds.")
    print(f"Device: {native_st.device}, Data Pointer: 0x{native_st.data_ptr:x}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_memory()
