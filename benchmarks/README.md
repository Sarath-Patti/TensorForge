# TensorForge Benchmark Suite

This directory contains lightweight performance and memory benchmarks comparing TensorForge operations with baseline NumPy implementations.

---

## Benchmark Scripts

1. **`benchmark_matmul.py`**:
   - Benchmarks float32 matrix multiplication at sizes:
     - 128 x 128
     - 256 x 256
     - 512 x 512
   - Compares NumPy (`@`), TensorForge Python Tensor, and Native C++ kernel (when compiled).

2. **`benchmark_elementwise.py`**:
   - Benchmarks element-wise float32 addition and multiplication at sizes:
     - 10,000 elements (10K)
     - 100,000 elements (100K)
     - 1,000,000 elements (1M)
   - Measures execution time and throughput (M elements/sec).

3. **`benchmark_memory.py`**:
   - Analyzes memory overhead, allocation sizes, item sizes, and data pointer reporting across different tensor configurations.

---

## How to Run

```bash
# Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py

# Element-wise operations benchmark
python benchmarks/benchmark_elementwise.py

# Memory inspection benchmark
python benchmarks/benchmark_memory.py
```
