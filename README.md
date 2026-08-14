# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.1 – Project Foundation & Tensor Core`

> **Development Status:** `v0.1 (Active Milestone)`
> TensorForge v0.1 establishes the project foundation, memory storage abstractions, dtype registry, and the core multi-dimensional `Tensor` object.

---

## Project Overview

TensorForge is designed to provide explicit control over memory representation, tensor operations, and inference execution without relying on monolithic deep learning framework runtimes. In future milestones, it will expand into automatic differentiation, neural network modules, C++ execution runtimes, and optimized quantized inference.

In **v0.1**, TensorForge delivers a clean, lightweight tensor engine with clear separation between tensor metadata and memory storage.

---

## Current Capabilities & Memory Model (v0.1)

- **Contiguous-Only Tensor Abstraction:**
  - Multi-dimensional array handling with full metadata tracking (`shape`, `ndim`, `numel`, `strides`, `dtype`, `nbytes`, `is_contiguous`).
  - Strict **contiguous memory model**: all tensors maintain contiguous row-major (C-contiguous) layout.
  - Multi-dimensional indexing, slicing, and item assignment: in v0.1, slicing materializes a new contiguous `Tensor` rather than a shared-storage strided view.
  - Bidirectional NumPy conversion: contiguous NumPy arrays of matching dtype are wrapped efficiently; non-contiguous arrays are converted into contiguous storage.
  - Intuitive string representations (`repr` and `str`).

- **Separation of Metadata and Storage:**
  - Abstract `Storage` hierarchy decouples tensor metadata (`Shape`, `Strides`, `DType`) from physical buffer ownership (`NumPyStorage`), laying the foundation for future C++ custom allocators and device backends.

- **Data Types (`DType`):**
  - First-class support for `float32`, `float64`, `int32`, `int64`.
  - Storage-level support for low-precision types (`int8`, `float16`) to prepare for future milestones. Scale/zero-point quantization math and `QuantizedTensor` are intentionally deferred to milestone v0.7.
  - Automatic dtype promotion (`promote_dtypes`) and explicit casting (`.astype()`).

- **Mathematical & Structural Operations:**
  - Element-wise arithmetic (`+`, `-`, `*`, `/`, unary `-`) with NumPy-style broadcasting, returning new contiguous tensors.
  - Matrix multiplication (`@` / `matmul`) supporting 1D, 2D, and batched n-D operations.
  - Structural transformations (`reshape` with `-1` inference, `transpose`, `.T` property): operations materialize new contiguous tensors rather than arbitrary-stride views.
  - Reductions (`sum`, `mean`) over specific axes or all dimensions with `keepdims` support.

- **Developer-Friendly Validation & Errors:**
  - Specific exception hierarchy (`TensorForgeError`, `ShapeError`, `DimensionError`, `DTypeError`, `IndexError_`, `StorageError`) with actionable error messages.

---

## Architecture

```
                    ┌─────────────────────────┐
                    │     Tensor (Metadata)   │
                    │  - shape: (M, N)        │
                    │  - strides: (N, 1)      │
                    │  - dtype: DType         │
                    │  - is_contiguous: True  │
                    └────────────┬────────────┘
                                 │ owns/references
                    ┌────────────▼────────────┐
                    │ Storage (Contiguous Buf)│
                    │  - numel: M * N         │
                    │  - nbytes: numel * size │
                    │  - data_ptr / buffer    │
                    └─────────────────────────┘
```

TensorForge decouples:
1. **Tensor Metadata:** Shape, strides, dtype, and dimensionality describing the tensor.
2. **Physical Storage:** Contiguous byte allocations owned by `Storage` (`NumPyStorage` in v0.1).

### v0.1 Architectural Scope & Limitations
- **Contiguous Invariant:** TensorForge v0.1 operates strictly on contiguous memory buffers.
- **No Arbitrary-Stride Views:** Operations such as `transpose`, `reshape`, and slicing materialize new contiguous `Tensor` instances rather than creating non-contiguous views or shared-storage offset views.
- **Quantization:** Low-precision data types like `int8` are available as contiguous storage types; quantization arithmetic, scales, and zero-points will be implemented in milestone `v0.7`.

---

## Installation

### Prerequisites
- Python 3.11+
- NumPy >= 1.24.0

### Installing from source
```bash
# Clone the repository
git clone https://github.com/your-username/TensorForge.git
cd TensorForge

# Install in editable mode
pip install -e .

# Install optional development dependencies (pytest)
pip install -e ".[dev]"
```

---

## Basic Usage

```python
import tensorforge as tf

# 1. Create tensors
x = tf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
w = tf.tensor([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.5]], dtype=tf.float32)

# 2. Inspect metadata
print(f"Shape: {x.shape}, Dtype: {x.dtype}, Memory: {x.nbytes} bytes")

# 3. Arithmetic with broadcasting
bias = tf.tensor([10.0, 20.0, 30.0], dtype=tf.float32)
y = x + bias

# 4. Matrix Multiplication: (2, 3) @ (3, 2) -> (2, 2)
out = x @ w
print(f"Output:\n{out}")

# 5. Reshape and Transpose
reshaped = x.reshape(3, 2)
transposed = x.T

# 6. Reductions
total = x.sum()
col_means = x.mean(axis=0)
```

Run the included example script:
```bash
python examples/basic_tensor.py
```

---

## Project Structure

```
TensorForge/
├── tensorforge/
│   ├── __init__.py          # Package exports & version
│   ├── tensor/
│   │   ├── __init__.py      # Tensor module interface
│   │   ├── tensor.py        # Core Tensor abstraction & factory methods
│   │   ├── dtype.py         # DType registry & type promotion rules
│   │   ├── shape.py         # Shape, stride computation & broadcasting geometry
│   │   ├── storage.py       # Storage abstraction & NumPyStorage
│   │   └── operations.py   # Arithmetic, matmul, reductions, transformations
│   │
│   └── utils/
│       ├── __init__.py      # Utility exports
│       └── validation.py    # Custom exception hierarchy & shape validators
│
├── tests/
│   └── tensor/
│       └── test_tensor.py   # Comprehensive unit tests
│
├── examples/
│   └── basic_tensor.py      # Working example demonstration
│
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## Development Status & Roadmap

| Milestone | Status | Description |
|---|---|---|
| **v0.1 – Project Foundation & Tensor Core** | **Current** | Tensor abstraction, metadata/storage decoupling, basic ops, broadcasting, dtype handling |
| **v0.2 – Automatic Differentiation** | Upcoming | Dynamic computational graph (DAG), tape-based reverse-mode autograd, gradient accumulation |
| **v0.3 – Neural Network Modules & Layers** | Planned | Parameter abstraction, Modules/Containers, Linear, Conv2D, Activations, Loss functions |
| **v0.4 – Optimizers & Training Pipeline** | Planned | SGD, Adam, AdamW, learning rate schedulers, dataloaders, training loops |
| **v0.5 – Model Serialization & Checkpointing** | Planned | Memory-mapped weight serialization, state_dict format, format converters |
| **v0.6 – C++ Inference Runtime & Custom Allocators** | Planned | Native C++ tensor engine, arena allocator, SIMD/AVX kernels, zero-copy Pybind11 integration |
| **v0.7 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
| **v0.8 – Production Inference Engine & C API** | Planned | High-throughput serving runtime, batching queue, C/C++ embedding API |
