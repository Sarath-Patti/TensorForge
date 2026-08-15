# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.6 – Native Operation Dispatch & Runtime Integration`

> **Development Status:** `v0.6 (Active Milestone)`
> TensorForge v0.6 introduces a flexible **Backend Dispatcher Layer** that seamlessly integrates native C++ compute kernels (`add`, `sub`, `mul`, `matmul`) into Python `Tensor` operations while preserving `NumPy` as the robust default and automatic fallback backend.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, and model training without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.6**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation DAG engine (`autograd`).
- Neural network layers & activations (`Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, `CrossEntropyLoss`, `Sequential`).
- Optimization & Training pipeline (`SGD`, `Adam`, `TensorDataset`, `DataLoader`, `Trainer`, `accuracy`).
- **C++17 Native Runtime Subsystem (`native/`):**
  - Aligned 64-byte `DefaultCPUAllocator` with active memory tracking.
  - Native C++ `Storage` with RAII memory lifetime ownership.
  - Native C++ `Tensor` and `Shape` representations preserving row-major contiguous layout.
  - Handcrafted CPU compute kernels: element-wise arithmetic and cache-aware $(i, k, j)$ matrix multiplication.
- **Backend Dispatcher Subsystem (`tensorforge/backend/`):**
  - Explicit runtime backend selection (`"numpy"` vs `"native"`).
  - Robust automatic fallback to NumPy when native kernels are not eligible (e.g. broadcasting or non-float32 dtypes).
  - Real-time execution tracking (`get_last_backend()`).
  - Seamless Python autograd backpropagation across both native and NumPy forward computations.
- **Benchmarking Suite (`benchmarks/`):**
  - Multi-backend benchmarks comparing NumPy baseline, TensorForge NumPy backend, and TensorForge Native backend.

---

## Operation Dispatch Architecture

```
                         Tensor Operations (a + b, a @ b)
                                       │
                                       ▼
                            Backend Dispatcher Layer
                                       │
                    ┌──────────────────┴──────────────────┐
                    │                                     │
             Backend: "numpy"                      Backend: "native"
            (Default Backend)                     (Explicit Selection)
                    │                                     │
                    ▼                                     ▼
             NumPy Backend                     Is Native Op Supported?
          (NumPy CPU Kernels)                   (Float32, Shapes Match)
                    │                                /            \
                    │                          [YES]               [NO]
                    │                            │                   │
                    │                            ▼                   ▼
                    │                     Native Backend       NumPy Fallback
                    │                    (C++17 Kernels)     (Explicit Fallback)
                    │                            │                   │
                    └────────────────────────────┼───────────────────┘
                                                 │
                                                 ▼
                                        Result Tensor / Autograd
```

---

## Backend Selection & Control API

```python
import tensorforge as tf

# Check default backend (always 'numpy')
print(tf.get_backend())  # Output: 'numpy'

# Switch to Native C++ execution (requires compiled native extension)
tf.set_backend("native")

# Execute operations
a = tf.randn((256, 256), dtype=tf.float32)
b = tf.randn((256, 256), dtype=tf.float32)
c = a @ b

# Check which backend executed the operation
print(tf.get_last_backend())  # Output: 'native'

# Scoped execution context
with tf.backend_context("numpy"):
    d = a + b
    print(tf.get_last_backend())  # Output: 'numpy'
```

---

## Supported Native Operations & Fallback Behavior

| Operation | Native C++ Fast Path | Automatic Fallback Conditions |
|---|---|---|
| **`matmul` (`@`)** | 2D `float32` matrices: $(M, K) \times (K, N) \to (M, N)$ | 1D vectors, Batched 3D+ tensors, non-`float32` dtypes |
| **`add` (`+`)** | Same-shape `float32` tensors | Multi-dimensional broadcasting, scalar operands, non-`float32` dtypes |
| **`sub` (`-`)** | Same-shape `float32` tensors | Multi-dimensional broadcasting, scalar operands, non-`float32` dtypes |
| **`mul` (`*`)** | Same-shape `float32` tensors | Multi-dimensional broadcasting, scalar operands, non-`float32` dtypes |
| **Other Ops** | Division, Negation, Reductions, Activations, Losses | Executed via reference NumPy backend |

---

## Building the Native C++ Runtime

### Prerequisites
- CMake >= 3.15
- C++17 compatible compiler (Clang, GCC, or MSVC)
- pybind11 (optional, for Python C-extension bindings)

### Standalone C++ Build
```bash
# Configure and build native C++ library and tests
cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
cmake --build native/build

# Run standalone C++ native verification tests
ctest --test-dir native/build --output-on-failure
```

---

## Running Benchmarks & Demonstrations

```bash
# Multi-backend Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py

# Multi-backend Element-wise operations benchmark
python benchmarks/benchmark_elementwise.py

# Memory introspection benchmark
python benchmarks/benchmark_memory.py

# End-to-end training demo
python examples/training_demo.py
```

---

## Project Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime
│   ├── CMakeLists.txt               # CMake configuration
│   ├── include/tensorforge/         # Public C++ headers (dtype, shape, allocator, storage, tensor, kernels)
│   ├── src/                         # Native C++ implementations & pybind11 bindings
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (0.6.0)
│   ├── backend/                     # NEW: Backend Dispatcher Subsystem
│   │   ├── __init__.py              # Backend exports
│   │   ├── dispatcher.py            # Global & scoped backend management
│   │   ├── numpy_backend.py         # Reference NumPy operations
│   │   └── native_backend.py        # Native C++ kernel interop & checks
│   ├── native/                      # Python native subpackage
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── data/                        # Data loading subsystem (Dataset, DataLoader)
│   ├── training/                    # Training loop & metrics (Trainer, accuracy)
│   ├── nn/                          # Neural Network subsystem (Parameter, Module, Linear, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem (Tensor, Storage, NumPyStorage, NativeStorage)
│   └── utils/
│       ├── validation.py
│       └── profiling.py             # Profiling context manager with backend reporting
│
├── benchmarks/                      # Benchmark Suite
│   ├── README.md
│   ├── benchmark_matmul.py          # Multi-backend matmul benchmark
│   ├── benchmark_elementwise.py     # Multi-backend elementwise benchmark
│   └── benchmark_memory.py          # Memory overhead & allocation benchmark
│
├── tests/                           # Python Test Suite
│   ├── backend/                     # NEW: Backend & Native Dispatch tests
│   │   ├── test_dispatcher.py
│   │   └── test_native_ops.py
│   ├── autograd/
│   ├── nn/
│   ├── optim/
│   ├── tensor/
│   └── training/
│
├── examples/                        # Demonstrations
│   ├── basic_tensor.py
│   ├── autograd_demo.py
│   ├── neural_network_demo.py
│   └── training_demo.py
│
├── README.md
├── pyproject.toml
└── .gitignore
```

---

## Development Status & Roadmap

| Milestone | Status | Description |
|---|---|---|
| **v0.1 – Project Foundation & Tensor Core** | **Complete** | Tensor abstraction, metadata/storage decoupling, basic ops, broadcasting, dtype handling |
| **v0.2 – Automatic Differentiation** | **Complete** | Reverse-mode autodiff DAG engine, topological backpropagation, broadcast reductions |
| **v0.3 – Neural Network Modules & Layers** | **Complete** | Parameter, Module, Linear, Activations (ReLU, Sigmoid, Tanh, Softmax), Losses, Sequential |
| **v0.4 – Optimizers & Training Pipeline** | **Complete** | SGD, Adam, Dataset, DataLoader, Trainer, Metrics, Training History |
| **v0.5 – Native Runtime & Performance Foundation** | **Complete** | C++17 runtime, CPU allocator, native storage, CPU kernels, benchmark suite |
| **v0.6 – Native Operation Dispatch & Runtime Integration** | **Current** | Backend dispatcher, runtime backend switching, automatic NumPy fallback, autograd integration |
| **v0.7 – Advanced Memory Allocators & Inference Runtime** | Planned | Arena allocators, memory pooling, SIMD/AVX vectorization, graph execution engine |
| **v0.8 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
