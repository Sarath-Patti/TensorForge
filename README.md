# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.5 – Native Runtime & Performance Foundation`

> **Development Status:** `v0.5 (Active Milestone)`
> TensorForge v0.5 introduces a clean C++17 native runtime layer beneath the Python Tensor and Storage abstractions. It provides an aligned CPU memory allocator, RAII-managed native `Storage` and `Tensor` abstractions, standalone CPU kernels (element-wise add, sub, mul, and cache-friendly matrix multiplication), optional `pybind11` bindings, and a standardized benchmark suite.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, and model training without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.5**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation engine (`autograd`).
- Neural network layers & activations (`Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, `CrossEntropyLoss`, `Sequential`).
- Optimization & Training pipeline (`SGD`, `Adam`, `TensorDataset`, `DataLoader`, `Trainer`, `accuracy`).
- **C++17 Native Runtime Subsystem (`native/`):**
  - Aligned 64-byte `DefaultCPUAllocator` with active memory tracking.
  - Native C++ `Storage` with RAII memory lifetime ownership.
  - Native C++ `Tensor` and `Shape` representations preserving row-major contiguous layout.
  - Handcrafted CPU compute kernels: element-wise arithmetic and cache-aware $(i, k, j)$ matrix multiplication.
  - Optional `pybind11` bridge (`_tensorforge_native`) and `NativeStorage` Python backend.
- **Benchmarking Suite (`benchmarks/`):**
  - Matrix multiplication, element-wise arithmetic, and memory overhead benchmarks.

---

## Native Runtime Architecture

```
                  TensorForge
                      │
               Python Tensor API
                      │
                 Storage API
                  /        \
                 /          \
        NumPyStorage      NativeStorage
      (Default Backend) (Optional Backend)
             │                 │
           NumPy          C++17 Runtime
                               │
                     ┌─────────┴─────────┐
                     │                   │
                  Storage             Kernels
                     │                   │
                 Allocator           CPU Ops
                                         │
                                      Matmul
```

---

## Supported Native & Python Components (v0.5)

| Component | Layer | Description |
|---|---|---|
| **`DefaultCPUAllocator`** | Native C++ | 64-byte aligned memory allocator tracking active allocations |
| **`Storage`** | Native C++ | RAII contiguous memory buffer on CPU |
| **`Tensor`** | Native C++ | Lightweight native tensor with shape and stride metadata |
| **`kernels::matmul`** | Native C++ | Cache-aware $(i, k, j)$ float32 matrix multiplication kernel |
| **`kernels::add/sub/mul`** | Native C++ | Vectorized element-wise float32 operations |
| **`NativeStorage`** | Python | Optional backend storage interfacing with C++ runtime |
| **`NumPyStorage`** | Python | Reference contiguous storage backend (Default) |
| **`profile`** | Python | Lightweight wall-clock profiling context manager |

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
# Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py

# Element-wise operations benchmark
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
├── native/                          # NEW: Native C++17 Runtime
│   ├── CMakeLists.txt               # CMake configuration
│   ├── include/tensorforge/         # Public C++ headers
│   │   ├── dtype.hpp                # DType enum & metadata
│   │   ├── shape.hpp                # Shape & contiguous stride calculation
│   │   ├── allocator.hpp            # Allocator interface & DefaultCPUAllocator
│   │   ├── storage.hpp              # Native Storage RAII buffer
│   │   ├── tensor.hpp               # Native Tensor metadata & buffer link
│   │   └── kernels.hpp              # Element-wise and Matmul kernel declarations
│   ├── src/                         # Native C++ implementations
│   │   ├── dtype.cpp
│   │   ├── shape.cpp
│   │   ├── allocator.cpp
│   │   ├── storage.cpp
│   │   ├── tensor.cpp
│   │   ├── kernels.cpp
│   │   └── bindings.cpp             # pybind11 module bindings
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (0.5.0)
│   ├── native/                      # NEW: Python native subpackage
│   │   └── __init__.py              # Native availability & operations
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── data/                        # Data loading subsystem (Dataset, DataLoader)
│   ├── training/                    # Training loop & metrics (Trainer, accuracy)
│   ├── nn/                          # Neural Network subsystem (Parameter, Module, Linear, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem (Tensor, Storage, NumPyStorage, NativeStorage)
│   └── utils/
│       ├── __init__.py
│       ├── validation.py
│       └── profiling.py             # NEW: Profile context manager
│
├── benchmarks/                      # NEW: Benchmark framework
│   ├── README.md
│   ├── benchmark_matmul.py
│   ├── benchmark_elementwise.py
│   └── benchmark_memory.py
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
| **v0.5 – Native Runtime & Performance Foundation** | **Current** | C++17 runtime, CPU allocator, native storage, CPU kernels, benchmark suite |
| **v0.6 – Advanced Inference Runtime & Allocators** | Planned | Memory arena allocators, SIMD/AVX vectorization, graph execution engine |
| **v0.7 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
| **v0.8 – Production Inference Engine & C API** | Planned | High-throughput serving runtime, batching queue, C/C++ embedding API |
