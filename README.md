# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.7 – Quantization Runtime & INT8 Inference`

> **Development Status:** `v0.7 (Active Milestone)`
> TensorForge v0.7 introduces a complete **Quantization Subsystem** (`tensorforge.quantization`) for low-precision INT8 post-training inference. It features contiguous physical INT8 storage via `QuantizedTensor` (providing 4x memory compression), symmetric and asymmetric affine quantization routines, representative dataset calibration, INT8 matrix multiplication with 32-bit overflow prevention, native C++ kernel acceleration, and signal-to-quantization-noise error metrics.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, and model training without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.7**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation DAG engine (`autograd`).
- Neural network layers & activations (`Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, `CrossEntropyLoss`, `Sequential`).
- Optimization & Training pipeline (`SGD`, `Adam`, `TensorDataset`, `DataLoader`, `Trainer`, `accuracy`).
- **C++17 Native Runtime Subsystem (`native/`):**
  - Aligned 64-byte `DefaultCPUAllocator` with active memory tracking.
  - Native C++ `Storage` with RAII memory lifetime ownership.
  - Native C++ `Tensor` and `Shape` representations preserving row-major contiguous layout.
  - Handcrafted CPU compute kernels: FP32 element-wise arithmetic, cache-aware FP32 matmul, and INT8 quantized matmul.
- **Backend Dispatcher Subsystem (`tensorforge/backend/`):**
  - Runtime backend selection (`"numpy"` vs `"native"`).
  - Automatic fallback to NumPy when native kernels are not eligible.
  - Seamless Python autograd backpropagation across both native and NumPy forward computations.
- **Quantization & INT8 Inference Subsystem (`tensorforge/quantization/`):**
  - **`QuantizedTensor`:** Low-precision data structure storing contiguous INT8 physical memory alongside linear scale and zero-point parameters.
  - **Quantization Math:** Symmetric ($z=0$) and Asymmetric Affine quantization algorithms with numerical safety for constant and zero-range tensors.
  - **Calibration Algorithms:** `MinMaxCalibrator`, `MovingAverageCalibrator`, and outlier-resistant `PercentileCalibrator`.
  - **INT8 Compute Kernels:** `qmatmul` matrix multiplication with 32-bit integer accumulation to prevent overflow, accelerated in C++ and NumPy.
  - **Evaluation Metrics:** `max_absolute_error`, `mean_absolute_error`, `mean_squared_error`, `relative_error`, and `quantization_snr` (SQNR in dB).

---

## Quantization Architecture

```
                            FP32 Tensor / Model
                                     │
                                     ▼
                          Calibration Utilities
                     (MinMax / Percentile / MovingAvg)
                                     │
                                     ▼
                            Quantization Math
                   (Symmetric / Asymmetric Affine INT8)
                                     │
                                     ▼
                              QuantizedTensor
                      ├── INT8 Contiguous Storage (1 byte/elem)
                      ├── Scale (float)
                      ├── Zero-Point (int)
                      └── Original Shape & DType Metadata
                                     │
                                     ▼
                        Backend Dispatcher Execution
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
             NumPy Backend                     Native C++ Backend
           (int8 x int8 -> int32             (qmatmul_int8 Cache-Tiled
            NumPy Accumulation)               (i,k,j) int32 Accumulator)
                    │                                 │
                    └────────────────┬────────────────┘
                                     │
                                     ▼
                           Dequantized FP32 Output
                                     │
                                     ▼
                            Quantization Metrics
                     (Max Error, MAE, MSE, Relative, SNR)
```

---

## Quantization Formulations

### 1. Symmetric Quantization (Zero-Point = 0)
- **Scale:**
  $$\text{scale} = \frac{\max(|x_{\min}|, |x_{\max}|)}{127}$$
- **Quantization:**
  $$q = \text{clamp}\left(\left\lfloor \frac{x}{\text{scale}} \right\rceil, -128, 127\right)$$
- **Dequantization:**
  $$\hat{x} = q \times \text{scale}$$

### 2. Asymmetric Affine Quantization
- **Scale:**
  $$\text{scale} = \frac{x_{\max} - x_{\min}}{255}$$
- **Zero-Point:**
  $$\text{zero\_point} = \text{clamp}\left(\left\lfloor -\frac{x_{\min}}{\text{scale}} \right\rceil - 128, -128, 127\right)$$
- **Quantization:**
  $$q = \text{clamp}\left(\left\lfloor \frac{x}{\text{scale}} \right\rceil + \text{zero\_point}, -128, 127\right)$$
- **Dequantization:**
  $$\hat{x} = (q - \text{zero\_point}) \times \text{scale}$$

### 3. INT8 Matrix Multiplication Accumulation
For matrices $A_q \in \mathbb{Z}^{M \times K}$ and $B_q \in \mathbb{Z}^{K \times N}$:
$$C(i, j) = s_A s_B \sum_{k=0}^{K-1} (A_q(i, k) - z_A)(B_q(k, j) - z_B)$$
Accumulation is performed in 32-bit signed integer registers (`int32`) before scaling by $s_A \times s_B$, preventing 8-bit overflow.

---

## Quantization Usage Example

```python
import tensorforge as tf
from tensorforge.quantization import (
    quantize,
    dequantize,
    qmatmul,
    compare_tensors,
    MinMaxCalibrator,
)

# 1. Create FP32 Matrices
A = tf.randn((64, 128), dtype=tf.float32)
B = tf.randn((128, 32), dtype=tf.float32)

# 2. Quantize to INT8 (Symmetric)
A_q = quantize(A, scheme="symmetric")
B_q = quantize(B, scheme="symmetric")

print(f"FP32 Memory: {A.nbytes}B | INT8 Memory: {A_q.nbytes}B (4x compression)")

# 3. Quantized Matrix Multiplication
C_int8 = qmatmul(A_q, B_q)  # or A_q @ B_q

# 4. FP32 Reference & Error Analysis
C_fp32 = A @ B
metrics = compare_tensors(C_fp32, C_int8)
print(f"MAE: {metrics['mean_abs_error']:.6f} | SQNR: {metrics['sqnr_db']:.2f} dB")
```

---

## Building the Native C++ Runtime

### Prerequisites
- CMake >= 3.15
- C++17 compatible compiler (Clang, GCC, or MSVC)
- pybind11 >= 2.10.0

### Standalone C++ Build
```bash
# Configure and build native C++ library, Python bindings, and test suite
cmake -B native/build -S native -DCMAKE_BUILD_TYPE=Release
cmake --build native/build

# Run standalone C++ native verification tests
ctest --test-dir native/build --output-on-failure
```

---

## Running Benchmarks & Demonstrations

```bash
# Quantization & INT8 Inference Benchmark (Memory, Latency, Error)
python benchmarks/benchmark_quantization.py

# End-to-end model quantization and inference demo
python examples/quantization_demo.py

# Multi-backend Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py

# Multi-backend Element-wise operations benchmark
python benchmarks/benchmark_elementwise.py
```

---

## Project Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime
│   ├── CMakeLists.txt               # CMake build configuration
│   ├── include/tensorforge/         # Public C++ headers (dtype, shape, allocator, storage, tensor, kernels)
│   ├── src/                         # Native C++ implementations & pybind11 bindings
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (0.7.0)
│   ├── quantization/                # NEW: Quantization & INT8 Inference Subsystem
│   │   ├── __init__.py              # Public quantization exports
│   │   ├── quantized_tensor.py      # QuantizedTensor data structure
│   │   ├── quantize.py              # quantize, dequantize, qmatmul
│   │   ├── calibration.py           # MinMax, MovingAverage, Percentile calibrators
│   │   └── metrics.py               # MAE, Max Error, MSE, Relative Error, SQNR
│   ├── backend/                     # Backend Dispatcher Subsystem
│   │   ├── __init__.py
│   │   ├── dispatcher.py            # Global & scoped backend management
│   │   ├── numpy_backend.py         # Reference NumPy operations
│   │   └── native_backend.py        # Native C++ kernel interop & qmatmul
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
│   ├── benchmark_quantization.py    # NEW: FP32 vs INT8 memory, error, and latency benchmark
│   ├── benchmark_matmul.py          # Multi-backend matmul benchmark
│   ├── benchmark_elementwise.py     # Multi-backend elementwise benchmark
│   └── benchmark_memory.py          # Memory overhead & allocation benchmark
│
├── tests/                           # Python Test Suite
│   ├── quantization/                # NEW: Quantization tests
│   │   ├── test_quantization.py
│   │   ├── test_calibrator.py
│   │   ├── test_metrics.py
│   │   └── test_qmatmul.py
│   ├── backend/
│   ├── autograd/
│   ├── nn/
│   ├── optim/
│   ├── tensor/
│   └── training/
│
├── examples/                        # Demonstrations
│   ├── quantization_demo.py         # NEW: End-to-end post-training quantization demo
│   ├── training_demo.py
│   ├── neural_network_demo.py
│   ├── autograd_demo.py
│   └── basic_tensor.py
│
├── README.md
├── pyproject.toml
├── setup.py
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
| **v0.6 – Native Operation Dispatch & Runtime Integration** | **Complete** | Backend dispatcher, runtime backend switching, automatic NumPy fallback, autograd integration |
| **v0.7 – Quantization Runtime & INT8 Inference** | **Current** | QuantizedTensor, symmetric & asymmetric INT8 quantization, calibration, INT8 matmul, error metrics |
| **v0.8 – Production Inference Engine & Operator Fusion** | Planned | Graph execution engine, operator fusion (Linear+ReLU), batching queue, C/C++ embedding API |
