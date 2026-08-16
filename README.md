# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.0 – Production Inference Runtime & Operator Fusion`

> **Development Status:** `v1.0 (Production Release)`
> TensorForge v1.0 establishes a complete, production-grade **Inference Engine** with graph-level **Operator Fusion** (`InferenceRuntime.optimize()`). By collapsing multi-operation subgraphs into specialized single-pass native C++ compute kernels, v1.0 eliminates intermediate buffer allocations, maximizes CPU cache locality, and delivers significant latency and throughput improvements across both FP32 and INT8 low-precision workloads.

---

## Project Overview

TensorForge provides end-to-end, first-principles implementations of tensor memory representation, automatic differentiation, neural network composition, model training, post-training quantization, serialization, and high-performance inference.

In **v1.0**, TensorForge includes:
- **Core Tensor Abstraction (`tensorforge/tensor/`):** Contiguous physical memory buffers (`Storage`, `NumPyStorage`, `NativeStorage`), dynamic shape/stride utilities, broadcasting, and strong DType semantics.
- **Autograd Engine (`tensorforge/autograd/`):** Custom reverse-mode automatic differentiation DAG engine with topological backpropagation, gradient accumulation, and context control (`no_grad()`, `detach()`, `zero_grad()`).
- **Neural Network Library (`tensorforge/nn/`):** `Parameter`, `Module`, `Linear`, activation functions (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`), loss criteria (`MSELoss`, `CrossEntropyLoss`), and containers (`Sequential`).
- **Training Subsystem (`tensorforge/optim/`, `tensorforge/training/`, `tensorforge/data/`):** First-order optimizers (`SGD`, `Adam`), `TensorDataset`, `DataLoader`, and high-level `Trainer`.
- **C++17 Native Acceleration Subsystem (`native/`):**
  - 64-byte aligned `DefaultCPUAllocator` with active memory tracking.
  - Native C++ `Storage` with RAII memory lifetime ownership.
  - Native C++ `Tensor` and `Shape` representations preserving row-major contiguous layout.
  - Handcrafted CPU compute kernels: FP32 element-wise arithmetic, cache-aware FP32 matmul, and INT8 quantized matmul.
  - **NEW in v1.0: Fused Forward Kernels:** `fused_linear`, `fused_linear_relu`, `fused_linear_sigmoid`, `fused_linear_tanh`, `fused_linear_softmax`, and INT8 `fused_qlinear_relu_int8`.
- **Quantization & INT8 Inference Subsystem (`tensorforge/quantization/`):**
  - **`QuantizedTensor`:** Low-precision data structure storing contiguous INT8 physical memory alongside linear scale and zero-point parameters.
  - **Quantization Math:** Symmetric ($z=0$) and Asymmetric Affine quantization algorithms with numerical safety.
  - **Calibration Algorithms:** `MinMaxCalibrator`, `MovingAverageCalibrator`, and outlier-resistant `PercentileCalibrator`.
  - **INT8 Compute Kernels:** `qmatmul` and `fused_qlinear_relu_int8` with 32-bit integer accumulation to prevent overflow.
- **Model Serialization & Checkpointing Subsystem (`tensorforge/serialization/`):**
  - Safe `.tfmodel` and `.tfckpt` structured ZIP archives (**strictly avoiding `pickle`** for tensor storage).
  - In-place physical parameter loading preserving `Parameter` object identity and `requires_grad`.
  - Optimizer state persistence and training resumption.
- **Production Inference & Operator Fusion Subsystem (`tensorforge/inference/`):**
  - **`InferenceRuntime`:** Standalone prediction engine operating in `eval` mode with strict `no_grad` guarantees (zero backward graph allocation).
  - **`InferenceGraph` & `OperatorFusionPass`:** Intermediate graph representation with pattern-matching operator fusion.
  - **`GraphOptimizer`:** Execution pipeline with multi-backend dispatch and robust fallback guarantees.

---

## Operator Fusion Architecture

```
                 Original Serialized Model (.tfmodel)
                                  │
                                  ▼
                             ModelLoader
                   (Reconstructs Layer Hierarchy)
                                  │
                                  ▼
                           InferenceGraph
                    ┌─────────────────────────┐
                    │ [0] Linear (16 -> 32)   │
                    │ [1] ReLU                │
                    │ [2] Linear (32 -> 4)    │
                    │ [3] Softmax (dim=-1)    │
                    └─────────────────────────┘
                                  │
                                  ▼
                         OperatorFusionPass
                    (Collapses Adjacent Patterns)
                                  │
                                  ▼
                      Optimized InferenceGraph
                    ┌─────────────────────────┐
                    │ [0] FusedLinear(ReLU)   │
                    │ [1] FusedLinear(Softmax)│
                    └─────────────────────────┘
                                  │
                                  ▼
                          InferenceRuntime
                                  │
          ┌───────────────────────┴───────────────────────┐
          ▼                                               ▼
   Native C++ Backend                               NumPy Backend
(fused_linear_relu, etc.)                     (Single-Pass Fused Reference)
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                     Prediction Output Tensor
                (requires_grad=False, grad_fn=None)
```

---

## Supported Fusion Patterns

| Pattern | Source Layers | Fused Operator | Native Kernel |
|---|---|---|---|
| **Linear + ReLU** | `Linear(M, K)` $\to$ `ReLU()` | `FusedLinear(activation='relu')` | `native_fused_linear_relu` |
| **Linear + Sigmoid** | `Linear(M, K)` $\to$ `Sigmoid()` | `FusedLinear(activation='sigmoid')` | `native_fused_linear_sigmoid` |
| **Linear + Tanh** | `Linear(M, K)` $\to$ `Tanh()` | `FusedLinear(activation='tanh')` | `native_fused_linear_tanh` |
| **Linear + Softmax** | `Linear(M, K)` $\to$ `Softmax()` | `FusedLinear(activation='softmax')` | `native_fused_linear_softmax` |
| **Quantized Linear + ReLU** | `QuantizedLinear` $\to$ `ReLU()` | `FusedQuantizedLinear(ReLU)` | `native_fused_qlinear_relu` |

---

## Execution Fallback Hierarchy

TensorForge guarantees deterministic execution with zero unhandled operation errors through an explicit 4-tier fallback hierarchy:

```
1. Fused Native C++     ──(if native extension loaded & operands eligible)──►
2. Fused NumPy Reference──(if native unavailable / input requires fallback)──►
3. Unfused Native C++   ──(if unoptimized & native backend selected)────────►
4. Unfused NumPy        ──(universal base reference)────────────────────────►
```

---

## Inference Runtime Usage Examples

### 1. Basic Model Loading & Graph Optimization

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load exported model artifact
runtime = InferenceRuntime.load("classifier.tfmodel")

# 2. Apply graph-level operator fusion
runtime.optimize()

# 3. Inspect optimization diagnostic summary
summary = runtime.summary()
print(f"Optimized: {summary['is_optimized']}")
print(f"Collapsed: {summary['original_nodes']} -> {summary['optimized_nodes']} nodes")
print(f"Fused Patterns: {summary['fused_patterns']}")

# 4. Execute prediction on sample or batch
x = tf.randn((8, runtime.input_shape[0]))
predictions = runtime.predict(x)

assert predictions.requires_grad is False
assert predictions.grad_fn is None
```

### 2. Multi-Backend Fused Inference

```python
import tensorforge as tf
from tensorforge.backend import backend_context
from tensorforge.inference import InferenceRuntime

runtime = InferenceRuntime.load("classifier.tfmodel").optimize()
x = tf.randn((16, runtime.input_shape[0]))

# Run with NumPy Fused Backend
with backend_context("numpy"):
    out_np = runtime.predict(x)

# Run with Native C++ Fused Fast Path
with backend_context("native"):
    out_native = runtime.predict(x)
```

### 3. INT8 Low-Precision Fused Inference

```python
from tensorforge.inference import InferenceRuntime

# Load quantized model and optimize
runtime_int8 = InferenceRuntime.load("quantized_classifier.tfmodel").optimize()

print(f"Is Quantized: {runtime_int8.is_quantized}")
print(f"Is Optimized: {runtime_int8.is_optimized}")

output = runtime_int8.predict(x)
```

---

## Running Benchmarks & Demonstrations

```bash
# Production inference & operator fusion demonstration
python examples/inference_demo.py

# Inference performance benchmark (NumPy Unfused vs Native Unfused vs Native Fused)
python benchmarks/benchmark_inference.py

# Model serialization demonstration
python examples/serialization_demo.py

# INT8 Quantization demonstration
python examples/quantization_demo.py
```

---

## Project Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime Subsystem
│   ├── CMakeLists.txt               # CMake build configuration
│   ├── include/tensorforge/         # Public C++ headers
│   │   ├── allocator.hpp            # 64-byte aligned CPU allocator
│   │   ├── dtype.hpp                # DType enum & traits
│   │   ├── kernels.hpp              # Unfused & Fused inference kernels
│   │   ├── shape.hpp                # Shape & contiguous stride math
│   │   ├── storage.hpp              # Native Storage buffer
│   │   └── tensor.hpp               # Native Tensor abstraction
│   ├── src/                         # Native implementations & pybind11 bindings
│   │   ├── allocator.cpp
│   │   ├── bindings.cpp             # pybind11 module bindings
│   │   ├── dtype.cpp
│   │   ├── kernels.cpp              # Handcrafted SIMD-friendly compute kernels
│   │   ├── shape.cpp
│   │   ├── storage.cpp
│   │   └── tensor.cpp
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.0.0)
│   ├── inference/                   # Production Inference & Operator Fusion Subsystem
│   │   ├── __init__.py              # Public inference exports
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime engine
│   │   └── loader.py                # ModelLoader & architecture reconstitution
│   ├── serialization/               # Model Serialization Subsystem (.tfmodel, .tfckpt)
│   ├── quantization/                # Quantization Subsystem (INT8, Calibration, qmatmul)
│   ├── backend/                     # Multi-backend Dispatcher Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── nn/                          # Neural Network subsystem (Linear, Sequential, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/                       # Validation & Exception hierarchy
│
├── tests/                           # Python Test Suite
│   ├── inference/                   # Inference & Fusion test suite
│   │   ├── test_fusion.py           # Pattern matching and node collapsing tests
│   │   ├── test_fused_correctness.py# Fused vs unfused mathematical parity tests
│   │   ├── test_fused_backend_dispatch.py # Multi-backend execution tests
│   │   ├── test_loader.py
│   │   ├── test_runtime.py
│   │   ├── test_fp32_inference.py
│   │   ├── test_backend_dispatch.py
│   │   ├── test_inference_no_grad.py
│   │   └── test_quantized_inference.py
│   ├── serialization/
│   ├── quantization/
│   ├── backend/
│   ├── autograd/
│   ├── nn/
│   └── optim/
│
├── examples/                        # Demonstrations
│   ├── inference_demo.py            # End-to-end production inference & fusion demo
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── benchmark_inference.py       # Latency, throughput, and speedup benchmark
│   ├── benchmark_quantization.py
│   └── benchmark_matmul.py
│
├── README.md
├── pyproject.toml
└── setup.py
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
| **v0.7 – Quantization Runtime & INT8 Inference** | **Complete** | QuantizedTensor, symmetric & asymmetric INT8 quantization, calibration, INT8 matmul, error metrics |
| **v0.8 – Model Serialization & Checkpointing** | **Complete** | Safe .tfmodel & .tfckpt formats, state_dict, optimizer state persistence, training resumption |
| **v0.9 – Portable Inference Runtime & Model Export** | **Complete** | Dedicated InferenceRuntime, ModelLoader, zero-code architecture reconstitution, multi-backend dispatch |
| **v1.0 – Production Inference Engine & Operator Fusion** | **Complete** | InferenceGraph, OperatorFusionPass, native fused C++ kernels, multi-backend fallback, production release |
