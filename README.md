# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.9 – Portable Inference Runtime & Model Export`

> **Development Status:** `v0.9 (Active Milestone)`
> TensorForge v0.9 introduces a dedicated, high-performance **Portable Inference Runtime** (`tensorforge.inference.InferenceRuntime`) capable of loading serialized `.tfmodel` artifacts, automatically reconstructing network graphs, restoring FP32 and INT8 parameters, and executing predictions across NumPy and Native C++ acceleration backends without requiring the training or autograd stack.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, model training, post-training quantization, and dedicated standalone inference execution.

In **v0.9**, TensorForge features:
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
  - **Quantization Math:** Symmetric ($z=0$) and Asymmetric Affine quantization algorithms with numerical safety.
  - **Calibration Algorithms:** `MinMaxCalibrator`, `MovingAverageCalibrator`, and outlier-resistant `PercentileCalibrator`.
  - **INT8 Compute Kernels:** `qmatmul` matrix multiplication with 32-bit integer accumulation to prevent overflow.
- **Model Serialization & Checkpointing Subsystem (`tensorforge/serialization/`):**
  - Safe `.tfmodel` and `.tfckpt` structured ZIP archives (**strictly avoiding `pickle`** for tensor storage).
  - In-place physical parameter loading preserving `Parameter` object identity and `requires_grad`.
  - Optimizer state persistence and training resumption.
- **Portable Inference Runtime Subsystem (`tensorforge/inference/`):**
  - **`InferenceRuntime`:** Standalone prediction engine operating in `eval` mode with strict `no_grad` guarantees (zero backward graph allocation).
  - **`ModelLoader`:** Automated architecture reconstitution from serialized `.tfmodel` metadata.
  - **Multi-Backend Execution:** Seamless execution on either NumPy reference or accelerated Native C++ kernels.
  - **INT8 Low-Precision Inference:** Direct execution of quantized models with 4x memory savings.

---

## Inference Runtime Architecture

```
                            Serialized Artifact (.tfmodel)
                                          │
                                          ▼
                                     ModelLoader
                           (Reconstructs Layer Graph &
                            Loads FP32/INT8 Parameters)
                                          │
                                          ▼
                                  InferenceRuntime
                            ├── Model in eval() Mode
                            ├── no_grad() Execution Context
                            └── Backend Dispatcher Link
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
                 NumPy Backend                      Native C++ Backend
            (Universal CPU Reference)             (qmatmul / matmul C++ Kernels)
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                             Prediction Output Tensor
                         (requires_grad=False, grad_fn=None)
```

---

## Inference Runtime Usage Examples

### 1. Basic FP32 Inference

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load exported model artifact
runtime = InferenceRuntime.load("classifier.tfmodel")

# 2. Inspect runtime summary
print(runtime.summary())

# 3. Execute prediction on sample or batch
x = tf.randn((8, runtime.input_shape[0]))
predictions = runtime.predict(x)

print("Predictions shape:", predictions.shape)
assert predictions.requires_grad is False
```

### 2. Multi-Backend Inference Dispatch

```python
import tensorforge as tf
from tensorforge.backend import backend_context
from tensorforge.inference import InferenceRuntime

runtime = InferenceRuntime.load("classifier.tfmodel")
x = tf.randn((16, runtime.input_shape[0]))

# Run with NumPy Backend
with backend_context("numpy"):
    out_np = runtime.predict(x)

# Run with Native C++ Fast Path (if available)
with backend_context("native"):
    out_native = runtime.predict(x)
```

### 3. INT8 Quantized Model Inference

```python
from tensorforge.inference import InferenceRuntime

# Load quantized model
runtime_int8 = InferenceRuntime.load("quantized_classifier.tfmodel")

print(f"Is Quantized: {runtime_int8.is_quantized}")
output = runtime_int8.predict(x)
```

---

## Running Benchmarks & Demonstrations

```bash
# Inference runtime demonstration
python examples/inference_demo.py

# Inference performance benchmark (Latency & Throughput)
python benchmarks/benchmark_inference.py

# Serialization demonstration
python examples/serialization_demo.py

# Quantization demonstration
python examples/quantization_demo.py
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
│   ├── __init__.py                  # Top-level exports & version (0.9.0)
│   ├── inference/                   # NEW: Portable Inference Runtime Subsystem
│   │   ├── __init__.py              # Public inference exports (InferenceRuntime, ModelLoader)
│   │   ├── runtime.py               # Standalone InferenceRuntime execution engine
│   │   └── loader.py                # ModelLoader & dynamic architecture reconstitution
│   ├── serialization/               # Model Serialization & Checkpointing Subsystem
│   │   ├── format.py                # Safe structured ZIP container (.tfmodel, .tfckpt)
│   │   └── checkpoint.py            # save_model, load_model, save_checkpoint, load_checkpoint
│   ├── quantization/                # Quantization & INT8 Inference Subsystem
│   │   ├── quantized_tensor.py      # QuantizedTensor data structure
│   │   ├── quantize.py              # quantize, dequantize, qmatmul
│   │   ├── calibration.py           # MinMax, MovingAverage, Percentile calibrators
│   │   └── metrics.py               # MAE, Max Error, MSE, Relative Error, SQNR
│   ├── backend/                     # Backend Dispatcher Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam with state_dict)
│   ├── nn/                          # Neural Network subsystem (Module, Linear, Sequential, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/
│       └── validation.py            # Custom exception hierarchy
│
├── tests/                           # Python Test Suite
│   ├── inference/                   # NEW: Inference runtime test suite
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
│   ├── inference_demo.py            # NEW: Standalone inference runtime demo
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Benchmarks
│   ├── benchmark_inference.py       # NEW: Inference latency and throughput benchmark
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
| **v0.9 – Portable Inference Runtime & Model Export** | **Current** | Dedicated InferenceRuntime, ModelLoader, zero-code architecture reconstitution, multi-backend dispatch |
| **v1.0 – Production Inference Engine & Operator Fusion** | Planned | Graph execution engine, operator fusion (Linear+ReLU), batching queue, C/C++ embedding API |
