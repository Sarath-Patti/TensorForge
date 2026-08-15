# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.8 – Model Serialization & Checkpointing`

> **Development Status:** `v0.8 (Active Milestone)`
> TensorForge v0.8 introduces a robust **Model Serialization & Checkpointing Subsystem** (`tensorforge.serialization`) featuring safe `.tfmodel` and `.tfckpt` container formats. It enables full model state persistence, in-place parameter restoration preserving object identity, complete training checkpointing (model weights, optimizer states, epoch, step, and metrics), training resumption, and low-precision `QuantizedTensor` serialization without relying on `pickle` for model tensors.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, model training, and low-precision inference without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.8**, TensorForge features:
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
- **Model Serialization & Checkpointing Subsystem (`tensorforge/serialization/`):**
  - **`model.state_dict()` / `model.load_state_dict()`:** In-place physical parameter loading preserving `Parameter` object identity, `requires_grad`, and leaf status without creating autograd graph edges.
  - **Safe Container Formats (`.tfmodel`, `.tfckpt`):** Structured ZIP archives containing JSON metadata and raw `.npy` binary arrays (**strictly avoiding `pickle`** for tensor buffers).
  - **Optimizer State Persistence:** Complete serialization of momentum buffers (SGD), first/second raw moment estimates (Adam), and step counters.
  - **Quantized Model Serialization:** Serialization and exact restoration of `QuantizedTensor` parameters (INT8 data, scale, zero_point, scheme, shape, dtype).
  - **Model Size Inspection:** `compute_model_size` utility reporting parameter counts, byte footprints, and compression ratios.

---

## Serialization & Checkpoint Architecture

```
                          Neural Network / Module
                                     │
                     ┌───────────────┴───────────────┐
                     ▼                               ▼
            model.state_dict()            optimizer.state_dict()
       (Named Parameters & Tensors)       (Moments, Buffers, Steps)
                     │                               │
                     └───────────────┬───────────────┘
                                     │
                                     ▼
                      Serialization Engine & Formats
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
        Model Archive (.tfmodel)               Training Checkpoint (.tfckpt)
    ├── metadata.json (Version, Types)     ├── checkpoint.json (Epoch, Step, Loss)
    └── tensors/<param_name>.npy           ├── model_tensors/<param_name>.npy
                                           └── optim_tensors/<param_idx>_<key>.npy
                                     │
                                     ▼
                         Safe Deserialization & Load
                     (Direct in-place np.copyto into
                      Parameter physical storage)
```

---

## Serialization Usage Examples

### 1. Saving and Loading a Model

```python
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.serialization import save_model, load_model

# Construct and save model
model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)
save_model(model, "classifier.tfmodel", metadata={"task": "classification"})

# Instantiate fresh model and restore weights
fresh_model = nn.Sequential(
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 4),
)
load_model(fresh_model, "classifier.tfmodel")
```

### 2. Saving and Resuming Training Checkpoints

```python
import tensorforge as tf
import tensorforge.nn as nn
import tensorforge.optim as optim
from tensorforge.serialization import save_checkpoint, load_checkpoint

model = nn.Linear(8, 2)
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Save checkpoint
save_checkpoint({
    "model": model,
    "optimizer": optimizer,
    "epoch": 5,
    "step": 250,
    "loss": 0.042,
}, "checkpoint.tfckpt")

# Restore in a fresh training script
fresh_model = nn.Linear(8, 2)
fresh_optimizer = optim.Adam(fresh_model.parameters(), lr=0.01)

checkpoint_data = load_checkpoint("checkpoint.tfckpt")
fresh_model.load_state_dict(checkpoint_data["model_state_dict"])
fresh_optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
start_epoch = checkpoint_data["epoch"]
```

---

## Running Benchmarks & Demonstrations

```bash
# Model serialization and checkpointing demonstration
python examples/serialization_demo.py

# End-to-end post-training quantization and inference demo
python examples/quantization_demo.py

# Quantization benchmark
python benchmarks/benchmark_quantization.py

# Multi-backend Matrix multiplication benchmark
python benchmarks/benchmark_matmul.py
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
│   ├── __init__.py                  # Top-level exports & version (0.8.0)
│   ├── serialization/               # NEW: Model Serialization & Checkpointing Subsystem
│   │   ├── __init__.py              # Public serialization exports
│   │   ├── format.py                # Safe structured ZIP container (.tfmodel, .tfckpt)
│   │   └── checkpoint.py            # save_model, load_model, save_checkpoint, load_checkpoint
│   ├── quantization/                # Quantization & INT8 Inference Subsystem
│   │   ├── quantized_tensor.py      # QuantizedTensor data structure
│   │   ├── quantize.py              # quantize, dequantize, qmatmul
│   │   ├── calibration.py           # MinMax, MovingAverage, Percentile calibrators
│   │   └── metrics.py               # MAE, Max Error, MSE, Relative Error, SQNR
│   ├── backend/                     # Backend Dispatcher Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam with state_dict)
│   ├── nn/                          # Neural Network subsystem (Module with state_dict, Linear, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/
│       └── validation.py            # SerializationError, QuantizationError, ShapeError
│
├── tests/                           # Python Test Suite
│   ├── serialization/               # NEW: Serialization test suite
│   │   ├── test_state_dict.py
│   │   ├── test_serialization.py
│   │   ├── test_checkpoint.py
│   │   └── test_quantized_serialization.py
│   ├── quantization/
│   ├── backend/
│   ├── autograd/
│   ├── nn/
│   └── optim/
│
├── examples/                        # Demonstrations
│   ├── serialization_demo.py        # NEW: Model serialization & training resume demo
│   ├── quantization_demo.py
│   └── training_demo.py
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
| **v0.8 – Model Serialization & Checkpointing** | **Current** | Safe .tfmodel & .tfckpt formats, state_dict, optimizer state persistence, training resumption |
| **v0.9 – Production Inference Engine & Operator Fusion** | Planned | Graph execution engine, operator fusion (Linear+ReLU), batching queue, C/C++ embedding API |
