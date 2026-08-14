# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.3 – Neural Network Modules`

> **Development Status:** `v0.3 (Active Milestone)`
> TensorForge v0.3 introduces an object-oriented neural network architecture (`tensorforge.nn`) with `Module` base class, `Parameter` abstraction, dense `Linear` layers, activation modules (`ReLU`, `Sigmoid`, `Tanh`, `Softmax`), loss functions (`MSELoss`, `CrossEntropyLoss`), and the `Sequential` container, fully integrated with TensorForge's custom reverse-mode automatic differentiation engine.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, and neural network construction without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.3**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation engine (`autograd`).
- Dynamic computational graph construction with iterative topological backpropagation.
- Object-oriented neural network layer (`tensorforge.nn`):
  - `Parameter`: trainable tensor representation with automatic gradient tracking.
  - `Module`: base class with recursive parameter discovery, submodule registration, `train()` / `eval()` state, and `zero_grad()`.
  - `Linear`: dense fully-connected layer ($y = xW^T + b$) with uniform parameter initialization.
  - Activation modules: `ReLU`, `Sigmoid`, `Tanh`, and `Softmax`.
  - Loss functions: `MSELoss` and numerically stable `CrossEntropyLoss`.
  - `Sequential`: in-order composable module container.

---

## Framework Architecture

```
                       ┌─────────────────────────┐
                       │     Tensor (Metadata)   │
                       │  - shape, strides, dtype│
                       └────────────┬────────────┘
                                    │ owns
                       ┌────────────▼────────────┐
                       │ Storage (Contiguous Buf)│
                       │  - NumPyStorage         │
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │     Autograd Engine     │
                       │  - DAG / Topological DFS│
                       │  - Gradient accumulation│
                       └────────────┬────────────┘
                                    │
                       ┌────────────▼────────────┐
                       │      Module Base        │
                       │  - parameters()         │
                       │  - named_parameters()   │
                       │  - zero_grad(), train() │
                       └──────┬───────────┬──────┘
                              │           │
           ┌──────────────────┴───┐   ┌───┴──────────────────┐
           │                      │   │                      │
   ┌───────▼────────┐     ┌───────▼───▼────┐         ┌───────▼────────┐
   │   Parameter    │     │  Linear Layer  │         │   Sequential   │
   │  - requires_grad│    │  - weight, bias│         │  - child mods  │
   └────────────────┘     └────────────────┘         └────────────────┘
                                  │
                          ┌───────▼────────┐
                          │  Activations   │
                          │ - ReLU, Sigmoid│
                          │ - Tanh, Softmax│
                          └───────┬────────┘
                                  │
                          ┌───────▼────────┐
                          │ Loss Functions │
                          │ - MSELoss      │
                          │ - CrossEntropy │
                          └────────────────┘
```

---

## Supported Neural Network Components (v0.3)

| Component | Description | Forward Mathematical Formulation |
|---|---|---|
| **`Parameter`** | Trainable model parameter | Wraps `Tensor` with `requires_grad=True` |
| **`Module`** | Base layer / model abstraction | Manages parameters, submodules, and state |
| **`Linear`** | Dense fully-connected layer | $y = x W^T + b$ |
| **`ReLU`** | Rectified linear unit activation | $y = \max(0, x)$ |
| **`Sigmoid`** | Logistic sigmoid activation | $y = \frac{1}{1 + \exp(-x)}$ |
| **`Tanh`** | Hyperbolic tangent activation | $y = \tanh(x)$ |
| **`Softmax`** | Normalized probability distribution | $S_i = \frac{\exp(x_i - \max(x))}{\sum_j \exp(x_j - \max(x))}$ |
| **`MSELoss`** | Mean squared error loss | $L = \text{mean}((y_{\text{pred}} - y_{\text{true}})^2)$ |
| **`CrossEntropyLoss`** | Multi-class cross-entropy loss | $L = -\frac{1}{N} \sum_{i=1}^N \log P(y_i)$ (stable log-sum-exp) |
| **`Sequential`** | Sequential module pipeline | $y = f_n(\dots f_2(f_1(x)))$ |

---

## Installation

### Prerequisites
- Python 3.11+
- NumPy >= 1.24.0

### Installing from source
```bash
# Clone the repository
git clone https://github.com/Sarath-Patti/TensorForge.git
cd TensorForge

# Install in editable mode
pip install -e .

# Install optional development dependencies (pytest)
pip install -e ".[dev]"
```

---

## Basic Usage

### Building and Training a Neural Network
```python
import tensorforge as tf
from tensorforge.nn import CrossEntropyLoss, Linear, ReLU, Sequential

# 1. Define network architecture
model = Sequential(
    Linear(in_features=4, out_features=8),
    ReLU(),
    Linear(in_features=8, out_features=3),
)

# 2. Input data and ground truth targets
x = tf.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [1.0, 0.5, -0.2, 0.8],
    [-0.5, 1.2, 0.0, -0.3],
], dtype=tf.float32)
targets = [0, 2, 1]

# 3. Forward pass
logits = model(x)

# 4. Compute loss
criterion = CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"Loss: {loss.item():.4f}")

# 5. Backpropagation
loss.backward()

# 6. Inspect gradients on parameters
for name, param in model.named_parameters():
    print(f"{name} grad shape: {param.grad.shape}")

# 7. Reset gradients
model.zero_grad()
```

Run the included demonstration scripts:
```bash
python examples/basic_tensor.py
python examples/autograd_demo.py
python examples/neural_network_demo.py
```

---

## Project Structure

```
TensorForge/
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version
│   │
│   ├── nn/                          # Neural Network subsystem
│   │   ├── __init__.py              # NN module exports
│   │   ├── parameter.py             # Parameter class
│   │   ├── module.py                # Base Module class
│   │   ├── linear.py                # Dense Linear layer
│   │   ├── activations.py           # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── losses.py                # MSELoss, CrossEntropyLoss
│   │   ├── sequential.py            # Sequential container
│   │   └── init.py                  # Parameter initialization utilities
│   │
│   ├── autograd/                    # Automatic Differentiation engine
│   │   ├── __init__.py              # Autograd exports
│   │   ├── engine.py                # Topological sort & backward engine
│   │   └── function.py              # Backward Node graph definitions
│   │
│   ├── tensor/                      # Core Tensor subsystem
│   │   ├── __init__.py              # Tensor exports
│   │   ├── tensor.py                # Core Tensor abstraction
│   │   ├── dtype.py                 # DType registry & type promotions
│   │   ├── shape.py                 # Shape, strides, broadcasting
│   │   ├── storage.py               # Storage abstraction & NumPyStorage
│   │   └── operations.py           # Differentiable tensor operations
│   │
│   └── utils/
│       ├── __init__.py              # Utility exports
│       └── validation.py            # Error hierarchy & validators
│
├── tests/
│   ├── nn/                          # NN unit tests
│   │   ├── test_parameter.py
│   │   ├── test_module.py
│   │   ├── test_linear.py
│   │   ├── test_activations.py
│   │   ├── test_losses.py
│   │   └── test_sequential.py
│   │
│   ├── autograd/                    # Autograd unit tests
│   │   ├── test_utils.py            # Finite-difference gradient checker
│   │   ├── test_basic_autograd.py
│   │   ├── test_broadcast_gradients.py
│   │   ├── test_matmul_gradients.py
│   │   └── test_reduction_gradients.py
│   │
│   └── tensor/
│       └── test_tensor.py           # Tensor core unit tests
│
├── examples/
│   ├── basic_tensor.py              # Tensor core demo
│   ├── autograd_demo.py             # Autograd demo
│   └── neural_network_demo.py       # Neural network demo
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
| **v0.3 – Neural Network Modules & Layers** | **Current** | Parameter, Module, Linear, Activations (ReLU, Sigmoid, Tanh, Softmax), Losses, Sequential |
| **v0.4 – Optimizers & Training Pipeline** | Planned | SGD, Adam, AdamW, learning rate schedulers, dataloaders, training loops |
| **v0.5 – Model Serialization & Checkpointing** | Planned | Memory-mapped weight serialization, state_dict format, format converters |
| **v0.6 – C++ Inference Runtime & Custom Allocators** | Planned | Native C++ tensor engine, arena allocator, SIMD/AVX kernels, zero-copy Pybind11 integration |
| **v0.7 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
| **v0.8 – Production Inference Engine & C API** | Planned | High-throughput serving runtime, batching queue, C/C++ embedding API |
