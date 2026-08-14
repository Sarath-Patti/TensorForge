# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.2 – Automatic Differentiation`

> **Development Status:** `v0.2 (Active Milestone)`
> TensorForge v0.2 introduces a custom reverse-mode automatic differentiation engine with dynamic computational graph (DAG) construction, iterative topological backpropagation, and broadcast gradient reduction.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, and automatic differentiation without relying on external deep learning framework runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.2**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation engine (`autograd`).
- Dynamic computational graph construction during forward operations.
- Iterative topological sort backpropagation (`tensor.backward()`).
- Analytical gradient formulas with automatic broadcast dimension reduction.
- Gradient accumulation across branching graph paths (`y = x * x + x`).

---

## Autograd & Computation Graph Architecture

```
Forward Pass (Graph Construction):
  x (requires_grad=True) ──┐
                           ├──> Mul (MulBackward) ──> h ──> Sum (SumBackward) ──> loss
  w (requires_grad=True) ──┘

Backward Pass (Iterative Topological Traversal):
  grad(loss) = 1.0
      │
      ▼
  SumBackward
      │
      ▼ grad(h)
  MulBackward
      │
      ├───────────────────────┐
      ▼                       ▼
  grad(x) [accumulated]   grad(w) [accumulated]
```

### Key Autograd Capabilities:
- **Leaf & Non-Leaf Tensors:** User-created inputs are marked as `is_leaf=True`. Intermediate operation results are non-leaf tensors carrying a `grad_fn` pointer to their backward graph node.
- **Topological Backpropagation:** Backpropagation traverses the DAG in reverse topological order using an iterative DFS, avoiding Python recursion limits.
- **Gradient Accumulation:** Repeated tensor uses correctly sum incoming gradient contributions without overwriting existing gradients.
- **Broadcast Gradient Reduction:** Gradients flowing back across broadcasted dimensions are automatically reduced to match original operand shapes via `reduce_gradient_to_shape`.
- **Gradient Utilities:** `.zero_grad()` to reset gradients, `.detach()` to disconnect tensors from the computation graph, and `no_grad()` context manager.

---

## Supported Differentiable Operations (v0.2)

| Operation | Forward Syntax | Backward Gradient Rule |
|---|---|---|
| **Addition** | `a + b` | $\frac{\partial z}{\partial a} = 1, \quad \frac{\partial z}{\partial b} = 1$ |
| **Subtraction** | `a - b` | $\frac{\partial z}{\partial a} = 1, \quad \frac{\partial z}{\partial b} = -1$ |
| **Multiplication** | `a * b` | $\frac{\partial z}{\partial a} = b, \quad \frac{\partial z}{\partial b} = a$ |
| **Division** | `a / b` | $\frac{\partial z}{\partial a} = \frac{1}{b}, \quad \frac{\partial z}{\partial b} = -\frac{a}{b^2}$ |
| **Negation** | `-a` | $\frac{\partial z}{\partial a} = -1$ |
| **Matrix Multiplication** | `a @ b` | $\frac{\partial C}{\partial A} = \text{grad} \cdot B^T, \quad \frac{\partial C}{\partial B} = A^T \cdot \text{grad}$ |
| **Sum Reduction** | `a.sum(axis, keepdims)` | Broadcasts upstream gradient across reduced axes |
| **Mean Reduction** | `a.mean(axis, keepdims)` | Scales upstream gradient by $\frac{1}{N}$ and broadcasts |
| **Reshape** | `a.reshape(*shape)` | Reshapes upstream gradient back to input shape |
| **Transpose** | `a.transpose(*axes)` | Applies inverse axis permutation to upstream gradient |

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

### Autograd Example
```python
import tensorforge as tf

# 1. Create leaf tensors requiring gradients
x = tf.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, requires_grad=False)
w = tf.tensor([[0.5, -1.0], [2.0, 1.5]], dtype=tf.float32, requires_grad=True)
b = tf.tensor([0.1, 0.2], dtype=tf.float32, requires_grad=True)

# 2. Forward pass: Build dynamic computation graph
h = x @ w + b
loss = (h * h).mean()

# 3. Backward pass: Reverse-mode automatic differentiation
loss.backward()

# 4. Inspect analytical gradients
print(f"dL/dW (w.grad):\n{w.grad}")
print(f"dL/dB (b.grad):\n{b.grad}")

# 5. Clear gradients
w.zero_grad()
b.zero_grad()
```

Run the included demonstration scripts:
```bash
python examples/basic_tensor.py
python examples/autograd_demo.py
```

---

## Project Structure

```
TensorForge/
├── tensorforge/
│   ├── __init__.py                  # Package exports & version
│   ├── autograd/
│   │   ├── __init__.py              # Autograd package interface
│   │   ├── engine.py                # Topological sort & backward propagation engine
│   │   └── function.py              # Backward graph Node definitions & unbroadcasting
│   │
│   ├── tensor/
│   │   ├── __init__.py              # Tensor module interface
│   │   ├── tensor.py                # Core Tensor abstraction with autograd state
│   │   ├── dtype.py                 # DType registry & type promotion rules
│   │   ├── shape.py                 # Shape, stride computation & broadcasting geometry
│   │   ├── storage.py               # Storage abstraction & NumPyStorage
│   │   └── operations.py           # Forward ops with autograd graph hooks
│   │
│   └── utils/
│       ├── __init__.py              # Utility exports
│       └── validation.py            # Custom exception hierarchy & shape validators
│
├── tests/
│   ├── autograd/
│   │   ├── test_utils.py            # Finite-difference numerical gradient checker
│   │   ├── test_basic_autograd.py   # Core ops, accumulation, detach, no_grad
│   │   ├── test_broadcast_gradients.py # Broadcast gradient reduction tests
│   │   ├── test_matmul_gradients.py # 1D, 2D, and batched matmul gradient tests
│   │   └── test_reduction_gradients.py # Sum, mean, reshape, transpose gradient tests
│   │
│   └── tensor/
│       └── test_tensor.py           # Tensor core unit tests
│
├── examples/
│   ├── basic_tensor.py              # Tensor core demonstration
│   └── autograd_demo.py             # Automatic differentiation demonstration
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
| **v0.2 – Automatic Differentiation** | **Current** | Reverse-mode autodiff DAG engine, topological backpropagation, broadcast reductions |
| **v0.3 – Neural Network Modules & Layers** | Planned | Parameter abstraction, Modules/Containers, Linear, Conv2D, Activations, Loss functions |
| **v0.4 – Optimizers & Training Pipeline** | Planned | SGD, Adam, AdamW, learning rate schedulers, dataloaders, training loops |
| **v0.5 – Model Serialization & Checkpointing** | Planned | Memory-mapped weight serialization, state_dict format, format converters |
| **v0.6 – C++ Inference Runtime & Custom Allocators** | Planned | Native C++ tensor engine, arena allocator, SIMD/AVX kernels, zero-copy Pybind11 integration |
| **v0.7 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
| **v0.8 – Production Inference Engine & C API** | Planned | High-throughput serving runtime, batching queue, C/C++ embedding API |
