# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance inference engine built from first principles.

---

## Current Milestone: `v0.4 – Optimizers & Training Engine`

> **Development Status:** `v0.4 (Active Milestone)`
> TensorForge v0.4 introduces an end-to-end training and optimization subsystem: parameter optimizers (`SGD` with momentum and weight decay, `Adam` with bias correction), dataset abstractions (`Dataset`, `TensorDataset`), mini-batch `DataLoader`, evaluation metrics, and the `Trainer` loop orchestrator.

---

## Project Overview

TensorForge provides explicit control over memory representation, tensor operations, automatic differentiation, neural network composition, and model training without relying on external deep learning runtimes (such as PyTorch, TensorFlow, or JAX).

In **v0.4**, TensorForge features:
- Core multi-dimensional `Tensor` abstraction with contiguous physical storage.
- Custom reverse-mode automatic differentiation engine (`autograd`).
- Neural network layers & activations (`Parameter`, `Module`, `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, `MSELoss`, `CrossEntropyLoss`, `Sequential`).
- **Optimization Subsystem (`tensorforge.optim`):**
  - Base `Optimizer` with safe in-place parameter buffer updates preserving `is_leaf` status.
  - `SGD` with momentum velocity accumulation and weight decay.
  - `Adam` with running first/second moment estimation and analytical bias correction.
- **Data Subsystem (`tensorforge.data`):**
  - `Dataset` & `TensorDataset` indexing abstractions.
  - `DataLoader` providing mini-batching, reproducible shuffling, and partial-batch controls.
- **Training Engine (`tensorforge.training`):**
  - `Trainer` supporting standard training loops (`model.train()`, `zero_grad()`, `loss.backward()`, `optimizer.step()`) and isolated evaluation passes (`model.eval()`, `no_grad()`).
  - History logging (`train_loss`, `val_loss`, `train_acc`, `val_acc`).

---

## Training Pipeline Architecture

```
                    ┌─────────────────────────┐
                    │     Dataset / Data      │
                    │ - Dataset Base          │
                    │ - TensorDataset         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │       DataLoader        │
                    │ - batching, shuffling   │
                    │ - drop_last             │
                    └───────────┬─────────────┘
                                │ (x_batch, y_batch)
                                ▼
                    ┌─────────────────────────┐
                    │      Model / Loss       │
                    │ - prediction = model(x) │
                    │ - loss = loss_fn(p, y)  │
                    │ - loss.backward()       │
                    └───────────┬─────────────┘
                                │ param.grad
                                ▼
                    ┌─────────────────────────┐
                    │    Optimizer Engine     │
                    │ - SGD (momentum, wd)    │
                    │ - Adam (beta1, beta2)   │
                    │ - step(), zero_grad()   │
                    └───────────┬─────────────┘
                                │ (Updated parameters)
                                ▼
                    ┌─────────────────────────┐
                    │      Trainer Loop       │
                    │ - fit(train, val, ep)   │
                    │ - evaluate()            │
                    │ - accuracy, loss history│
                    └─────────────────────────┘
```

---

## Supported Optimization & Training Components (v0.4)

| Component | Category | Description |
|---|---|---|
| **`SGD`** | Optimizer | Stochastic Gradient Descent with optional momentum and L2 weight decay |
| **`Adam`** | Optimizer | Adaptive Moment Estimation with bias correction |
| **`Dataset`** | Data | Abstract base class for indexed datasets |
| **`TensorDataset`** | Data | Wraps multi-field tensors indexing samples along dimension 0 |
| **`DataLoader`** | Data | Mini-batch generator with shuffling and `drop_last` options |
| **`Trainer`** | Training | Training and validation loop orchestrator with history logging |
| **`accuracy`** | Metrics | Multi-class classification accuracy calculation |

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

## Complete Training Example

```python
import numpy as np
import tensorforge as tf
from tensorforge.data import DataLoader, TensorDataset
from tensorforge.nn import CrossEntropyLoss, Linear, ReLU, Sequential
from tensorforge.optim import Adam
from tensorforge.training import Trainer

# 1. Generate synthetic dataset
X = np.random.randn(200, 4).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

dataset = TensorDataset(tf.tensor(X), tf.tensor(y))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Define neural network model
model = Sequential(
    Linear(in_features=4, out_features=16),
    ReLU(),
    Linear(in_features=16, out_features=2),
)

# 3. Setup loss function and optimizer
loss_fn = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.01)

# 4. Train with Trainer
trainer = Trainer(model=model, optimizer=optimizer, loss_fn=loss_fn)
history = trainer.fit(train_loader=train_loader, epochs=10)

print(f"Final training loss: {history['train_loss'][-1]:.4f}")
print(f"Final training accuracy: {history['train_acc'][-1] * 100:.2f}%")
```

Run the included demonstration scripts:
```bash
python examples/basic_tensor.py
python examples/autograd_demo.py
python examples/neural_network_demo.py
python examples/training_demo.py
```

---

## Project Structure

```
TensorForge/
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version
│   │
│   ├── optim/                       # Optimizer subsystem
│   │   ├── __init__.py              # Optimizer exports
│   │   ├── optimizer.py             # Optimizer base class
│   │   ├── sgd.py                   # SGD with momentum & weight decay
│   │   └── adam.py                  # Adam optimizer
│   │
│   ├── data/                        # Data loading subsystem
│   │   ├── __init__.py              # Data exports
│   │   ├── dataset.py               # Dataset & TensorDataset
│   │   └── dataloader.py            # Mini-batch DataLoader
│   │
│   ├── training/                    # Training engine
│   │   ├── __init__.py              # Training exports
│   │   ├── trainer.py               # Trainer fit & evaluate loop
│   │   └── metrics.py               # Accuracy & evaluation metrics
│   │
│   ├── nn/                          # Neural Network subsystem
│   │   ├── __init__.py              # NN module exports
│   │   ├── parameter.py             # Parameter class
│   │   ├── module.py                # Base Module class
│   │   ├── linear.py                # Dense Linear layer
│   │   ├── activations.py           # ReLU, Sigmoid, Tanh, Softmax
│   │   ├── losses.py                # MSELoss, CrossEntropyLoss
│   │   ├── sequential.py            # Sequential container
│   │   └── init.py                  # Parameter initialization
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
│   ├── optim/                       # Optimizer unit tests
│   │   ├── test_optimizer.py
│   │   ├── test_sgd.py
│   │   └── test_adam.py
│   │
│   ├── data/                        # Data unit tests
│   │   ├── test_dataset.py
│   │   └── test_dataloader.py
│   │
│   ├── training/                    # Training unit tests
│   │   ├── test_trainer.py
│   │   └── test_metrics.py
│   │
│   ├── nn/                          # NN unit tests
│   ├── autograd/                    # Autograd unit tests
│   └── tensor/                      # Tensor core unit tests
│
├── examples/
│   ├── basic_tensor.py              # Tensor core demo
│   ├── autograd_demo.py             # Autograd demo
│   ├── neural_network_demo.py       # Neural network demo
│   └── training_demo.py             # End-to-end training demo
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
| **v0.4 – Optimizers & Training Pipeline** | **Current** | SGD, Adam, Dataset, DataLoader, Trainer, Metrics, Training History |
| **v0.5 – Model Serialization & Checkpointing** | Planned | Memory-mapped weight serialization, state_dict format, format converters |
| **v0.6 – C++ Inference Runtime & Custom Allocators** | Planned | Native C++ tensor engine, arena allocator, SIMD/AVX kernels, zero-copy Pybind11 integration |
| **v0.7 – Quantization & Graph Optimizations** | Planned | INT8/FP16 post-training quantization, operator fusion, constant folding |
| **v0.8 – Production Inference Engine & C API** | Planned | High-throughput serving runtime, batching queue, C/C++ embedding API |
