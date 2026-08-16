# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.3 – Production Runtime Reliability & Concurrency`

> **Development Status:** `v1.3 (Production Release)`
> TensorForge v1.3 delivers complete multi-threaded concurrency safety and production reliability for the inference runtime. Multiple Python threads can now execute `predict()` simultaneously against a single `InferenceRuntime` instance with zero data races, deterministic numerical outputs, per-prediction workspace memory isolation (`ExecutionContext`, `ExecutionContextPool`), RAII lifecycle management (`close()`, context-manager support), and lightweight operational diagnostics (`health()`, `stats()`).

---

## End-to-End System Architecture

```
Tensor Core (NumPyStorage, NativeStorage, Strides, Broadcasting)
  │
  ▼
Automatic Differentiation Engine (Reverse-Mode DAG, Topological Backprop)
  │
  ▼
Neural Network Subsystem (Parameter, Module, Linear, Activations, Losses)
  │
  ▼
Training & Optimization Pipeline (SGD, Adam, Dataset, DataLoader, Trainer)
  │
  ▼
Model Serialization & Checkpointing (Safe .tfmodel / .tfckpt Containers)
  │
  ▼
Inference Graph Representation (InferenceGraph, InferenceNode)
  │
  ▼
Operator Fusion Pass (FusedLinear: ReLU, Sigmoid, Tanh, Softmax)
  │
  ▼
Inference Compiler (InferenceCompiler, CompiledPlanCache)
  ├── Static Shape Propagation (ShapePropagator)
  ├── Interval Memory Planner & Region Reuse (MemoryPlanner, MemoryPlan)
  ├── Parallel Workload Evaluator & Thread Allocation (ExecutionStep)
  └── Thread-Safe Plan Cache
  │
  ▼
Execution Plan IR (Immutable ExecutionPlan, MemoryRegions, Parallel Flags)
  │
  ▼
Thread-Safe Inference Runtime (InferenceRuntime, ContextPool, ThreadPool)
  ├── Per-Prediction ExecutionContext & Workspace Isolation (ExecutionContext)
  ├── Multi-Threaded Concurrent Serving (ThreadPoolExecutor-safe)
  ├── Dynamic Thread Configuration (set_num_threads)
  ├── Deterministic Lifecycle Management (close, context manager)
  └── Operational Health & Diagnostics (health, stats)
```

---

## Key Features & Subsystems

### 1. Per-Prediction Execution Context & Workspace Pool (`tensorforge/inference/context.py`)
- **`ExecutionContext`:** Provides an isolated, per-prediction execution workspace and buffer slot manager. Intermediate activations are never shared between concurrent `predict()` calls.
- **`ExecutionContextPool`:** Thread-safe object pool managing reusable contexts using a fast, synchronized LIFO queue, avoiding memory allocation churn during high-throughput multi-threaded serving.

### 2. Thread-Safe Concurrent Runtime (`tensorforge/inference/runtime.py`)
- **Multi-Threaded Serving:** Multiple threads can invoke `runtime.predict()` simultaneously without data races or corruption.
- **Immutable Weights:** Model parameters are strictly read-only and never duplicated or mutated during inference.
- **Lock-Free Prediction Path:** Predictions run concurrently without global serialization locks.

### 3. Lifecycle & Resource Management
- **Context Manager Support:**
  ```python
  with InferenceRuntime.load("model.tfmodel") as runtime:
      predictions = runtime.predict(x)
  # Automatically closed on exit
  ```
- **Explicit Lifecycle:** `runtime.close()`, `runtime.is_closed`, and `runtime.closed`.
- **Fail-Safe Exceptions:** Attempting to predict on a closed runtime raises `RuntimeClosedError`.

### 4. Operational Health & Diagnostics
- **`runtime.health()`:** Returns status (`"healthy"` / `"closed"`), active and pooled contexts, prediction count, error count, and thread configurations.
- **`runtime.stats()`:** Returns extended memory, parameter, and execution metrics.

---

## Inference Runtime Usage Examples

### 1. Concurrent Multi-Threaded Serving

```python
from concurrent.futures import ThreadPoolExecutor
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load runtime with context manager
with InferenceRuntime.load("classifier.tfmodel") as runtime:
    runtime.set_num_threads(4)
    runtime.compile(input_shape=(16, 32))

    # 2. Dispatch concurrent predictions from multiple worker threads
    batches = [tf.randn((16, 32)) for _ in range(20)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(runtime.predict, batches))

    # 3. Inspect runtime health and metrics
    print("Runtime Health:", runtime.health())
```

### 2. Inspecting Memory and Concurrency Diagnostics

```python
stats = runtime.stats()
print(f"Total Predictions: {stats['prediction_count']}")
print(f"Active Contexts:   {stats['active_contexts']}")
print(f"Pooled Contexts:   {stats['pooled_contexts']}")
print(f"Workspace Memory:  {stats['workspace_bytes']} bytes")
```

---

## Running Benchmarks & Demonstrations

```bash
# Concurrent multi-threaded inference demonstration
python examples/concurrent_inference_demo.py

# Multi-worker concurrency and throughput benchmark
python benchmarks/concurrent_inference.py

# Multi-thread scaling benchmark (1, 2, 4, 8 native threads)
python benchmarks/benchmark_inference.py

# Compiled inference demonstration
python examples/compiled_inference_demo.py

# Operator fusion demonstration
python examples/inference_demo.py

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
│   │   ├── arena.hpp                # 64-byte aligned workspace arena
│   │   ├── thread_pool.hpp          # C++17 inference ThreadPool
│   │   ├── dtype.hpp                # DType enum & traits
│   │   ├── kernels.hpp              # Parallel CPU compute kernels
│   │   ├── shape.hpp                # Shape & contiguous stride math
│   │   ├── storage.hpp              # Native Storage buffer
│   │   └── tensor.hpp               # Native Tensor abstraction
│   ├── src/                         # Native implementations & pybind11 bindings
│   │   ├── allocator.cpp
│   │   ├── arena.cpp                # WorkspaceArena implementation
│   │   ├── thread_pool.cpp          # ThreadPool implementation & synchronization
│   │   ├── bindings.cpp             # pybind11 module bindings
│   │   ├── dtype.cpp
│   │   ├── kernels.cpp              # Multi-threaded parallel compute kernels
│   │   ├── shape.cpp
│   │   ├── storage.cpp
│   │   └── tensor.cpp
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.3.0)
│   ├── inference/                   # Production Inference, Compiler & Concurrency Subsystem
│   │   ├── __init__.py              # Public inference exports
│   │   ├── context.py               # NEW: ExecutionContext & ExecutionContextPool
│   │   ├── compiler.py              # InferenceCompiler & thread-safe CompiledPlanCache
│   │   ├── plan.py                  # Immutable ExecutionPlan & ExecutionStep IR
│   │   ├── shapes.py                # Static ShapePropagator engine
│   │   ├── memory.py                # Interval-based MemoryPlanner & MemoryPlan
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime (thread-safe predict, lifecycle, health)
│   │   └── loader.py                # ModelLoader & architecture reconstitution
│   ├── serialization/               # Model Serialization Subsystem (.tfmodel, .tfckpt)
│   ├── quantization/                # Quantization Subsystem (INT8, Calibration, qmatmul)
│   ├── backend/                     # Multi-backend Dispatcher & Thread Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── nn/                          # Neural Network subsystem (Linear, Sequential, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/                       # Validation & Exception hierarchy
│
├── tests/                           # Python Test Suite
│   ├── inference/                   # Inference, Concurrency & Lifecycle test suite
│   │   ├── test_concurrency.py      # NEW: Concurrent multi-threaded prediction tests
│   │   ├── test_runtime_lifecycle.py# NEW: Lifecycle, close(), health(), and stats() tests
│   │   ├── test_memory_lifetime.py
│   │   ├── test_thread_pool.py
│   │   ├── test_parallel_kernels.py
│   │   ├── test_parallel_runtime.py
│   │   ├── test_compiler.py
│   │   ├── test_shapes.py
│   │   ├── test_memory_planner.py
│   │   ├── test_compiled_runtime.py
│   │   ├── test_fusion.py
│   │   ├── test_fused_correctness.py
│   │   ├── test_fused_backend_dispatch.py
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
│   ├── concurrent_inference_demo.py # NEW: Multi-worker concurrent inference demo
│   ├── parallel_inference_demo.py
│   ├── compiled_inference_demo.py
│   ├── inference_demo.py
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── concurrent_inference.py      # NEW: Multi-worker concurrency benchmark
│   ├── benchmark_inference.py
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
| **v1.1 – Inference Compiler & Execution Planning** | **Complete** | InferenceCompiler, ExecutionPlan IR, static shape propagation, memory planner, native arena, plan caching |
| **v1.2 – Runtime Memory Optimization & Parallel CPU Execution** | **Complete** | Interval memory planner, MemoryRegions, native ThreadPool, parallel CPU kernels, thread scaling |
| **v1.3 – Production Runtime Reliability & Concurrency** | **Complete** | Thread-safe InferenceRuntime, ExecutionContext pool, workspace isolation, lifecycle APIs, diagnostics |
