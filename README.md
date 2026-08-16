# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.2 – Runtime Memory Optimization & Parallel CPU Execution`

> **Development Status:** `v1.2 (Production Release)`
> TensorForge v1.2 enhances the production inference runtime with general **Interval-Based Memory Lifetime Planning** and **Multi-Threaded Parallel CPU Execution** (`ThreadPool`, `MemoryPlan`, `MemoryRegion`). By analyzing intermediate buffer liveness intervals, the runtime packs transient activations into 64-byte aligned reusable memory regions. Concurrently, native C++ kernels leverage row-partitioned multi-threading across independent output slices with zero data races and dynamic thread configuration.

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
  └── Plan Cache
  │
  ▼
Execution Plan IR (ExecutionPlan, MemoryRegions, Parallel Flags)
  │
  ▼
Native C++ Parallel Runtime & Memory Arena (WorkspaceArena, ThreadPool, CPU Kernels)
```

---

## Key Features & Subsystems

### 1. Interval Memory Lifetime Analysis (`tensorforge/inference/memory.py`)
- **`BufferLifetime`:** Tracks creation (`first_use`) and final read (`last_use`) intervals for all intermediate activations.
- **`MemoryRegion`:** Allocates contiguous, reusable physical memory regions packed with 64-byte SIMD alignment padding.
- **`MemoryPlanner`:** Uses interval coloring to schedule buffers into non-overlapping regions, minimizing peak workspace consumption across general topologies.

### 2. Multi-Threaded Parallel CPU Subsystem (`native/include/tensorforge/thread_pool.hpp`, `native/src/thread_pool.cpp`)
- **`ThreadPool`:** Lightweight, header-backed C++17 thread pool creating reusable worker threads across predictions with clean RAII shutdown.
- **Row-Level Output Partitioning:** Distributes $M$ output rows across worker threads for `matmul`, `fused_linear`, and activation fusions, ensuring each thread writes exclusively to independent memory addresses.
- **Small Workload Fallback:** Workloads below `PARALLEL_WORKLOAD_THRESHOLD` ($< 8192$ operations or batch size $\le 1$) execute synchronously on the calling thread to eliminate synchronization overhead.

### 3. Thread Configuration APIs
- Global configuration: `tensorforge.backend.set_num_threads(n)` / `tensorforge.backend.get_num_threads()`.
- Runtime-level configuration: `runtime.set_num_threads(n)` / `runtime.num_threads`.

### 4. Thread-Safety Guarantees & Constraints
- **Parameters:** Weight parameters are strictly immutable during inference, preventing race conditions on model weights.
- **Instance Concurrency:** A single `InferenceRuntime` instance uses internal stateful planning descriptors and workspace memory slots. Concurrent predictions on the exact same `InferenceRuntime` instance should use separate runtime instances or be externally synchronized.

---

## Inference Runtime Usage Examples

### 1. Compiling and Running Multi-Threaded Inference

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load exported model artifact (.tfmodel)
runtime = InferenceRuntime.load("classifier.tfmodel")

# 2. Configure 4 CPU worker threads and compile for batch size 32
runtime.set_num_threads(4)
runtime.compile(input_shape=(32, 16))

# 3. Inspect runtime and memory plan
summary = runtime.summary()
print(f"Active Backend:      {summary['backend']}")
print(f"Configured Threads:  {summary['num_threads']}")
print(f"Workspace Memory:    {summary['workspace_bytes']} bytes")
print(f"Memory Regions:      {summary['workspace_regions']}")
print(f"Reused Buffers:      {summary['reused_buffers']}")

# 4. Predict with multi-threaded parallel execution
x = tf.randn((32, 16))
predictions = runtime.predict(x)

assert predictions.requires_grad is False
assert predictions.grad_fn is None
```

### 2. Inspecting Reusable Memory Regions

```python
mem_plan = runtime.memory_plan
for r_id, region in mem_plan.regions.items():
    print(f"Region {r_id}: Capacity={region.size_bytes}B, Offset={region.offset_bytes}B, Buffers={region.assigned_buffers}")
```

---

## Running Benchmarks & Demonstrations

```bash
# Parallel multi-threaded inference demonstration
python examples/parallel_inference_demo.py

# Multi-thread scaling performance benchmark (1, 2, 4, 8 threads)
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
│   │   ├── thread_pool.hpp          # NEW: C++17 inference ThreadPool
│   │   ├── dtype.hpp                # DType enum & traits
│   │   ├── kernels.hpp              # Parallel CPU compute kernels
│   │   ├── shape.hpp                # Shape & contiguous stride math
│   │   ├── storage.hpp              # Native Storage buffer
│   │   └── tensor.hpp               # Native Tensor abstraction
│   ├── src/                         # Native implementations & pybind11 bindings
│   │   ├── allocator.cpp
│   │   ├── arena.cpp                # WorkspaceArena implementation
│   │   ├── thread_pool.cpp          # NEW: ThreadPool implementation
│   │   ├── bindings.cpp             # pybind11 module bindings (thread controls)
│   │   ├── dtype.cpp
│   │   ├── kernels.cpp              # Multi-threaded parallel compute kernels
│   │   ├── shape.cpp
│   │   ├── storage.cpp
│   │   └── tensor.cpp
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.2.0)
│   ├── inference/                   # Production Inference, Compiler & Parallel Subsystem
│   │   ├── __init__.py              # Public inference exports
│   │   ├── compiler.py              # InferenceCompiler & CompiledPlanCache
│   │   ├── plan.py                  # ExecutionPlan & ExecutionStep IR (parallel flags)
│   │   ├── shapes.py                # Static ShapePropagator engine
│   │   ├── memory.py                # Interval-based MemoryPlanner & MemoryPlan
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime (thread controls & memory plan)
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
│   ├── inference/                   # Inference & Parallel test suite
│   │   ├── test_memory_lifetime.py  # NEW: Interval liveness & memory region tests
│   │   ├── test_thread_pool.py      # NEW: CPU thread pool configuration tests
│   │   ├── test_parallel_kernels.py # NEW: Parallel kernel numerical parity tests
│   │   ├── test_parallel_runtime.py # NEW: Multi-threaded runtime & immutability tests
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
│   ├── parallel_inference_demo.py   # NEW: Parallel CPU execution & memory plan demo
│   ├── compiled_inference_demo.py
│   ├── inference_demo.py
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── benchmark_inference.py       # Thread scaling & backend comparative benchmark
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
