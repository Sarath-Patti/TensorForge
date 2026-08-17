# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.4 – Runtime Observability & Performance Diagnostics`

> **Development Status:** `v1.4 (Production Release)`
> TensorForge v1.4 introduces a production-grade inference profiler and observability subsystem. It provides high-resolution monotonic execution timing, per-operator performance breakdown, backend execution analytics (Native vs NumPy), latency distribution percentiles (min, max, mean, p50, p95, p99), throughput telemetry, compiler cache efficiency tracking, and workspace memory telemetry. Profiling is **disabled by default** with near-zero overhead on the hot prediction path.

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
Thread-Safe Inference Runtime & Telemetry (InferenceRuntime, ContextPool, ThreadPool)
  ├── Per-Prediction ExecutionContext & Workspace Isolation (ExecutionContext)
  ├── Multi-Threaded Concurrent Serving (ThreadPoolExecutor-safe)
  ├── Dynamic Thread Configuration (set_num_threads)
  ├── Deterministic Lifecycle Management (close, context manager)
  ├── Operational Health & Diagnostics (health, stats)
  └── High-Resolution Telemetry & Profiler (RuntimeProfiler, ProfileEvent, ProfileSession, PerformanceReport)
```

---

## Key Features & Subsystems

### 1. Runtime Profiler & Performance Telemetry (`tensorforge/inference/profiler.py`)
- **`RuntimeProfiler`:** Thread-safe manager handling monotonic nanosecond timing, bounded latency history ring buffers, operator statistics, backend operation distribution, and compiler cache hit/miss tracking.
- **`ProfileEvent`:** Lightweight event descriptor recording operator name, op_type, backend, execution mode, timestamps, duration, shapes, dtype, batch size, estimated FLOPs, workspace bytes, thread count, and context ID.
- **`ProfileSession`:** Context manager for scoped profiling sessions with automatic state restoration.
- **`PerformanceReport`:** Rich diagnostic report providing formatted text summaries, operation breakdowns, backend distributions, latency percentiles, memory telemetry, and compiler cache analytics.

### 2. Low-Overhead Observability Modes
- **Disabled by Default:** Zero event allocations and zero hot-path synchronization when profiling is disabled.
- **Summary Mode (`runtime.enable_profiling(detailed=False)`):** Collects aggregate latency statistics and backend counters with minimal overhead.
- **Detailed Mode (`runtime.enable_profiling(detailed=True)`):** Captures fine-grained per-step execution events across eager, fused, and compiled kernels.

### 3. Per-Prediction Execution Context & Workspace Pool (`tensorforge/inference/context.py`)
- **`ExecutionContext`:** Provides an isolated, per-prediction execution workspace and buffer slot manager. Intermediate activations are never shared between concurrent `predict()` calls.
- **`ExecutionContextPool`:** Thread-safe object pool managing reusable contexts using a fast, synchronized LIFO queue, avoiding memory allocation churn during high-throughput multi-threaded serving.

### 4. Lifecycle & Resource Management
- **Context Manager Support:**
  ```python
  with InferenceRuntime.load("model.tfmodel") as runtime:
      predictions = runtime.predict(x)
  # Automatically closed on exit
  ```
- **Explicit Lifecycle:** `runtime.close()`, `runtime.is_closed`, and `runtime.closed`.
- **Fail-Safe Exceptions:** Attempting to predict on a closed runtime raises `RuntimeClosedError`.

### 5. Operational Health & Diagnostics
- **`runtime.health()`:** Returns status (`"healthy"` / `"closed"`), active and pooled contexts, prediction count, error count, mean latency, p95 latency, and throughput.
- **`runtime.stats()`:** Returns extended memory, parameter, execution, compiler, and latency distribution metrics.

---

## Inference Runtime Usage Examples

### 1. Enabling Profiling & Printing Performance Report

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load and compile model
runtime = InferenceRuntime.load("classifier.tfmodel").compile(input_shape=(16, 32))

# 2. Enable detailed profiling
runtime.enable_profiling(detailed=True)

# 3. Execute inference requests
for _ in range(50):
    output = runtime.predict(tf.randn((16, 32)))

# 4. Generate and print performance report
report = runtime.profile()
print(report.summary())
```

### 2. Scoped Profiling Sessions

```python
with runtime.profile_session(detailed=True) as session:
    output = runtime.predict(x)

print(f"Session latency: {session.duration_ms:.3f} ms")
print(session.summary())
```

### 3. Concurrent Multi-Threaded Serving with Observability

```python
from concurrent.futures import ThreadPoolExecutor
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load runtime with context manager
with InferenceRuntime.load("classifier.tfmodel") as runtime:
    runtime.set_num_threads(4)
    runtime.compile(input_shape=(16, 32))
    runtime.enable_profiling(detailed=False)

    # 2. Dispatch concurrent predictions from multiple worker threads
    batches = [tf.randn((16, 32)) for _ in range(20)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(runtime.predict, batches))

    # 3. Inspect concurrent latency statistics
    stats = runtime.latency_stats()
    print(f"Mean Latency: {stats['mean_ms']:.4f} ms, P95: {stats['p95_ms']:.4f} ms")
    print(f"Throughput:   {stats['throughput_samples_per_sec']:.1f} samples/sec")
```

---

## Package Structure

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
│   ├── __init__.py                  # Top-level exports & version (1.4.0)
│   ├── inference/                   # Production Inference, Compiler, Concurrency & Profiler
│   │   ├── __init__.py              # Public inference exports
│   │   ├── profiler.py              # NEW: ProfileEvent, ProfileSession, RuntimeProfiler, PerformanceReport
│   │   ├── context.py               # ExecutionContext & ExecutionContextPool
│   │   ├── compiler.py              # InferenceCompiler & thread-safe CompiledPlanCache
│   │   ├── plan.py                  # Immutable ExecutionPlan & ExecutionStep IR
│   │   ├── shapes.py                # Static ShapePropagator engine
│   │   ├── memory.py                # Interval-based MemoryPlanner & MemoryPlan
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime (thread-safe predict, lifecycle, profiler)
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
│   ├── inference/                   # Inference, Concurrency & Observability test suite
│   │   ├── test_profiler.py         # NEW: ProfileEvent, RuntimeProfiler modes & operations
│   │   ├── test_profile_session.py  # NEW: Scoped ProfileSession & state restoration
│   │   ├── test_latency_stats.py    # NEW: Latency distribution percentiles & throughput
│   │   ├── test_backend_stats.py    # NEW: Backend execution counters & fallbacks
│   │   ├── test_memory_stats.py     # NEW: Workspace memory & region telemetry
│   │   ├── test_compiler_stats.py   # NEW: Compiler cache hits, misses & times
│   │   ├── test_concurrent_profiling.py # NEW: Thread safety under concurrent predict()
│   │   ├── test_profiling_overhead.py   # NEW: Zero overhead disabled mode verification
│   │   ├── test_concurrency.py
│   │   ├── test_runtime_lifecycle.py
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
│   ├── profiling_demo.py            # NEW: Observability, sessions & performance reports
│   ├── concurrent_inference_demo.py
│   ├── parallel_inference_demo.py
│   ├── compiled_inference_demo.py
│   ├── inference_demo.py
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── profiling_overhead.py        # NEW: Profiling overhead across batch sizes
│   ├── concurrent_inference.py
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
| **v1.4 – Runtime Observability & Performance Diagnostics** | **Complete** | Profiler subsystem, ProfileEvent, ProfileSession, PerformanceReport, latency percentiles, low overhead |
