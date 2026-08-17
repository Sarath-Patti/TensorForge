# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.5 – Production Reliability, Resource Management & Runtime Safety`

> **Development Status:** `v1.5 (Production Release)`
> TensorForge v1.5 hardens the inference runtime for mission-critical, long-running production environments. It introduces formal lifecycle management (`CREATED`, `READY`, `CLOSED`), configurable admission control and resource constraints (`RuntimeLimits`), strict input validation (rank, feature dimension, finite value checks), workspace memory protection, concurrent request limiting (`RuntimeBusyError`), soft timeout enforcement (`RuntimeTimeoutError`), request-level isolation with unique request IDs, clean failure recovery, and extended production diagnostics in `health()` and `stats()`.

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
Hardened Production Inference Runtime (InferenceRuntime, Limits, ContextPool, ThreadPool)
  ├── Admission Control & Resource Limits (RuntimeLimits: batch, elements, workspace, timeout, concurrency)
  ├── Request Isolation & Unique Request IDs (ExecutionContext.request_id)
  ├── Input Validation & Finite Value Protection (TensorForgeInputError)
  ├── Graceful Lifecycle & Idempotent Shutdown (RuntimeState: CREATED, READY, CLOSED)
  ├── Operational Health & Diagnostics (health, stats)
  └── High-Resolution Telemetry & Profiler (RuntimeProfiler, ProfileEvent, PerformanceReport)
```

---

## Key Features & Subsystems

### 1. Admission Control & Resource Limits (`tensorforge/inference/limits.py`)
- **`RuntimeLimits`:** Configurable resource limits protecting the runtime against out-of-memory errors and overloaded compute:
  - `max_batch_size`: Rejects batch sizes exceeding the threshold before allocation.
  - `max_input_elements`: Limits total input tensor element count.
  - `max_workspace_bytes`: Guards against excessive execution plan workspace memory demands.
  - `max_prediction_time_ms`: Soft timeout monitoring for runaway predictions.
  - `max_concurrent_requests`: Rejects incoming requests with `RuntimeBusyError` when capacity is full.

### 2. Runtime Lifecycle & Graceful Shutdown
- **Lifecycle States:** Clear state progression (`CREATED` -> `READY` -> `CLOSED`).
- **Idempotent `close()`:** Releasing pooled execution contexts and native workspace arenas cleanly without double-free errors.
- **Fail-Safe Invariants:** Attempting predictions after shutdown raises `RuntimeClosedError`.

### 3. Input Validation & Fault Isolation
- **Comprehensive Validation:** Verifies tensor rank (scalar/0D inputs rejected), feature dimension matching, and non-finite value checks (`NaN`/`Inf` rejected with `TensorForgeInputError`).
- **Failure Recovery:** A failed or rejected request never corrupts model weights, compiled plans, profiler metrics, or subsequent prediction requests.

### 4. Operational Health & Reliability Statistics
- **`runtime.health()`:** Lightweight report containing lifecycle state, `accepting_requests`, `active_requests`, `rejected_requests`, `resource_limit_violations`, `last_error`, and active limits.
- **`runtime.stats()`:** Extended diagnostic telemetry tracking accepted/completed/failed/rejected requests, peak concurrent requests, timeout occurrences, and input validation failures.

---

## Production Runtime Usage Examples

### 1. Configuring Resource Limits & Handling Admission Control

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.utils.validation import RuntimeBusyError, RuntimeLimitError

# 1. Define production limits
limits = RuntimeLimits(
    max_batch_size=32,
    max_input_elements=2048,
    max_workspace_bytes=1024 * 1024,  # 1 MB workspace limit
    max_concurrent_requests=8,
    max_prediction_time_ms=100.0,
)

# 2. Load model with limits
runtime = InferenceRuntime.load("classifier.tfmodel", limits=limits)

# 3. Safe inference with exception handling
try:
    x = tf.randn((16, 64))
    output = runtime.predict(x)
except RuntimeLimitError as e:
    print(f"Request violated resource constraints: {e}")
except RuntimeBusyError as e:
    print(f"Runtime is currently at peak capacity: {e}")
```

### 2. Inspecting Operational Diagnostics & Health

```python
health = runtime.health()
print(f"Status: {health['status']}, Lifecycle: {health['lifecycle_state']}")
print(f"Active Requests: {health['active_requests']} / Max: {health['max_concurrent_requests']}")
print(f"Total Completed: {health['completed_requests']}, Rejected: {health['rejected_requests']}")
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
│   ├── __init__.py                  # Top-level exports & version (1.5.0)
│   ├── inference/                   # Production Inference, Compiler, Safety & Profiler
│   │   ├── __init__.py              # Public inference exports
│   │   ├── limits.py                # NEW: RuntimeLimits & RuntimeState
│   │   ├── context.py               # ExecutionContext (with request_id) & ExecutionContextPool
│   │   ├── profiler.py              # ProfileEvent, ProfileSession, RuntimeProfiler, PerformanceReport
│   │   ├── compiler.py              # InferenceCompiler & thread-safe CompiledPlanCache
│   │   ├── plan.py                  # Immutable ExecutionPlan & ExecutionStep IR
│   │   ├── shapes.py                # Static ShapePropagator engine
│   │   ├── memory.py                # Interval-based MemoryPlanner & MemoryPlan
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime (safety limits, admission, predict, health)
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
│   ├── inference/                   # Inference, Safety, Reliability & Observability tests
│   │   ├── test_runtime_limits.py   # NEW: RuntimeLimits configuration & enforcement
│   │   ├── test_runtime_errors.py   # NEW: Exception hierarchy & inheritance
│   │   ├── test_request_isolation.py# NEW: Request ID & context isolation
│   │   ├── test_resource_protection.py # NEW: Workspace memory limit protection
│   │   ├── test_concurrent_limits.py# NEW: Concurrency limits & RuntimeBusyError
│   │   ├── test_runtime_shutdown.py # NEW: Lifecycle transitions & graceful shutdown
│   │   ├── test_failure_recovery.py # NEW: Fault tolerance & recovery after failures
│   │   ├── test_input_validation.py # NEW: Comprehensive input validation checks
│   │   ├── test_profiler.py
│   │   ├── test_profile_session.py
│   │   ├── test_latency_stats.py
│   │   ├── test_backend_stats.py
│   │   ├── test_memory_stats.py
│   │   ├── test_compiler_stats.py
│   │   ├── test_concurrent_profiling.py
│   │   ├── test_profiling_overhead.py
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
│   ├── production_runtime_demo.py   # NEW: Comprehensive production safety demo
│   ├── profiling_demo.py
│   ├── concurrent_inference_demo.py
│   ├── parallel_inference_demo.py
│   ├── compiled_inference_demo.py
│   ├── inference_demo.py
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── runtime_safety_overhead.py   # NEW: Safety & admission control overhead benchmark
│   ├── profiling_overhead.py
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
| **v1.5 – Production Reliability, Resource Management & Runtime Safety** | **Complete** | RuntimeLimits, admission control, input validation, request IDs, failure recovery, graceful shutdown |
