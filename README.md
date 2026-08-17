# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.6 – Production Inference Scheduling & Dynamic Batching`

> **Development Status:** `v1.6 (Production Release)`
> TensorForge v1.6 introduces an in-process request scheduling and dynamic batching subsystem (`InferenceScheduler`). The scheduler sits above `InferenceRuntime`, queueing concurrent client requests with bounded backpressure (`SchedulerQueueFullError`), automatically aggregating compatible sub-batches into larger execution batches up to `max_batch_size` or upon `batch_timeout_ms`, executing them through the underlying compiled/native runtime, and demultiplexing the resulting output tensors back to the originating requests.

---

## End-to-End System Architecture

```
                  Inference Request (predict / submit)
                                 │
                                 ▼
                         Runtime Admission
                                 │
                                 ▼
                          Request Scheduler (InferenceScheduler)
                                 │
                          ┌──────┴──────┐
                          │             │
                      Queue          Batching (FIFO / LARGEST_BATCH_FIRST)
                          │             │
                          └──────┬──────┘
                                 ▼
                          Dynamic Batch
                                 │
                                 ▼
                          InferenceRuntime (FP32 / INT8, Compiled / Fused / Native)
                                 │
                          Batch Output
                                 │
                                 ▼
                        Result Demultiplexer
                                 │
                          ┌──────┼──────┐
                          ▼      ▼      ▼
                        Req A  Req B  Req C
```

---

## Key Features & Subsystems

### 1. In-Process Inference Scheduler (`tensorforge/inference/scheduler.py`)
- **`InferenceScheduler`**: Wraps any `InferenceRuntime` to manage request concurrency and dynamic batching.
- **`SchedulerConfig`**:
  - `max_batch_size`: Maximum dynamically aggregated batch dimension.
  - `max_queue_size`: Bounded queue depth protecting against memory exhaustion.
  - `batch_timeout_ms`: Maximum time to wait for a full batch before dispatching.
  - `policy`: Scheduling policy (`FIFO` or `LARGEST_BATCH_FIRST`).
  - `drain_on_close`: Whether to drain pending requests prior to shutdown.

### 2. Dynamic Batching & Result Demultiplexing
- **Automatic Batch Assembly**: Combines compatible 1D and 2D tensor requests along the batch dimension into a single contiguous tensor.
- **Batch Compatibility Rules**: Verifies rank, feature dimensions, and dtype compatibility before joining a batch. Incompatible requests remain queued for compatible batches.
- **Precise Output Slicing**: Slices and demultiplexes output rows directly back to individual request futures without data leakage across requests.

### 3. Synchronous & Asynchronous APIs
- **Synchronous API (`scheduler.predict(x)`)**: Blocks until the dynamically formed batch completes and returns the sliced output tensor.
- **Asynchronous API (`scheduler.submit(x) -> InferenceFuture`)**: Returns a lightweight future supporting `.result()`, `.done()`, `.exception()`, and pre-execution `.cancel()`.

### 4. Backpressure & Lifecycle Protection
- **Bounded Queue**: Immediately rejects requests when queue capacity is reached (`SchedulerQueueFullError`).
- **Graceful Draining & Shutdown**: `scheduler.close(drain=True)` finishes in-flight requests and cleanly joins the background worker thread.
- **Post-Shutdown Protection**: Rejects submissions with `SchedulerClosedError`.

---

## Usage Examples

### 1. Initializing Dynamic Batching Scheduler

```python
import tensorforge as tf
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    SchedulerConfig,
    SchedulingPolicy,
)

# 1. Load compiled runtime
runtime = InferenceRuntime.load("model.tfmodel")

# 2. Configure dynamic batching
config = SchedulerConfig(
    max_batch_size=32,
    max_queue_size=128,
    batch_timeout_ms=2.0,
    policy=SchedulingPolicy.FIFO,
)

# 3. Initialize scheduler
scheduler = InferenceScheduler(runtime, config=config)

# 4. Synchronous prediction
x = tf.randn((2, 64))
out = scheduler.predict(x)

# 5. Asynchronous submission
future = scheduler.submit(x)
out = future.result(timeout=1.0)

# 6. Shutdown
scheduler.close()
runtime.close()
```

---

## Package Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime Subsystem
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.6.0)
│   ├── inference/                   # Production Inference, Compiler, Safety & Scheduler
│   │   ├── __init__.py              # Public inference exports
│   │   ├── scheduler.py             # NEW: InferenceScheduler, SchedulerConfig, InferenceFuture
│   │   ├── limits.py                # RuntimeLimits & RuntimeState
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
│   ├── inference/                   # Inference, Safety, Reliability & Scheduling tests
│   │   ├── test_scheduler.py        # NEW: Basic scheduler construction & predict
│   │   ├── test_dynamic_batching.py # NEW: Dynamic batch assembly & demultiplexing
│   │   ├── test_scheduler_queue.py  # NEW: Bounded queue & backpressure rejection
│   │   ├── test_scheduler_shutdown.py # NEW: Draining lifecycle & close
│   │   ├── test_scheduler_failure_recovery.py # NEW: Fault tolerance across batches
│   │   ├── test_scheduler_concurrency.py # NEW: Multi-producer concurrent submissions
│   │   ├── test_scheduler_statistics.py # NEW: Health & stats telemetry
│   │   └── test_scheduler_backend.py # NEW: Compiled & quantized runtime integration
│   ├── serialization/
│   ├── quantization/
│   ├── backend/
│   ├── autograd/
│   ├── nn/
│   └── optim/
│
├── examples/                        # Demonstrations
│   ├── dynamic_batching_demo.py     # NEW: Dynamic batching and scheduler demo
│   ├── production_runtime_demo.py
│   ├── profiling_demo.py
│   └── concurrent_inference_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── dynamic_batching_benchmark.py # NEW: Dynamic batching vs direct benchmark
│   ├── runtime_safety_overhead.py
│   └── profiling_overhead.py
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
| **v1.6 – Production Inference Scheduling & Dynamic Batching** | **Complete** | InferenceScheduler, SchedulerConfig, dynamic batching, result demultiplexing, queue backpressure |
