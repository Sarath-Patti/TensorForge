# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.7 – Production Inference Observability & Performance Analytics`

> **Development Status:** `v1.7 (Production Release)`
> TensorForge v1.7 introduces a unified, low-overhead observability and performance analytics subsystem (`MetricsCollector`, `PerformanceSnapshot`). It provides end-to-end telemetry across both single-request and dynamically batched workloads: request lifecycle counters, queue latency distributions, execution and end-to-end percentiles (min, max, mean, p50, p90, p95, p99) via bounded reservoirs, monotonic throughput tracking (requests/sec, samples/sec, batches/sec), backend execution and fallback analytics, compiler cache hit rates, workspace memory utilization, thread-safe reset mechanics, and structured JSON export.

---

## End-to-End System Architecture

```
                    Inference Request (predict / submit)
                                   │
                                   ▼
                            v1.6 Scheduler
                                   │
                     ┌─────────────┴─────────────┐
                     │                           │
                Request Metrics             Batch Metrics
                     │                           │
                     └─────────────┬─────────────┘
                                   ▼
                          Metrics Collector (MetricsCollector)
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
                  Runtime       Backend       Memory
                  Metrics       Metrics       Metrics
                     │             │             │
                     └─────────────┼─────────────┘
                                   ▼
                         PerformanceSnapshot
                                   │
                     ┌─────────────┼─────────────┐
                     ▼             ▼             ▼
                  Console       JSON Export    Profiler
```

---

## Key Features & Subsystems

### 1. Unified Observability Engine (`tensorforge/inference/observability.py`)
- **`MetricsCollector`**: Thread-safe central metrics accumulator recording request outcomes, queue times, execution durations, backend dispatches, dynamic batches, compiler cache hits, and workspace memory.
- **`PerformanceSnapshot`**: Immutable, comprehensive diagnostic snapshot containing structured dataclasses:
  - `requests`: Submitted, completed, failed, rejected, cancelled, active, and queue depth.
  - `batches`: Formed, processed samples, average batch size, and capacity utilization.
  - `latency`: Bounded reservoir percentile distributions (`p50`, `p90`, `p95`, `p99`, `mean`, `min`, `max`) for queue wait, execution, and end-to-end duration.
  - `throughput`: Monotonic rates for requests/sec, samples/sec, and batches/sec.
  - `backends`: Invocation breakdown across NumPy, Native, Fused, Compiled, and Fallback reasons.
  - `compiler`: Cache lookups, hits, misses, hit rate, and plan compilation time.
  - `memory`: Workspace bytes, peak workspace, planned memory, and parameter sizes.
  - `scheduler`: Queue capacity, configured max batch, timeout, and lifecycle state.

### 2. Runtime & Scheduler Integration
- **`runtime.performance_snapshot()` & `scheduler.performance_snapshot()`**: Generate point-in-time immutable performance analytics snapshots.
- **`runtime.export_metrics("metrics.json")`**: Save analytics snapshots to structured JSON.
- **`runtime.reset_metrics()`**: Safely reset counters and latency reservoirs without altering runtime configuration or active models.

### 3. Bounded Memory & Low-Overhead Design
- **`LatencyHistogram`**: Ring buffer reservoir with $O(1)$ memory bounds (default: 2048 samples) ensuring predictable overhead during continuous, high-throughput inference serving.

---

## Usage Examples

### 1. Generating Performance Analytics Snapshots

```python
import tensorforge as tf
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    SchedulerConfig,
)

# 1. Load model and start scheduler
runtime = InferenceRuntime.load("classifier.tfmodel").compile(input_shape=(16, 64))
scheduler = InferenceScheduler(runtime, config=SchedulerConfig(max_batch_size=16, batch_timeout_ms=2.0))

# 2. Run predictions
for _ in range(20):
    _ = scheduler.predict(tf.randn((2, 64)))

# 3. Obtain performance snapshot
snapshot = scheduler.performance_snapshot()

# Print formatted human-readable summary
print(snapshot.summary())

# 4. Export to JSON
scheduler.export_metrics("inference_metrics.json", indent=2)

# 5. Reset metrics
scheduler.reset_metrics()

scheduler.close()
runtime.close()
```

---

## Package Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime Subsystem
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.7.0)
│   ├── inference/                   # Production Inference, Compiler, Safety, Scheduler & Observability
│   │   ├── __init__.py              # Public inference exports
│   │   ├── observability.py         # NEW: MetricsCollector, PerformanceSnapshot, LatencyHistogram
│   │   ├── scheduler.py             # InferenceScheduler, SchedulerConfig, InferenceFuture
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
│   ├── inference/                   # Inference, Safety, Reliability, Scheduling & Observability tests
│   │   ├── test_observability.py    # NEW: MetricsCollector lifecycle & snapshot structure
│   │   ├── test_latency_metrics.py  # NEW: LatencyHistogram percentiles & bounding
│   │   ├── test_throughput_metrics.py # NEW: Throughput rate calculations
│   │   ├── test_backend_metrics.py  # NEW: Backend execution & fallback tracking
│   │   ├── test_scheduler_observability.py # NEW: Scheduler performance snapshot
│   │   ├── test_runtime_observability.py # NEW: Runtime performance snapshot
│   │   ├── test_metrics_reset.py    # NEW: Metrics reset mechanics
│   │   ├── test_metrics_export.py   # NEW: JSON export & deserialization
│   │   ├── test_observability_concurrency.py # NEW: Thread-safe metric collection
│   │   ├── test_scheduler.py
│   │   ├── test_dynamic_batching.py
│   │   ├── test_scheduler_queue.py
│   │   ├── test_scheduler_shutdown.py
│   │   ├── test_scheduler_failure_recovery.py
│   │   ├── test_scheduler_concurrency.py
│   │   ├── test_scheduler_statistics.py
│   │   ├── test_scheduler_backend.py
│   │   └── ...
│
├── examples/                        # Demonstrations
│   ├── observability_demo.py        # NEW: Unified performance analytics demo
│   ├── dynamic_batching_demo.py
│   ├── production_runtime_demo.py
│   ├── profiling_demo.py
│   └── concurrent_inference_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── observability_overhead.py    # NEW: Metrics collection overhead benchmark
│   ├── dynamic_batching_benchmark.py
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
| **v1.7 – Production Inference Observability & Performance Analytics** | **Complete** | MetricsCollector, PerformanceSnapshot, bounded latency histograms, throughput, backend analytics |
