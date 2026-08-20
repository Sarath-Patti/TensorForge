# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v2.0 – Production Runtime API & Deployment Foundation`

> **Development Status:** `v2.0 (Production Release)`
> TensorForge v2.0 formalizes the production-facing runtime API, deployment manifest bootstrapping system, and pre-packaged workload runtime profiles (`LOW_LATENCY`, `HIGH_THROUGHPUT`, `BALANCED`, `EMBEDDED`). It introduces the high-level `InferenceClient` application interface, standardized `InferenceRequestContract` SLA specifications, declarative `DeploymentManifest` JSON loaders (`InferenceServer.from_manifest()`), context manager support, vector/batch prediction helpers, and production diagnostics.

> [!NOTE]
> `InferenceServer` and `InferenceClient` provide an in-process serving abstraction designed for high-performance Python application integration, embedded inference engines, and framework backends. It is **not** a distributed HTTP/gRPC network daemon, process supervisor, or cloud cluster manager.

---

## End-to-End System Architecture

```
                    Application
                         │
                         ▼
                 InferenceClient
                         │
                 Request Contract
                         │
                         ▼
                 InferenceServer
                         │
            ┌────────────┴────────────┐
            │                         │
        Model Routing            Runtime Config
            │                         │
            ▼                         ▼
      Model Version              Runtime Profile
            │
            ▼
       InferenceRuntime
            │
       Compiler / Scheduler
            │
            ▼
       Native / NumPy
```

---

## Key Features & Subsystems

### 1. In-Process Multi-Model Server (`tensorforge/inference/server.py`)
- **`InferenceServer`**: Central serving manager orchestrating model runtimes, dynamic batch schedulers, resource limits, and observability.
- **`ModelRegistry`**: Thread-safe registry mapping `(model_name, version) -> ModelEntry` and active version pointers.
- **Model Lifecycle States**: Deterministic state transitions (`UNLOADED` → `LOADING` → `READY` → `DRAINING` → `FAILED`).
- **Model Versioning & Atomic Switching**: Supports multiple version registrations (`classifier:1`, `classifier:2`) and zero-downtime active version switching (`set_active_version`).
- **Atomic Model Reloading**: Loads new runtime in staging state; on validation success, atomically swaps registry pointer and drains old version (`reload_model`).
- **Per-Model Queue & Resource Isolation**: Each model version owns its dedicated `InferenceScheduler` and `InferenceRuntime`, preventing queue contention and batch pollution across models.
- **Multi-Model Observability Aggregation**: Aggregates model-specific request counters, queue wait distributions, throughput rates, backend dispatches, and workspace memory into unified server performance snapshots.

---

## Usage Examples

### 1. Multi-Model Inference Serving & Version Routing

```python
import tensorforge as tf
from tensorforge.inference import (
    InferenceServer,
    SchedulerConfig,
    ServerConfig,
)

# 1. Initialize server
with InferenceServer(config=ServerConfig(max_loaded_models=5)) as server:

    # 2. Load model versions
    server.load_model(
        name="classifier",
        path="classifier_v1.tfmodel",
        version="1",
        active=True,
        scheduler_config=SchedulerConfig(max_batch_size=8, batch_timeout_ms=3.0),
    )

    server.load_model(
        name="classifier",
        path="classifier_v2.tfmodel",
        version="2",
        active=False,
    )

    # 3. Synchronous prediction (default active version 1)
    x = tf.randn((2, 16))
    output_v1 = server.predict("classifier", x)

    # 4. Explicit version routing (version 2)
    output_v2 = server.predict("classifier", x, version="2")

    # 5. Atomic active version switch
    server.set_active_version("classifier", "2")

    # 6. Inspect server health & performance analytics
    print(server.health())

    # 7. Export performance analytics to JSON
    server.export_metrics("server_performance.json", indent=2)

    # 8. Unload old model version
    server.unload_model("classifier", version="1")
```

---

## Package Structure

```
TensorForge/
├── native/                          # Native C++17 Runtime Subsystem
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.8.0)
│   ├── inference/                   # Production Inference, Compiler, Safety, Scheduler, Observability & Serving
│   │   ├── __init__.py              # Public inference exports
│   │   ├── server.py                # NEW: InferenceServer, ModelRegistry, ModelEntry, ServerConfig
│   │   ├── observability.py         # MetricsCollector, PerformanceSnapshot, LatencyHistogram
│   │   ├── scheduler.py             # InferenceScheduler, SchedulerConfig, InferenceFuture
│   │   ├── limits.py                # RuntimeLimits & RuntimeState
│   │   ├── context.py               # ExecutionContext & ExecutionContextPool
│   │   ├── profiler.py              # ProfileEvent, ProfileSession, RuntimeProfiler
│   │   ├── compiler.py              # InferenceCompiler & thread-safe CompiledPlanCache
│   │   ├── plan.py                  # Immutable ExecutionPlan & ExecutionStep IR
│   │   ├── shapes.py                # Static ShapePropagator engine
│   │   ├── memory.py                # Interval-based MemoryPlanner & MemoryPlan
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime
│   │   └── loader.py                # ModelLoader & architecture reconstitution
│   ├── serialization/               # Model Serialization Subsystem (.tfmodel, .tfckpt)
│   ├── quantization/                # Quantization Subsystem (INT8, Calibration, qmatmul)
│   ├── backend/                     # Multi-backend Dispatcher & Thread Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── nn/                          # Neural Network subsystem
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/                       # Validation & Exception hierarchy
│
├── tests/                           # Python Test Suite
│   ├── inference/                   # Inference, Safety, Reliability, Scheduling, Observability & Serving tests
│   │   ├── test_server.py           # NEW: InferenceServer construction and basic load/predict
│   │   ├── test_model_registry.py   # NEW: ModelRegistry active version resolution and duplicates
│   │   ├── test_model_versions.py   # NEW: Multi-version loading and active switching
│   │   ├── test_server_routing.py   # NEW: Sync predict and async submit routing
│   │   ├── test_server_lifecycle.py # NEW: Server lifecycle states and draining
│   │   ├── test_model_unload.py     # NEW: Model unloading and resource release
│   │   ├── test_model_reload.py     # NEW: Safe atomic model reloading
│   │   ├── test_server_concurrency.py # NEW: Multi-threaded server prediction
│   │   ├── test_server_health.py    # NEW: Server health report structure
│   │   ├── test_server_statistics.py # NEW: Server stats aggregation
│   │   ├── test_server_failure_isolation.py # NEW: Failure isolation between models
│   │   ├── test_server_limits.py    # NEW: Max loaded models server limits
│   │   ├── test_server_observability.py # NEW: Server performance snapshot and JSON export
│   │   └── ...
│
├── examples/                        # Demonstrations
│   ├── inference_server_demo.py     # NEW: Multi-model serving demonstration
│   ├── observability_demo.py
│   ├── dynamic_batching_demo.py
│   └── ...
│
├── benchmarks/                      # Performance Benchmarks
│   ├── inference_server_benchmark.py # NEW: Inference server overhead benchmark
│   ├── observability_overhead.py
│   └── ...
│
├── README.md
├── pyproject.toml
└── setup.py
```

---

## Development Status & Roadmap

| Milestone | Status | Description |
|---|---|---|
| **v0.1 – Project Foundation & Tensor Core** | **Complete** | Tensor abstraction, metadata/storage decoupling, basic ops, broadcasting |
| **v0.2 – Automatic Differentiation** | **Complete** | Reverse-mode autodiff DAG engine, topological backpropagation |
| **v0.3 – Neural Network Modules & Layers** | **Complete** | Parameter, Module, Linear, Activations, Losses, Sequential |
| **v0.4 – Optimizers & Training Pipeline** | **Complete** | SGD, Adam, Dataset, DataLoader, Trainer, Metrics |
| **v0.5 – Native Runtime & Performance Foundation** | **Complete** | C++17 runtime, CPU allocator, native storage, CPU kernels |
| **v0.6 – Native Operation Dispatch & Runtime Integration** | **Complete** | Backend dispatcher, runtime backend switching, automatic NumPy fallback |
| **v0.7 – Quantization Runtime & INT8 Inference** | **Complete** | QuantizedTensor, symmetric & asymmetric INT8 quantization, INT8 matmul |
| **v0.8 – Model Serialization & Checkpointing** | **Complete** | Safe .tfmodel & .tfckpt formats, state_dict, training resumption |
| **v0.9 – Portable Inference Runtime & Model Export** | **Complete** | Dedicated InferenceRuntime, ModelLoader, multi-backend dispatch |
| **v1.0 – Production Inference Engine & Operator Fusion** | **Complete** | InferenceGraph, OperatorFusionPass, native fused C++ kernels |
| **v1.1 – Inference Compiler & Execution Planning** | **Complete** | InferenceCompiler, ExecutionPlan IR, static shape propagation, memory planner |
| **v1.2 – Runtime Memory Optimization & Parallel CPU Execution** | **Complete** | Interval memory planner, MemoryRegions, native ThreadPool, parallel CPU kernels |
| **v1.3 – Production Runtime Reliability & Concurrency** | **Complete** | Thread-safe InferenceRuntime, ExecutionContext pool, workspace isolation |
| **v1.4 – Runtime Observability & Performance Diagnostics** | **Complete** | Profiler subsystem, ProfileEvent, ProfileSession, PerformanceReport |
| **v1.5 – Production Reliability, Resource Management & Runtime Safety** | **Complete** | RuntimeLimits, admission control, input validation, failure recovery |
| **v1.6 – Production Inference Scheduling & Dynamic Batching** | **Complete** | InferenceScheduler, SchedulerConfig, dynamic batching, result demultiplexing |
| **v1.7 – Production Inference Observability & Performance Analytics** | **Complete** | MetricsCollector, PerformanceSnapshot, bounded latency histograms |
| **v1.8 – Production Inference Serving Layer** | **Complete** | InferenceServer, ModelRegistry, model versioning, routing, failure isolation |
