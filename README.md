# TensorForge

**TensorForge** is a memory-aware deep learning framework and high-performance production inference engine built from first principles in Python and C++17.

---

## Current Milestone: `v1.1 – Inference Compiler & Execution Planning`

> **Development Status:** `v1.1 (Production Release)`
> TensorForge v1.1 introduces an ahead-of-time **Inference Compiler & Memory Planner** (`InferenceCompiler`, `ExecutionPlan`, `WorkspaceArena`). Building upon v1.0 operator fusion, v1.1 statically analyzes tensor shape flows, conducts buffer liveness analysis to eliminate steady-state memory allocations, pre-binds optimal native C++ kernel dispatches, and caches execution plans for zero-overhead repeated predictions across FP32 and INT8 workloads.

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
  ├── Buffer Lifetime & Reuse Planning (MemoryPlanner)
  └── Ahead-of-Time Kernel Resolution (ExecutionStep)
  │
  ▼
Execution Plan IR (ExecutionPlan, 64-Byte Aligned Ping-Pong Slots)
  │
  ▼
Native C++ Memory Arena & Kernel Execution (WorkspaceArena, CPU Kernels)
```

---

## Key Features & Compiler Subsystems

### 1. Execution Plan IR (`tensorforge/inference/plan.py`)
- **`ExecutionStep`:** Pre-resolved instruction storing operator type, input slot ID, output slot ID, input/output tensor shapes, parameter references, and pre-bound backend dispatch (`native_fused`, `native`, `numpy_fused`, `numpy`).
- **`ExecutionPlan`:** Immutable, deterministic execution schedule containing ordered execution steps, workspace memory layouts, and shape metadata.

### 2. Static Shape Propagation (`tensorforge/inference/shapes.py`)
- **`ShapePropagator`:** Infers input and output shapes through `Linear`, `ReLU`, `Sigmoid`, `Tanh`, `Softmax`, and `FusedLinear` without evaluating tensors or allocating runtime memory.
- Incompatible feature dimensions or invalid inputs trigger descriptive `ShapeError` exceptions ahead of execution.

### 3. Inference Memory Planner (`tensorforge/inference/memory.py`)
- **`MemoryPlanner`:** Conducts liveness interval analysis across the execution graph.
- Implements ping-pong workspace slot allocation (`slot 0` $\leftrightarrow$ `slot 1`), bounding intermediate memory overhead to at most two reusable memory buffers regardless of network depth.
- Computes 64-byte aligned memory offsets and exact peak workspace capacity requirements.

### 4. Native Workspace Arena (`native/include/tensorforge/arena.hpp`, `native/src/arena.cpp`)
- **`WorkspaceArena`:** RAII-managed contiguous CPU memory block allocated using 64-byte aligned allocators.
- Eliminates dynamic heap allocation and system call overhead during prediction loops.

### 5. Compiled Plan Cache (`tensorforge/inference/compiler.py`)
- **`CompiledPlanCache`:** In-memory plan cache indexed by `(graph_id, input_shape, dtype, backend, is_quantized)`.
- Eliminates redundant graph parsing, shape propagation, and memory planning for repeated predictions.

---

## Fallback & Execution Hierarchy

TensorForge provides guaranteed deterministic execution through an automated 4-tier fallback hierarchy:

```
1. Compiled Fused Native C++   ──(if native extension loaded & operands eligible)──►
2. Compiled Fused NumPy Ref    ──(if native unavailable / input requires fallback)──►
3. Eager Fused Native C++      ──(if uncompiled & native backend selected)────────►
4. Eager NumPy Reference       ──(universal base reference)────────────────────────►
```

---

## Inference Runtime Usage Examples

### 1. Compiling and Executing a Model

```python
import tensorforge as tf
from tensorforge.inference import InferenceRuntime

# 1. Load exported model artifact (.tfmodel)
runtime = InferenceRuntime.load("classifier.tfmodel")

# 2. Compile model for expected input shape
runtime.compile(input_shape=(8, 16))

# 3. Inspect compiled summary
summary = runtime.summary()
print(f"Is Compiled:       {summary['is_compiled']}")
print(f"Compiled Steps:    {summary['compiled_steps']}")
print(f"Workspace Memory:  {summary['workspace_bytes']} bytes")
print(f"Active Backend:    {summary['backend']}")

# 4. Inspect ExecutionPlan IR
print(runtime.execution_plan.summary())

# 5. Predict with zero allocation overhead
x = tf.randn((8, 16))
predictions = runtime.predict(x)

assert predictions.requires_grad is False
assert predictions.grad_fn is None
```

### 2. Dynamic Batch Compilation & Cached Execution

```python
# The runtime automatically retrieves or compiles plans for different batch sizes
out_single = runtime.predict(tf.randn((1, 16)))    # Batch size 1
out_batch  = runtime.predict(tf.randn((32, 16)))   # Batch size 32
```

### 3. INT8 Low-Precision Compiled Inference

```python
from tensorforge.inference import InferenceRuntime

# Load quantized model and compile
runtime_int8 = InferenceRuntime.load("quantized_classifier.tfmodel").compile(input_shape=(8, 16))

print(f"Quantized: {runtime_int8.is_quantized}")
print(f"Compiled:  {runtime_int8.is_compiled}")

output = runtime_int8.predict(x)
```

---

## Running Benchmarks & Demonstrations

```bash
# Compiled inference demonstration
python examples/compiled_inference_demo.py

# Production inference performance benchmark (Eager vs Fused vs Compiled)
python benchmarks/benchmark_inference.py

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
│   │   ├── arena.hpp                # NEW: 64-byte aligned workspace arena
│   │   ├── dtype.hpp                # DType enum & traits
│   │   ├── kernels.hpp              # Unfused & Fused inference kernels
│   │   ├── shape.hpp                # Shape & contiguous stride math
│   │   ├── storage.hpp              # Native Storage buffer
│   │   └── tensor.hpp               # Native Tensor abstraction
│   ├── src/                         # Native implementations & pybind11 bindings
│   │   ├── allocator.cpp
│   │   ├── arena.cpp                # NEW: WorkspaceArena implementation
│   │   ├── bindings.cpp             # pybind11 module bindings
│   │   ├── dtype.cpp
│   │   ├── kernels.cpp              # Handcrafted SIMD-friendly compute kernels
│   │   ├── shape.cpp
│   │   ├── storage.cpp
│   │   └── tensor.cpp
│   └── tests/
│       └── test_native.cpp          # Standalone C++ test suite
│
├── tensorforge/
│   ├── __init__.py                  # Top-level exports & version (1.1.0)
│   ├── inference/                   # Production Inference & Compiler Subsystem
│   │   ├── __init__.py              # Public inference exports
│   │   ├── compiler.py              # NEW: InferenceCompiler & CompiledPlanCache
│   │   ├── plan.py                  # NEW: ExecutionPlan and ExecutionStep IR
│   │   ├── shapes.py                # NEW: Static ShapePropagator engine
│   │   ├── memory.py                # NEW: MemoryPlanner & buffer lifetime reuse
│   │   ├── graph.py                 # InferenceGraph and InferenceNode representations
│   │   ├── fusion.py                # OperatorFusionPass pattern matching engine
│   │   ├── optimizer.py             # GraphOptimizer execution dispatcher
│   │   ├── runtime.py               # InferenceRuntime engine with compile() API
│   │   └── loader.py                # ModelLoader & architecture reconstitution
│   ├── serialization/               # Model Serialization Subsystem (.tfmodel, .tfckpt)
│   ├── quantization/                # Quantization Subsystem (INT8, Calibration, qmatmul)
│   ├── backend/                     # Multi-backend Dispatcher Subsystem
│   ├── optim/                       # Optimizer subsystem (SGD, Adam)
│   ├── nn/                          # Neural Network subsystem (Linear, Sequential, etc.)
│   ├── autograd/                    # Automatic Differentiation DAG engine
│   ├── tensor/                      # Core Tensor subsystem
│   └── utils/                       # Validation & Exception hierarchy
│
├── tests/                           # Python Test Suite
│   ├── inference/                   # Inference & Compiler test suite
│   │   ├── test_compiler.py         # NEW: Graph to ExecutionPlan conversion tests
│   │   ├── test_shapes.py           # NEW: Static shape propagation tests
│   │   ├── test_memory_planner.py   # NEW: Buffer lifetime & reuse tests
│   │   ├── test_compiled_runtime.py # NEW: compile(), caching, and execution tests
│   │   ├── test_fusion.py           # Operator fusion pattern tests
│   │   ├── test_fused_correctness.py# Fused mathematical parity tests
│   │   ├── test_fused_backend_dispatch.py # Multi-backend execution tests
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
│   ├── compiled_inference_demo.py   # NEW: Compiled inference & execution plan demo
│   ├── inference_demo.py            # Operator fusion demo
│   ├── serialization_demo.py
│   ├── quantization_demo.py
│   └── training_demo.py
│
├── benchmarks/                      # Performance Benchmarks
│   ├── benchmark_inference.py       # Eager vs Fused vs Compiled benchmark
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
