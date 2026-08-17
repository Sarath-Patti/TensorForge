"""Dedicated Production Inference Runtime with Ahead-of-Time Compilation, Multi-Threaded Parallel Execution, and Thread-Safe Concurrency."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.autograd.engine import no_grad
from tensorforge.backend.dispatcher import (
    backend_context,
    get_backend,
    get_last_backend,
    get_num_threads,
    set_backend,
    set_num_threads as set_backend_num_threads,
)
from tensorforge.inference.compiler import InferenceCompiler
from tensorforge.inference.context import ExecutionContext, ExecutionContextPool
from tensorforge.inference.graph import InferenceGraph
from tensorforge.inference.loader import ModelLoader
from tensorforge.inference.memory import MemoryPlan
from tensorforge.inference.optimizer import GraphOptimizer
from tensorforge.inference.plan import ExecutionPlan
from tensorforge.inference.profiler import PerformanceReport, ProfileEvent, ProfileSession, RuntimeProfiler
from tensorforge.nn.linear import Linear
from tensorforge.nn.module import Module
from tensorforge.nn.sequential import Sequential
from tensorforge.quantization.quantize import qmatmul, quantize
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import float32
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import RuntimeClosedError, TensorForgeError

try:
    import _tensorforge_native as _native
except ImportError:
    try:
        from tensorforge import _tensorforge_native as _native
    except ImportError:
        _native = None


class InferenceRuntime:
    """A thread-safe production-grade inference runtime supporting Operator Fusion, Memory Planning, and Parallel Concurrency.

    Loads serialized .tfmodel artifacts, reconstructs network graphs, supports graph-level
    operator fusion (Linear+ReLU, Linear+Sigmoid, Linear+Tanh, Linear+Softmax), compiles models
    into deterministic ExecutionPlans with memory region reuse, and executes forward predictions
    with multi-threaded CPU parallel kernels, strict no_grad guarantees, and per-prediction workspace isolation.

    Concurrency Guarantees:
        - Thread-Safe: Multiple Python threads may concurrently invoke `predict()` on the exact same
          InferenceRuntime instance without race conditions or memory corruption.
        - Workspace Isolation: Each prediction obtains a dedicated `ExecutionContext` from a thread-safe
          pool, ensuring no intermediate activation buffers are shared across concurrent calls.
        - Lock-Free Compute: Model parameters are immutable; predictions execute in parallel without
          global execution serialization locks.
        - Lifecycle Safety: Calling `predict()` on a closed runtime deterministically raises `RuntimeClosedError`.

    Args:
        model: Reconstructed Module instance in evaluation mode.
        metadata: Model archive metadata.
        is_quantized: Whether model parameters are stored in INT8 low precision.
        state_dict: State dictionary containing raw or quantized parameters.
        backend: Optional backend override ('numpy' or 'native').
        num_threads: Number of CPU threads to configure for native parallel execution.
    """

    def __init__(
        self,
        model: Module,
        metadata: Dict[str, Any],
        is_quantized: bool = False,
        state_dict: Optional[Dict[str, Any]] = None,
        backend: Optional[str] = None,
        num_threads: Optional[int] = None,
    ) -> None:
        self._model: Module = model
        self._metadata: Dict[str, Any] = metadata
        self._is_quantized: bool = is_quantized
        self._state_dict: Dict[str, Any] = state_dict or {}
        self._backend: Optional[str] = backend
        self._num_threads: int = num_threads if num_threads is not None else get_num_threads()

        self._model.eval()

        # Graph optimization state
        self._graph: InferenceGraph = InferenceGraph.from_module(self._model, self._state_dict)
        self._optimized_graph: Optional[InferenceGraph] = None
        self._is_optimized: bool = False
        self._optimization_stats: Dict[str, Any] = {
            "original_nodes": len(self._graph),
            "optimized_nodes": len(self._graph),
            "fused_count": 0,
            "fused_patterns": [],
        }

        # Compilation & Execution Plan state
        self._compiled_plan: Optional[ExecutionPlan] = None
        self._is_compiled: bool = False
        self._arena: Optional[Any] = None

        # Profiler and telemetry state
        self._profiler: RuntimeProfiler = RuntimeProfiler()

        # Thread-safe execution context pool & concurrency locks
        self._context_pool: ExecutionContextPool = ExecutionContextPool()
        self._lifecycle_lock: threading.Lock = threading.Lock()
        self._compile_lock: threading.RLock = threading.RLock()
        self._config_lock: threading.Lock = threading.Lock()
        self._stats_lock: threading.Lock = threading.Lock()
        self._is_closed: bool = False
        self._prediction_count: int = 0
        self._error_count: int = 0

        # Infer input and output dimensions
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._output_shape: Optional[Tuple[int, ...]] = None
        self._infer_shapes()

    def _infer_shapes(self) -> None:
        """Infer input and output feature shapes from model structure."""
        if isinstance(self._model, Linear):
            self._input_shape = (self._model.in_features,)
            self._output_shape = (self._model.out_features,)
        elif isinstance(self._model, Sequential) and len(self._model) > 0:
            for mod in self._model:
                if isinstance(mod, Linear) and self._input_shape is None:
                    self._input_shape = (mod.in_features,)
            for mod in reversed(list(self._model)):
                if isinstance(mod, Linear) and self._output_shape is None:
                    self._output_shape = (mod.out_features,)

    @classmethod
    def load(
        cls,
        filepath: str,
        backend: Optional[str] = None,
        num_threads: Optional[int] = None,
        strict: bool = True,
    ) -> InferenceRuntime:
        """Load a .tfmodel artifact and construct an InferenceRuntime.

        Args:
            filepath: Path to the .tfmodel archive.
            backend: Optional backend override ('numpy' or 'native').
            num_threads: Number of CPU threads for native execution.
            strict: Whether to enforce strict parameter key matching.

        Returns:
            Configured InferenceRuntime instance.
        """
        model, state_dict, metadata, is_quantized = ModelLoader.load(filepath, strict=strict)
        return cls(
            model=model,
            metadata=metadata,
            is_quantized=is_quantized,
            state_dict=state_dict,
            backend=backend,
            num_threads=num_threads,
        )

    def __enter__(self) -> InferenceRuntime:
        """Context manager support for deterministic runtime resource lifecycle."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit releasing workspace resources."""
        self.close()

    def close(self) -> None:
        """Close the runtime and release all pooled execution contexts and native arenas."""
        with self._lifecycle_lock:
            if self._is_closed:
                return
            self._is_closed = True
            self._context_pool.clear()
            if self._arena is not None:
                if hasattr(self._arena, "reset"):
                    self._arena.reset()
                self._arena = None

    @property
    def is_closed(self) -> bool:
        """Whether the runtime has been closed."""
        with self._lifecycle_lock:
            return self._is_closed

    @property
    def closed(self) -> bool:
        """Alias for is_closed."""
        return self.is_closed

    def set_num_threads(self, num_threads: int) -> InferenceRuntime:
        """Set the number of CPU threads used for parallel inference execution.

        Thread-safe: Safely reconfigures thread counts and updates compiled execution plans.

        Args:
            num_threads: Number of worker threads (must be >= 1).

        Returns:
            Self (enables method chaining).
        """
        if not isinstance(num_threads, int) or num_threads < 1:
            raise TensorForgeError(f"num_threads must be an integer >= 1, got {num_threads}.")

        with self._config_lock:
            with self._lifecycle_lock:
                if self._is_closed:
                    raise RuntimeClosedError("Cannot configure threads on a closed InferenceRuntime.")

            self._num_threads = num_threads
            set_backend_num_threads(num_threads)

            # If already compiled, invalidate plan so it recompiles with new thread configuration
            with self._compile_lock:
                if self._is_compiled and self._compiled_plan is not None:
                    self.compile(
                        input_shape=self._compiled_plan.input_shape,
                        backend=self._backend,
                        num_threads=num_threads,
                    )

        return self

    @property
    def num_threads(self) -> int:
        """Current number of configured CPU worker threads."""
        return self._num_threads

    def optimize(self) -> InferenceRuntime:
        """Perform graph-level operator fusion and kernel optimizations.

        Collapses adjacent fusible layers (Linear+ReLU, Linear+Sigmoid,
        Linear+Tanh, Linear+Softmax) into single FusedLinear execution nodes.

        Returns:
            Self (enables method chaining).
        """
        with self._compile_lock:
            with self._lifecycle_lock:
                if self._is_closed:
                    raise RuntimeClosedError("Cannot optimize a closed InferenceRuntime.")

            self._optimized_graph, self._optimization_stats = GraphOptimizer.optimize(self._graph)
            self._is_optimized = True
        return self

    def compile(
        self,
        input_shape: Tuple[int, ...],
        backend: Optional[str] = None,
        num_threads: Optional[int] = None,
        use_cache: bool = True,
    ) -> InferenceRuntime:
        """Compile the inference graph into a reusable, memory-planned ExecutionPlan.

        Args:
            input_shape: Input tensor shape (e.g. (batch_size, in_features) or (in_features,)).
            backend: Optional backend override.
            num_threads: Optional CPU thread count override.
            use_cache: Whether to use plan caching.

        Returns:
            Self (enables method chaining).
        """
        with self._compile_lock:
            with self._lifecycle_lock:
                if self._is_closed:
                    raise RuntimeClosedError("Cannot compile a closed InferenceRuntime.")

            if not self._is_optimized:
                self.optimize()

            target_graph = self.graph
            target_backend = backend if backend is not None else self.backend
            target_threads = num_threads if num_threads is not None else self._num_threads

            # Normalize 1D input shape to 2D (1, in_features) if necessary
            normalized_shape = (1, input_shape[0]) if len(input_shape) == 1 else input_shape

            self._compiled_plan = InferenceCompiler.compile(
                graph=target_graph,
                input_shape=normalized_shape,
                backend=target_backend,
                dtype=float32,
                is_quantized=self._is_quantized,
                num_threads=target_threads,
                use_cache=use_cache,
                profiler=self._profiler,
            )

            self._is_compiled = True
            self._num_threads = target_threads

            # Initialize native workspace arena if native runtime is available
            if target_backend == "native" and _native is not None and hasattr(_native, "WorkspaceArena"):
                ws_bytes = self._compiled_plan.total_workspace_bytes
                if ws_bytes > 0:
                    self._arena = _native.WorkspaceArena(ws_bytes)

        return self

    # -------------------------------------------------------------------------
    # Observability & Profiling API (v1.4)
    # -------------------------------------------------------------------------

    def enable_profiling(self, detailed: bool = False) -> InferenceRuntime:
        """Enable inference telemetry and runtime performance profiling.

        Args:
            detailed: Whether to record fine-grained per-operator execution events.

        Returns:
            Self (enables method chaining).

        Raises:
            RuntimeClosedError: If invoked on a closed InferenceRuntime.
        """
        with self._lifecycle_lock:
            if self._is_closed:
                raise RuntimeClosedError("Cannot enable profiling: InferenceRuntime has been closed.")
        self._profiler.enable(detailed=detailed)
        return self

    def disable_profiling(self) -> InferenceRuntime:
        """Disable inference runtime profiling and telemetry recording.

        Returns:
            Self (enables method chaining).
        """
        self._profiler.disable()
        return self

    @property
    def profiling_enabled(self) -> bool:
        """Check if profiling is currently active."""
        return self._profiler.is_enabled

    @property
    def profiling_mode(self) -> str:
        """Return active profiling mode string ('disabled', 'summary', or 'detailed')."""
        return self._profiler.mode

    def set_profile_history_size(self, size: int) -> InferenceRuntime:
        """Configure the capacity of the bounded latency history buffer for percentile calculations.

        Args:
            size: Maximum number of latencies retained in memory (must be >= 10).

        Returns:
            Self (enables method chaining).
        """
        self._profiler.set_history_size(size)
        return self

    def profile(self) -> PerformanceReport:
        """Generate a complete PerformanceReport containing latency, backend, operator, and memory diagnostics.

        Returns:
            Configured PerformanceReport instance.

        Raises:
            RuntimeClosedError: If invoked on a closed InferenceRuntime.
        """
        with self._lifecycle_lock:
            if self._is_closed:
                raise RuntimeClosedError("Cannot generate profile report: InferenceRuntime has been closed.")

        mem_summary = {
            "workspace_bytes": self.workspace_size,
            "num_regions": self.memory_plan.num_regions if self.memory_plan is not None else 0,
            "reused_buffers": self.memory_plan.num_reused_buffers if self.memory_plan is not None else 0,
            "active_contexts": self.active_contexts,
            "pooled_contexts": self._context_pool.total_count,
        }
        return self._profiler.generate_report(runtime_memory_stats=mem_summary)

    def profile_events(self) -> List[ProfileEvent]:
        """Return a list snapshot of all recorded ProfileEvent instances."""
        return self._profiler.get_events()

    def profile_session(self, detailed: bool = True) -> ProfileSession:
        """Create a scoped context manager for profiling a specific block of predictions.

        Example:
            >>> with runtime.profile_session() as session:
            ...     out = runtime.predict(x)
            >>> print(session.summary())

        Args:
            detailed: Whether to record per-operator execution events.

        Returns:
            ProfileSession context manager.

        Raises:
            RuntimeClosedError: If invoked on a closed InferenceRuntime.
        """
        with self._lifecycle_lock:
            if self._is_closed:
                raise RuntimeClosedError("Cannot start profile session: InferenceRuntime has been closed.")
        return ProfileSession(self._profiler, detailed=detailed)

    def clear_profiler(self) -> InferenceRuntime:
        """Reset and clear all collected profiling events and latency statistics.

        Returns:
            Self (enables method chaining).
        """
        self._profiler.clear()
        return self

    def reset_profiler(self) -> InferenceRuntime:
        """Alias for clear_profiler()."""
        return self.clear_profiler()

    def backend_stats(self) -> Dict[str, Any]:
        """Return backend execution telemetry."""
        return self._profiler.backend_stats()

    def compiler_stats(self) -> Dict[str, Any]:
        """Return compiler cache and compilation telemetry."""
        return self._profiler.compiler_stats()

    def latency_stats(self) -> Dict[str, Any]:
        """Return latency distribution percentiles and throughput."""
        return self._profiler.latency_stats()

    @property
    def is_optimized(self) -> bool:
        """Whether operator fusion optimizations are active."""
        return self._is_optimized

    @property
    def is_compiled(self) -> bool:
        """Whether the model has been compiled into an ExecutionPlan."""
        return self._is_compiled

    @property
    def execution_plan(self) -> Optional[ExecutionPlan]:
        """Access the active compiled ExecutionPlan."""
        return self._compiled_plan

    @property
    def memory_plan(self) -> Optional[MemoryPlan]:
        """Access detailed memory planning intervals and region allocations."""
        if self._compiled_plan is not None:
            return self._compiled_plan.memory_plan
        return None

    @property
    def workspace_size(self) -> int:
        """Planned workspace memory size in bytes."""
        if self._compiled_plan is not None:
            return self._compiled_plan.total_workspace_bytes
        return 0

    @property
    def graph(self) -> InferenceGraph:
        """Access the active (optimized or unoptimized) computation graph."""
        return self._optimized_graph if self._is_optimized and self._optimized_graph is not None else self._graph

    @property
    def fused_count(self) -> int:
        """Number of fused operator sequences in the optimized graph."""
        return int(self._optimization_stats.get("fused_count", 0))

    @property
    def fused_patterns(self) -> List[str]:
        """List of fused operator pattern names."""
        return list(self._optimization_stats.get("fused_patterns", []))

    @property
    def original_node_count(self) -> int:
        """Number of nodes in the original unoptimized graph."""
        return int(self._optimization_stats.get("original_nodes", len(self._graph)))

    @property
    def optimized_node_count(self) -> int:
        """Number of nodes in the optimized graph."""
        return int(self._optimization_stats.get("optimized_nodes", len(self.graph)))

    @property
    def model(self) -> Module:
        """Access the underlying reconstructed neural network Module."""
        return self._model

    @property
    def backend(self) -> str:
        """Return the active compute backend (configured or global)."""
        return self._backend if self._backend is not None else get_backend()

    @property
    def is_quantized(self) -> bool:
        """Whether the runtime is operating in INT8 low-precision mode."""
        return self._is_quantized

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dictionary of model metadata loaded from the .tfmodel container."""
        return self._metadata

    @property
    def input_shape(self) -> Optional[Tuple[int, ...]]:
        """Inferred expected input feature shape."""
        return self._input_shape

    @property
    def output_shape(self) -> Optional[Tuple[int, ...]]:
        """Inferred expected output feature shape."""
        return self._output_shape

    @property
    def prediction_count(self) -> int:
        """Total number of successful predictions executed on this runtime."""
        with self._stats_lock:
            return self._prediction_count

    @property
    def error_count(self) -> int:
        """Total number of prediction errors encountered on this runtime."""
        with self._stats_lock:
            return self._error_count

    @property
    def active_contexts(self) -> int:
        """Number of execution contexts currently active in concurrent predict() calls."""
        return self._context_pool.active_count

    def predict(
        self,
        input_data: Union[Tensor, np.ndarray, Sequence[Any]],
    ) -> Tensor:
        """Execute inference prediction on the given input sample or batch.

        Thread-safe: Safe for simultaneous execution by multiple Python threads.
        Each thread acquires an isolated ExecutionContext to guarantee workspace memory isolation.

        Guarantees that inference runs in `eval` mode with `no_grad` active,
        producing detached tensors with no autograd graph overhead.

        Args:
            input_data: Input tensor, NumPy array, or nested sequence.

        Returns:
            Output Tensor representing prediction results.

        Raises:
            RuntimeClosedError: If invoked on a closed InferenceRuntime.
        """
        with self._lifecycle_lock:
            if self._is_closed:
                raise RuntimeClosedError("Cannot execute predict(): InferenceRuntime has been closed.")

        if isinstance(input_data, Tensor):
            x = input_data
        elif isinstance(input_data, np.ndarray):
            x = Tensor(input_data, dtype=float32, copy=False)
        else:
            x = Tensor(np.asarray(input_data, dtype=np.float32), dtype=float32)

        target_backend = self.backend
        is_prof = self._profiler.is_enabled
        t0 = time.perf_counter_ns() if is_prof else 0

        try:
            with self._context_pool.acquire() as ctx:
                with backend_context(target_backend):
                    with no_grad():
                        # 1. Compiled Execution Path
                        if self._is_compiled and self._compiled_plan is not None:
                            # Check shape compatibility
                            if x.shape == self._compiled_plan.input_shape:
                                output = InferenceCompiler.execute_plan(
                                    self._compiled_plan,
                                    x,
                                    context=ctx,
                                    profiler=self._profiler,
                                )
                            elif len(x.shape) == len(self._compiled_plan.input_shape) and x.shape[1:] == self._compiled_plan.input_shape[1:]:
                                # Dynamic batch size recompilation (synchronized & cached)
                                plan = InferenceCompiler.compile(
                                    graph=self.graph,
                                    input_shape=x.shape,
                                    backend=target_backend,
                                    dtype=float32,
                                    is_quantized=self._is_quantized,
                                    num_threads=self._num_threads,
                                    use_cache=True,
                                    profiler=self._profiler,
                                )
                                output = InferenceCompiler.execute_plan(
                                    plan,
                                    x,
                                    context=ctx,
                                    profiler=self._profiler,
                                )
                            else:
                                # Fallback to eager optimized execution
                                eager_t0 = time.perf_counter_ns() if is_prof else 0
                                output = GraphOptimizer.execute(
                                    self.graph,
                                    x,
                                    backend=target_backend,
                                    is_quantized=self._is_quantized,
                                )
                                if is_prof:
                                    dur = time.perf_counter_ns() - eager_t0
                                    self._profiler.record_backend_op(
                                        backend_dispatch=target_backend,
                                        duration_ns=dur,
                                        is_fused=True,
                                    )
                                    if self._profiler.is_detailed:
                                        self._profiler.record_event(
                                            ProfileEvent(
                                                name="fused_graph_fallback",
                                                op_type="FusedGraph",
                                                backend=target_backend,
                                                mode="fused",
                                                start_time_ns=eager_t0,
                                                end_time_ns=eager_t0 + dur,
                                                input_shape=x.shape,
                                                output_shape=output.shape,
                                                dtype="float32",
                                                batch_size=x.shape[0] if len(x.shape) >= 2 else 1,
                                                is_fused=True,
                                                is_compiled=False,
                                            )
                                        )

                        # 2. Optimized Graph Path
                        elif self._is_optimized and self._optimized_graph is not None:
                            eager_t0 = time.perf_counter_ns() if is_prof else 0
                            output = GraphOptimizer.execute(
                                self._optimized_graph,
                                x,
                                backend=target_backend,
                                is_quantized=self._is_quantized,
                            )
                            if is_prof:
                                dur = time.perf_counter_ns() - eager_t0
                                self._profiler.record_backend_op(
                                    backend_dispatch=target_backend,
                                    duration_ns=dur,
                                    is_fused=True,
                                )
                                if self._profiler.is_detailed:
                                    self._profiler.record_event(
                                        ProfileEvent(
                                            name="optimized_graph_forward",
                                            op_type="OptimizedGraph",
                                            backend=target_backend,
                                            mode="fused",
                                            start_time_ns=eager_t0,
                                            end_time_ns=eager_t0 + dur,
                                            input_shape=x.shape,
                                            output_shape=output.shape,
                                            dtype="float32",
                                            batch_size=x.shape[0] if len(x.shape) >= 2 else 1,
                                            is_fused=True,
                                            is_compiled=False,
                                        )
                                    )

                        # 3. Quantized Eager Fallback Path
                        elif self._is_quantized:
                            eager_t0 = time.perf_counter_ns() if is_prof else 0
                            output = self._predict_quantized(x)
                            if is_prof:
                                dur = time.perf_counter_ns() - eager_t0
                                self._profiler.record_backend_op(
                                    backend_dispatch="numpy",
                                    duration_ns=dur,
                                    is_fallback=True,
                                )
                                if self._profiler.is_detailed:
                                    self._profiler.record_event(
                                        ProfileEvent(
                                            name="quantized_eager_forward",
                                            op_type="QuantizedEager",
                                            backend="numpy",
                                            mode="eager",
                                            start_time_ns=eager_t0,
                                            end_time_ns=eager_t0 + dur,
                                            input_shape=x.shape,
                                            output_shape=output.shape,
                                            dtype="int8",
                                            batch_size=x.shape[0] if len(x.shape) >= 2 else 1,
                                            is_fused=False,
                                            is_compiled=False,
                                        )
                                    )

                        # 4. Standard Eager Path
                        else:
                            eager_t0 = time.perf_counter_ns() if is_prof else 0
                            output = self._model(x)
                            if is_prof:
                                dur = time.perf_counter_ns() - eager_t0
                                self._profiler.record_backend_op(
                                    backend_dispatch="numpy",
                                    duration_ns=dur,
                                )
                                if self._profiler.is_detailed:
                                    self._profiler.record_event(
                                        ProfileEvent(
                                            name=f"eager_{type(self._model).__name__}",
                                            op_type=type(self._model).__name__,
                                            backend="numpy",
                                            mode="eager",
                                            start_time_ns=eager_t0,
                                            end_time_ns=eager_t0 + dur,
                                            input_shape=x.shape,
                                            output_shape=output.shape,
                                            dtype="float32",
                                            batch_size=x.shape[0] if len(x.shape) >= 2 else 1,
                                            is_fused=False,
                                            is_compiled=False,
                                        )
                                    )

            if is_prof:
                t1 = time.perf_counter_ns()
                batch_size = x.shape[0] if len(x.shape) >= 2 else 1
                self._profiler.record_prediction(t1 - t0, batch_size=batch_size)

            with self._stats_lock:
                self._prediction_count += 1

            return output.detach()

        except Exception:
            with self._stats_lock:
                self._error_count += 1
            raise

    def predict_batch(
        self,
        batch_data: Union[Tensor, np.ndarray, Sequence[Any]],
    ) -> Tensor:
        """Execute batched inference on multi-sample inputs (alias for predict)."""
        return self.predict(batch_data)

    def _predict_quantized(self, x: Tensor) -> Tensor:
        """Execute quantized INT8 forward inference path (unfused fallback)."""
        current: Tensor = x

        if isinstance(self._model, Sequential):
            for idx, layer in enumerate(self._model):
                w_key = f"{idx}.weight"
                b_key = f"{idx}.bias"

                if w_key in self._state_dict and isinstance(self._state_dict[w_key], QuantizedTensor):
                    w_q = self._state_dict[w_key]
                    w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                    x_q = quantize(current, scheme="symmetric") if not isinstance(current, QuantizedTensor) else current
                    h = qmatmul(x_q, w_q_t)

                    if b_key in self._state_dict:
                        bias_val = self._state_dict[b_key]
                        bias_t = bias_val.dequantize() if isinstance(bias_val, QuantizedTensor) else bias_val
                        h = h + bias_t
                    current = h
                else:
                    current = layer(current)
        elif isinstance(self._model, Linear):
            w_key = "weight"
            b_key = "bias"
            if w_key in self._state_dict and isinstance(self._state_dict[w_key], QuantizedTensor):
                w_q = self._state_dict[w_key]
                w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                x_q = quantize(current, scheme="symmetric") if not isinstance(current, QuantizedTensor) else current
                h = qmatmul(x_q, w_q_t)
                if b_key in self._state_dict:
                    bias_val = self._state_dict[b_key]
                    bias_t = bias_val.dequantize() if isinstance(bias_val, QuantizedTensor) else bias_val
                    h = h + bias_t
                current = h
            else:
                current = self._model(current)
        else:
            current = self._model(current)

        return current

    def health(self) -> Dict[str, Any]:
        """Perform a lightweight operational health check on the inference runtime.

        Returns:
            Dictionary containing health status, concurrency metrics, and error rates.
        """
        is_cls = self.is_closed
        lat = self._profiler.latency_stats()
        return {
            "status": "closed" if is_cls else "healthy",
            "backend": self.backend,
            "is_compiled": self._is_compiled,
            "is_optimized": self._is_optimized,
            "is_quantized": self._is_quantized,
            "num_threads": self._num_threads,
            "profiling_enabled": self._profiler.is_enabled,
            "profiling_mode": self._profiler.mode,
            "active_contexts": self._context_pool.active_count,
            "pooled_contexts": self._context_pool.total_count,
            "idle_contexts": self._context_pool.idle_count,
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "mean_latency_ms": lat.get("mean_ms", 0.0),
            "p95_latency_ms": lat.get("p95_ms", 0.0),
            "throughput_samples_per_sec": lat.get("throughput_samples_per_sec", 0.0),
        }

    def stats(self) -> Dict[str, Any]:
        """Generate an extended diagnostic and statistical report of runtime state."""
        summary = self.summary()
        lat = self.latency_stats()
        summary.update({
            "health": self.health(),
            "active_contexts": self.active_contexts,
            "pooled_contexts": self._context_pool.total_count,
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "profiling_enabled": self._profiler.is_enabled,
            "profiling_mode": self._profiler.mode,
            "total_inference_time_ms": lat.get("total_time_ms", 0.0),
            "mean_latency_ms": lat.get("mean_ms", 0.0),
            "p95_latency_ms": lat.get("p95_ms", 0.0),
            "throughput": lat.get("throughput_samples_per_sec", 0.0),
            "latency": lat,
            "backend_stats": self.backend_stats(),
            "compiler_stats": self.compiler_stats(),
        })
        return summary

    def summary(self) -> Dict[str, Any]:
        """Generate a diagnostic summary of the loaded model and runtime environment.

        Returns:
            Dictionary with architecture details, graph optimization status,
            execution plan details, workspace memory sizes, thread counts, and parameter counts.
        """
        from tensorforge.serialization.checkpoint import compute_model_size

        size_stats = compute_model_size(self._state_dict if self._is_quantized else self._model)
        ws_bytes = self.workspace_size
        mem_plan = self.memory_plan

        return {
            "model_type": type(self._model).__name__,
            "architecture": repr(self._model),
            "status": "closed" if self.is_closed else "active",
            "is_quantized": self._is_quantized,
            "is_optimized": self._is_optimized,
            "is_compiled": self._is_compiled,
            "backend": self.backend,
            "num_threads": self._num_threads,
            "last_dispatch": get_last_backend(),
            "input_shape": self._input_shape,
            "output_shape": self._output_shape,
            "original_nodes": self.original_node_count,
            "optimized_nodes": self.optimized_node_count,
            "compiled_steps": len(self._compiled_plan) if self._compiled_plan is not None else 0,
            "fused_count": self.fused_count,
            "fused_patterns": self.fused_patterns,
            "workspace_bytes": ws_bytes,
            "workspace_kb": ws_bytes / 1024.0,
            "workspace_regions": mem_plan.num_regions if mem_plan is not None else 0,
            "reused_buffers": mem_plan.num_reused_buffers if mem_plan is not None else 0,
            "alignment_padding_bytes": mem_plan.alignment_padding_bytes if mem_plan is not None else 0,
            "num_parameters": size_stats["num_parameters"],
            "total_bytes": size_stats["total_bytes"],
            "size_kb": size_stats["size_kb"],
            "format_version": self._metadata.get("format_version", "1.0"),
            "tensorforge_version": "1.4.0",
        }

    def __repr__(self) -> str:
        status_items = []
        if self.is_closed:
            status_items.append("CLOSED")
        if self._is_optimized:
            status_items.append(f"optimized ({self.fused_count} fused)")
        if self._is_compiled:
            status_items.append(f"compiled ({len(self._compiled_plan or [])} steps, ws={self.workspace_size}B, threads={self._num_threads})")
        if self._profiler.is_enabled:
            status_items.append(f"profiling={self._profiler.mode}")

        status_str = f", {', '.join(status_items)}" if status_items else ""

        return (
            f"InferenceRuntime(\n"
            f"  backend='{self.backend}',\n"
            f"  num_threads={self._num_threads},\n"
            f"  is_quantized={self._is_quantized}{status_str},\n"
            f"  input_shape={self._input_shape},\n"
            f"  output_shape={self._output_shape},\n"
            f"  model={repr(self._model)}\n"
            f")"
        )
