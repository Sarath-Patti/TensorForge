"""Inference Compiler for transforming InferenceGraphs into compiled ExecutionPlans."""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from tensorforge.backend.dispatcher import get_backend, get_num_threads, set_last_backend, set_num_threads
from tensorforge.backend.native_backend import (
    can_native_elementwise,
    can_native_fused_linear,
    can_native_fused_qlinear_relu,
    can_native_matmul,
    native_add,
    native_fused_linear,
    native_fused_linear_relu,
    native_fused_linear_sigmoid,
    native_fused_linear_softmax,
    native_fused_linear_tanh,
    native_fused_qlinear_relu,
    native_matmul,
    native_mul,
    native_sub,
)
from tensorforge.inference.context import ExecutionContext
from tensorforge.inference.graph import InferenceGraph, InferenceNode
from tensorforge.inference.memory import MemoryPlanner
from tensorforge.inference.plan import ExecutionPlan, ExecutionStep
from tensorforge.inference.shapes import ShapePropagator
from tensorforge.quantization.quantize import qmatmul, quantize
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import DType, float32, to_dtype
from tensorforge.tensor.operations import relu, sigmoid, softmax, tanh
from tensorforge.tensor.tensor import Tensor


class CompiledPlanCache:
    """Thread-safe in-memory cache for compiled ExecutionPlan instances to prevent redundant compilation."""

    def __init__(self) -> None:
        self._cache: Dict[Tuple[Any, ...], ExecutionPlan] = {}
        self._lock: threading.Lock = threading.Lock()

    def get(
        self,
        graph_id: int,
        input_shape: Tuple[int, ...],
        dtype: DType,
        backend: str,
        is_quantized: bool,
        num_threads: int = 1,
    ) -> Optional[ExecutionPlan]:
        key = (graph_id, tuple(input_shape), dtype.name, backend, is_quantized, num_threads)
        with self._lock:
            return self._cache.get(key)

    def put(
        self,
        graph_id: int,
        input_shape: Tuple[int, ...],
        dtype: DType,
        backend: str,
        is_quantized: bool,
        plan: ExecutionPlan,
        num_threads: int = 1,
    ) -> None:
        key = (graph_id, tuple(input_shape), dtype.name, backend, is_quantized, num_threads)
        with self._lock:
            self._cache[key] = plan

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


class InferenceCompiler:
    """Compiles optimized InferenceGraph models into deterministic, memory-planned ExecutionPlans."""

    _global_cache: CompiledPlanCache = CompiledPlanCache()

    @classmethod
    def compile(
        cls,
        graph: InferenceGraph,
        input_shape: Tuple[int, ...],
        backend: Optional[str] = None,
        dtype: Union[DType, str, np.dtype, type] = float32,
        is_quantized: bool = False,
        num_threads: Optional[int] = None,
        use_cache: bool = True,
        profiler: Optional[Any] = None,
    ) -> ExecutionPlan:
        """Compile an InferenceGraph for a specified input shape, backend, and CPU thread count.

        Args:
            graph: Optimized InferenceGraph.
            input_shape: Shape of the input tensor (e.g. (batch_size, in_features)).
            backend: Target backend ('numpy' or 'native'). Defaults to active backend.
            dtype: Floating-point precision (default: float32).
            is_quantized: Whether graph operates in INT8 quantization mode.
            num_threads: Number of CPU threads to configure for execution.
            use_cache: Whether to retrieve or store compiled plans in the plan cache.
            profiler: Optional RuntimeProfiler to record compilation performance events.

        Returns:
            Pre-resolved, deterministic ExecutionPlan.
        """
        start_ns = time.perf_counter_ns()
        resolved_dtype = to_dtype(dtype)
        target_backend = backend if backend is not None else get_backend()
        target_threads = num_threads if num_threads is not None else get_num_threads()
        graph_id = id(graph)

        if use_cache:
            cached_plan = cls._global_cache.get(
                graph_id=graph_id,
                input_shape=input_shape,
                dtype=resolved_dtype,
                backend=target_backend,
                is_quantized=is_quantized,
                num_threads=target_threads,
            )
            if cached_plan is not None:
                if profiler is not None:
                    profiler.record_compiler_event(cache_hit=True, cached_plans=len(cls._global_cache))
                return cached_plan

        # 1. Static Shape Propagation
        shape_flow = ShapePropagator.propagate(graph, input_shape)

        # 2. Workspace Memory Planning
        workspace_plan = MemoryPlanner.plan_sequential_workspace(shape_flow, dtype=resolved_dtype)

        # 3. Instruction, Parallelism & Backend Resolution
        steps: List[ExecutionStep] = []

        for i, node in enumerate(graph):
            in_shape, out_shape = shape_flow[i]
            in_slot = workspace_plan["step_input_slots"][i]
            out_slot = workspace_plan["step_output_slots"][i]
            node_is_quantized = bool(is_quantized or any(isinstance(p, QuantizedTensor) for p in node.params.values()))

            backend_dispatch = cls._resolve_backend_dispatch(
                node=node,
                in_shape=in_shape,
                out_shape=out_shape,
                target_backend=target_backend,
                dtype=resolved_dtype,
                is_quantized=node_is_quantized,
            )

            # Compute estimated work (FLOPs) and parallel eligibility
            estimated_flops = cls._estimate_step_flops(node, in_shape, out_shape)
            is_parallelizable = (
                target_backend == "native"
                and target_threads > 1
                and estimated_flops >= 8192
                and in_shape[0] > 1
            )

            step = ExecutionStep(
                step_index=i,
                op_type=node.op_type,
                input_slot=in_slot,
                output_slot=out_slot,
                input_shape=in_shape,
                output_shape=out_shape,
                backend_dispatch=backend_dispatch,
                params=node.params,
                attrs=node.attrs,
                dtype=resolved_dtype,
                is_quantized=node_is_quantized,
                is_parallelizable=is_parallelizable,
                estimated_flops=estimated_flops,
                num_threads=target_threads if is_parallelizable else 1,
            )
            steps.append(step)

        output_shape = shape_flow[-1][1] if shape_flow else tuple(input_shape)

        plan = ExecutionPlan(
            steps=steps,
            input_shape=input_shape,
            output_shape=output_shape,
            workspace_plan=workspace_plan,
            target_backend=target_backend,
            dtype=resolved_dtype,
            is_quantized=is_quantized,
            num_threads=target_threads,
        )

        if use_cache:
            cls._global_cache.put(
                graph_id=graph_id,
                input_shape=input_shape,
                dtype=resolved_dtype,
                backend=target_backend,
                is_quantized=is_quantized,
                plan=plan,
                num_threads=target_threads,
            )

        if profiler is not None:
            compile_time_ns = time.perf_counter_ns() - start_ns
            profiler.record_compiler_event(
                cache_hit=False,
                compilation_time_ns=compile_time_ns,
                cached_plans=len(cls._global_cache),
            )

        return plan

    @classmethod
    def _estimate_step_flops(cls, node: InferenceNode, in_shape: Tuple[int, ...], out_shape: Tuple[int, ...]) -> int:
        """Estimate the computational workload in FLOPs for an inference step."""
        op_type = node.op_type
        if op_type in ("Linear", "FusedLinear"):
            m = in_shape[0] if len(in_shape) >= 2 else 1
            k = in_shape[-1]
            n = out_shape[-1]
            return 2 * int(m) * int(n) * int(k)
        return int(np.prod(out_shape))

    @classmethod
    def _resolve_backend_dispatch(
        cls,
        node: InferenceNode,
        in_shape: Tuple[int, ...],
        out_shape: Tuple[int, ...],
        target_backend: str,
        dtype: DType,
        is_quantized: bool,
    ) -> str:
        """Resolve the optimal execution kernel for a node at compile time."""
        if target_backend != "native":
            return "numpy_fused" if node.op_type == "FusedLinear" else "numpy"

        op_type = node.op_type
        weight = node.params.get("weight")

        if op_type == "FusedLinear":
            if is_quantized or isinstance(weight, QuantizedTensor):
                w_shape = weight.shape if weight is not None else ()
                if (
                    node.attrs.get("activation") == "relu"
                    and can_native_fused_qlinear_relu(len(in_shape), in_shape, len(w_shape), w_shape)
                ):
                    return "native_fused"
                return "numpy_fused"
            else:
                w_shape = weight.shape if weight is not None else ()
                if can_native_fused_linear(dtype, len(in_shape), in_shape, w_shape):
                    return "native_fused"
                return "numpy_fused"

        elif op_type == "Linear":
            if is_quantized or isinstance(weight, QuantizedTensor):
                return "numpy"
            else:
                w_shape = weight.shape if weight is not None else ()
                w_t_shape = (w_shape[1], w_shape[0]) if len(w_shape) == 2 else w_shape
                if can_native_matmul(dtype, len(in_shape), in_shape, dtype, len(w_t_shape), w_t_shape):
                    return "native"
                return "numpy"

        return "numpy"

    @classmethod
    def execute_plan(
        cls,
        plan: ExecutionPlan,
        input_tensor: Tensor,
        context: Optional[ExecutionContext] = None,
        profiler: Optional[Any] = None,
    ) -> Tensor:
        """Execute a compiled ExecutionPlan against an input tensor using isolated workspace slots.

        When an `ExecutionContext` is provided, intermediate activation buffers are allocated
        and stored strictly within `context.slots`, preserving memory safety and thread-level
        concurrency isolation across threads.

        Args:
            plan: Pre-compiled ExecutionPlan.
            input_tensor: Input Tensor matching plan input shape.
            context: Per-prediction ExecutionContext holding isolated workspace slots.
            profiler: Optional RuntimeProfiler to record per-step execution telemetry.

        Returns:
            Output Tensor resulting from executing all compiled steps.
        """
        is_detailed = profiler is not None and getattr(profiler, "is_detailed", False)
        # Workspace memory slots (isolated per-context or local dictionary)
        slots: Dict[int, Tensor] = context.slots if context is not None else {}

        for step in plan.steps:
            # 1. Fetch Input Tensor
            if step.input_slot < 0:
                step_in = input_tensor
            else:
                step_in = slots[step.input_slot]

            # 2. Execute Step Kernel
            if is_detailed:
                t0 = time.perf_counter_ns()
                step_out = cls._execute_step(step, step_in)
                t1 = time.perf_counter_ns()
                duration_ns = t1 - t0

                from tensorforge.inference.profiler import ProfileEvent
                event = ProfileEvent(
                    name=f"step_{step.step_index}_{step.op_type}",
                    op_type=step.op_type,
                    backend=step.backend_dispatch,
                    mode="compiled",
                    start_time_ns=t0,
                    end_time_ns=t1,
                    input_shape=step.input_shape,
                    output_shape=step.output_shape,
                    dtype=step.dtype.name if hasattr(step.dtype, "name") else str(step.dtype),
                    batch_size=step.input_shape[0] if step.input_shape else 1,
                    estimated_flops=step.estimated_flops,
                    workspace_bytes=plan.total_workspace_bytes,
                    num_threads=step.num_threads,
                    is_fused=(step.op_type == "FusedLinear"),
                    is_compiled=True,
                    context_id=context.context_id if context is not None else 0,
                    extra=step.attrs,
                )
                profiler.record_event(event)
                profiler.record_backend_op(
                    backend_dispatch=step.backend_dispatch,
                    duration_ns=duration_ns,
                    is_fused=(step.op_type == "FusedLinear"),
                )
            else:
                step_out = cls._execute_step(step, step_in)

            # 3. Store into Output Slot
            slots[step.output_slot] = step_out

        final_step = plan.steps[-1] if plan.steps else None
        if final_step is not None:
            return slots[final_step.output_slot]
        return input_tensor

    @classmethod
    def _execute_step(cls, step: ExecutionStep, x: Tensor) -> Tensor:
        """Execute a single compiled step according to its pre-resolved backend."""
        op_type = step.op_type
        dispatch = step.backend_dispatch
        weight = step.params.get("weight")
        bias = step.params.get("bias")
        attrs = step.attrs

        # ---------------------------------------------------------------------
        # 1. FusedLinear Execution
        # ---------------------------------------------------------------------
        if op_type == "FusedLinear":
            activation = attrs.get("activation", "linear")

            if dispatch == "native_fused":
                w_arr = weight.numpy() if isinstance(weight, (Tensor, QuantizedTensor)) else np.asarray(weight)
                b_arr = bias.numpy() if isinstance(bias, (Tensor, QuantizedTensor)) else (np.asarray(bias) if bias is not None else None)
                x_arr = x.numpy() if isinstance(x, (Tensor, QuantizedTensor)) else np.asarray(x)

                if step.is_quantized or isinstance(weight, QuantizedTensor):
                    w_q: QuantizedTensor = weight if isinstance(weight, QuantizedTensor) else quantize(weight, scheme="symmetric")
                    x_q: QuantizedTensor = x if isinstance(x, QuantizedTensor) else quantize(x, scheme="symmetric")
                    out_arr = native_fused_qlinear_relu(
                        x_q.numpy(),
                        w_q.numpy(),
                        b_arr,
                        float(x_q.scale),
                        int(x_q.zero_point),
                        float(w_q.scale),
                        int(w_q.zero_point),
                    )
                else:
                    if activation == "relu":
                        out_arr = native_fused_linear_relu(x_arr, w_arr, b_arr)
                    elif activation == "sigmoid":
                        out_arr = native_fused_linear_sigmoid(x_arr, w_arr, b_arr)
                    elif activation == "tanh":
                        out_arr = native_fused_linear_tanh(x_arr, w_arr, b_arr)
                    elif activation == "softmax":
                        dim = attrs.get("dim", -1)
                        out_arr = native_fused_linear_softmax(x_arr, w_arr, b_arr, dim=dim)
                    else:
                        out_arr = native_fused_linear(x_arr, w_arr, b_arr)

                disp_name = "native (parallel fused)" if step.is_parallelizable else "native (compiled fused)"
                set_last_backend(disp_name)
                return Tensor(out_arr, dtype=float32)

            else:
                # Fused NumPy Path
                if step.is_quantized or isinstance(weight, QuantizedTensor):
                    w_q: QuantizedTensor = weight if isinstance(weight, QuantizedTensor) else quantize(weight, scheme="symmetric")
                    x_q: QuantizedTensor = x if isinstance(x, QuantizedTensor) else quantize(x, scheme="symmetric")
                    w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                    h = qmatmul(x_q, w_q_t)
                    if bias is not None:
                        bias_t = bias.dequantize() if isinstance(bias, QuantizedTensor) else bias
                        h = h + bias_t
                else:
                    w = weight if isinstance(weight, Tensor) else Tensor(weight, dtype=float32)
                    h = x @ w.T
                    if bias is not None:
                        b = bias if isinstance(bias, Tensor) else Tensor(bias, dtype=float32)
                        h = h + b

                set_last_backend("numpy (compiled fused)")
                if activation == "relu":
                    return relu(h)
                elif activation == "sigmoid":
                    return sigmoid(h)
                elif activation == "tanh":
                    return tanh(h)
                elif activation == "softmax":
                    return softmax(h, dim=attrs.get("dim", -1))
                return h

        # ---------------------------------------------------------------------
        # 2. Linear Execution
        # ---------------------------------------------------------------------
        elif op_type == "Linear":
            if dispatch == "native":
                w_arr = weight.numpy() if isinstance(weight, Tensor) else np.asarray(weight, dtype=np.float32)
                b_arr = bias.numpy() if isinstance(bias, Tensor) else (np.asarray(bias, dtype=np.float32) if bias is not None else None)
                x_arr = x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)
                out_arr = native_fused_linear(x_arr, w_arr, b_arr)
                disp_name = "native (parallel)" if step.is_parallelizable else "native (compiled)"
                set_last_backend(disp_name)
                return Tensor(out_arr, dtype=float32)
            else:
                if step.is_quantized or isinstance(weight, QuantizedTensor):
                    w_q: QuantizedTensor = weight if isinstance(weight, QuantizedTensor) else quantize(weight, scheme="symmetric")
                    x_q: QuantizedTensor = x if isinstance(x, QuantizedTensor) else quantize(x, scheme="symmetric")
                    w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                    h = qmatmul(x_q, w_q_t)
                    if bias is not None:
                        bias_t = bias.dequantize() if isinstance(bias, QuantizedTensor) else bias
                        h = h + bias_t
                else:
                    w = weight if isinstance(weight, Tensor) else Tensor(weight, dtype=float32)
                    h = x @ w.T
                    if bias is not None:
                        b = bias if isinstance(bias, Tensor) else Tensor(bias, dtype=float32)
                        h = h + b
                set_last_backend("numpy (compiled)")
                return h

        # ---------------------------------------------------------------------
        # 3. Activations
        # ---------------------------------------------------------------------
        elif op_type == "ReLU":
            return relu(x)
        elif op_type == "Sigmoid":
            return sigmoid(x)
        elif op_type == "Tanh":
            return tanh(x)
        elif op_type == "Softmax":
            dim = attrs.get("dim", -1)
            return softmax(x, dim=dim)
        else:
            mod = attrs.get("module")
            if mod is not None:
                return mod(x)
            return x
