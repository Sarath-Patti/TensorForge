"""Inference graph optimization engine and fused execution pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from tensorforge.backend.dispatcher import get_backend, set_last_backend
from tensorforge.backend.native_backend import (
    can_native_fused_linear,
    can_native_fused_qlinear_relu,
    native_fused_linear,
    native_fused_linear_relu,
    native_fused_linear_sigmoid,
    native_fused_linear_softmax,
    native_fused_linear_tanh,
    native_fused_qlinear_relu,
)
from tensorforge.inference.fusion import OperatorFusionPass
from tensorforge.inference.graph import InferenceGraph, InferenceNode
from tensorforge.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from tensorforge.quantization.quantize import qmatmul, quantize
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.dtype import float32
from tensorforge.tensor.operations import relu, sigmoid, softmax, tanh
from tensorforge.tensor.tensor import Tensor


class GraphOptimizer:
    """Coordinates graph-level optimization passes and provides high-performance fused execution."""

    @classmethod
    def optimize(cls, graph: InferenceGraph) -> Tuple[InferenceGraph, Dict[str, Any]]:
        """Run graph optimization and operator fusion passes.

        Args:
            graph: Original unoptimized InferenceGraph.

        Returns:
            Tuple of (optimized_graph, optimization_statistics).
        """
        return OperatorFusionPass.run(graph)

    @classmethod
    def execute(
        cls,
        graph: InferenceGraph,
        input_tensor: Tensor,
        backend: str = "numpy",
        is_quantized: bool = False,
    ) -> Tensor:
        """Execute an optimized InferenceGraph on an input Tensor.

        Evaluates fused and unfused nodes, prioritizing native C++ execution
        when eligible and safely falling back to NumPy.

        Args:
            graph: InferenceGraph to execute.
            input_tensor: Input tensor.
            backend: Target backend ('numpy' or 'native').
            is_quantized: Whether input or weights are INT8 quantized.

        Returns:
            Output Tensor resulting from the forward pass.
        """
        current: Tensor = input_tensor

        for node in graph:
            if node.op_type == "FusedLinear":
                current = cls._execute_fused_linear(node, current, backend, is_quantized)
            elif node.op_type == "Linear":
                current = cls._execute_linear(node, current, backend, is_quantized)
            elif node.op_type == "ReLU":
                current = relu(current)
            elif node.op_type == "Sigmoid":
                current = sigmoid(current)
            elif node.op_type == "Tanh":
                current = tanh(current)
            elif node.op_type == "Softmax":
                dim = node.attrs.get("dim", -1)
                current = softmax(current, dim=dim)
            else:
                mod = node.attrs.get("module")
                if mod is not None:
                    current = mod(current)

        return current

    @classmethod
    def _execute_fused_linear(
        cls,
        node: InferenceNode,
        x: Tensor,
        backend: str,
        is_quantized: bool,
    ) -> Tensor:
        """Execute a FusedLinear node across Native or NumPy backends."""
        weight = node.params.get("weight")
        bias = node.params.get("bias")
        activation = node.attrs.get("activation", "linear")

        # ---------------------------------------------------------------------
        # 1. INT8 Quantized Fused Execution
        # ---------------------------------------------------------------------
        if is_quantized or isinstance(weight, QuantizedTensor):
            w_q: QuantizedTensor = weight if isinstance(weight, QuantizedTensor) else quantize(weight, scheme="symmetric")
            x_q: QuantizedTensor = x if isinstance(x, QuantizedTensor) else quantize(x, scheme="symmetric")

            # Check for native INT8 Fused Linear + ReLU
            if (
                backend == "native"
                and activation == "relu"
                and can_native_fused_qlinear_relu(x_q.ndim, x_q.shape, w_q.ndim, w_q.shape)
            ):
                bias_arr = bias.numpy() if isinstance(bias, Tensor) else (bias if bias is not None else None)
                out_arr = native_fused_qlinear_relu(
                    x_q.numpy(),
                    w_q.numpy(),
                    bias_arr,
                    float(x_q.scale),
                    int(x_q.zero_point),
                    float(w_q.scale),
                    int(w_q.zero_point),
                )
                set_last_backend("native (fused)")
                return Tensor(out_arr, dtype=float32)
            else:
                # Fused INT8 NumPy Path
                w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
                h = qmatmul(x_q, w_q_t)
                if bias is not None:
                    bias_t = bias.dequantize() if isinstance(bias, QuantizedTensor) else bias
                    h = h + bias_t

                set_last_backend("numpy (fused)" if backend != "native" else "native (fused fallback)")
                if activation == "relu":
                    return relu(h)
                elif activation == "sigmoid":
                    return sigmoid(h)
                elif activation == "tanh":
                    return tanh(h)
                elif activation == "softmax":
                    return softmax(h, dim=node.attrs.get("dim", -1))
                return h

        # ---------------------------------------------------------------------
        # 2. FP32 Fused Execution
        # ---------------------------------------------------------------------
        w_arr = weight.numpy() if isinstance(weight, Tensor) else np.asarray(weight, dtype=np.float32)
        b_arr = bias.numpy() if isinstance(bias, Tensor) else (np.asarray(bias, dtype=np.float32) if bias is not None else None)
        x_arr = x.numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float32)

        # Native FP32 Fused Path
        if backend == "native" and can_native_fused_linear(x.dtype, x.ndim, x.shape, w_arr.shape):
            if activation == "relu":
                out_arr = native_fused_linear_relu(x_arr, w_arr, b_arr)
            elif activation == "sigmoid":
                out_arr = native_fused_linear_sigmoid(x_arr, w_arr, b_arr)
            elif activation == "tanh":
                out_arr = native_fused_linear_tanh(x_arr, w_arr, b_arr)
            elif activation == "softmax":
                dim = node.attrs.get("dim", -1)
                out_arr = native_fused_linear_softmax(x_arr, w_arr, b_arr, dim=dim)
            else:
                out_arr = native_fused_linear(x_arr, w_arr, b_arr)

            set_last_backend("native (fused)")
            return Tensor(out_arr, dtype=float32)

        # NumPy FP32 Fused Reference Path
        h = x @ (weight.T if isinstance(weight, Tensor) else Tensor(w_arr.T, dtype=float32))
        if bias is not None:
            h = h + (bias if isinstance(bias, Tensor) else Tensor(b_arr, dtype=float32))

        set_last_backend("numpy (fused)" if backend != "native" else "native (fused fallback)")
        if activation == "relu":
            return relu(h)
        elif activation == "sigmoid":
            return sigmoid(h)
        elif activation == "tanh":
            return tanh(h)
        elif activation == "softmax":
            return softmax(h, dim=node.attrs.get("dim", -1))
        return h

    @classmethod
    def _execute_linear(
        cls,
        node: InferenceNode,
        x: Tensor,
        backend: str,
        is_quantized: bool,
    ) -> Tensor:
        """Execute an unfused Linear node."""
        weight = node.params.get("weight")
        bias = node.params.get("bias")

        if is_quantized or isinstance(weight, QuantizedTensor):
            w_q: QuantizedTensor = weight if isinstance(weight, QuantizedTensor) else quantize(weight, scheme="symmetric")
            x_q: QuantizedTensor = x if isinstance(x, QuantizedTensor) else quantize(x, scheme="symmetric")
            w_q_t = quantize(w_q.dequantize().transpose(), scheme="symmetric")
            h = qmatmul(x_q, w_q_t)
            if bias is not None:
                bias_t = bias.dequantize() if isinstance(bias, QuantizedTensor) else bias
                h = h + bias_t
            return h
        else:
            w = weight if isinstance(weight, Tensor) else Tensor(weight, dtype=float32)
            h = x @ w.T
            if bias is not None:
                b = bias if isinstance(bias, Tensor) else Tensor(bias, dtype=float32)
                h = h + b
            return h
