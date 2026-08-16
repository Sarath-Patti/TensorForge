"""Inference computation graph representation for TensorForge."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from tensorforge.nn.activations import ReLU, Sigmoid, Softmax, Tanh
from tensorforge.nn.linear import Linear
from tensorforge.nn.module import Module
from tensorforge.nn.sequential import Sequential
from tensorforge.quantization.quantized_tensor import QuantizedTensor
from tensorforge.tensor.tensor import Tensor


class InferenceNode:
    """Represents a single executable node in an inference computation graph."""

    def __init__(
        self,
        name: str,
        op_type: str,
        params: Optional[Dict[str, Any]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name: str = name
        self.op_type: str = op_type
        self.params: Dict[str, Any] = params or {}
        self.attrs: Dict[str, Any] = attrs or {}

    @property
    def is_fused(self) -> bool:
        """Whether this node represents a fused multi-operator kernel."""
        return self.op_type.startswith("Fused")

    def __repr__(self) -> str:
        param_desc = ", ".join(f"{k}: shape={v.shape}" for k, v in self.params.items() if hasattr(v, "shape"))
        attr_desc = ", ".join(f"{k}={v}" for k, v in self.attrs.items())
        extra = f" [{param_desc}]" if param_desc else ""
        if attr_desc:
            extra += f" ({attr_desc})"
        return f"InferenceNode(name='{self.name}', op='{self.op_type}'{extra})"


class InferenceGraph:
    """Sequential/DAG container of inference nodes representing a deployable neural network."""

    def __init__(self, nodes: Optional[List[InferenceNode]] = None) -> None:
        self.nodes: List[InferenceNode] = list(nodes) if nodes is not None else []

    def add_node(self, node: InferenceNode) -> None:
        """Append an inference node to the graph."""
        self.nodes.append(node)

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> InferenceNode:
        return self.nodes[idx]

    def __iter__(self):
        return iter(self.nodes)

    @classmethod
    def from_module(
        cls,
        model: Module,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> InferenceGraph:
        """Construct an InferenceGraph from a TensorForge Module instance."""
        graph = cls()

        if isinstance(model, Sequential):
            for idx, child in enumerate(model):
                node_name = f"layer_{idx}"
                cls._extract_module_node(child, node_name, graph, state_dict, prefix=str(idx))
        else:
            cls._extract_module_node(model, "layer_0", graph, state_dict, prefix="")

        return graph

    @classmethod
    def _extract_module_node(
        cls,
        module: Module,
        name: str,
        graph: InferenceGraph,
        state_dict: Optional[Dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        """Extract a single module into an InferenceNode and append to graph."""
        if isinstance(module, Linear):
            params: Dict[str, Any] = {}
            w_key = f"{prefix}.weight" if prefix else "weight"
            b_key = f"{prefix}.bias" if prefix else "bias"

            if state_dict and w_key in state_dict:
                params["weight"] = state_dict[w_key]
            else:
                params["weight"] = module.weight

            if state_dict and b_key in state_dict:
                params["bias"] = state_dict[b_key]
            elif module.bias is not None:
                params["bias"] = module.bias
            else:
                params["bias"] = None

            attrs = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": params["bias"] is not None,
            }
            graph.add_node(InferenceNode(name=name, op_type="Linear", params=params, attrs=attrs))

        elif isinstance(module, ReLU):
            graph.add_node(InferenceNode(name=name, op_type="ReLU"))

        elif isinstance(module, Sigmoid):
            graph.add_node(InferenceNode(name=name, op_type="Sigmoid"))

        elif isinstance(module, Tanh):
            graph.add_node(InferenceNode(name=name, op_type="Tanh"))

        elif isinstance(module, Softmax):
            graph.add_node(InferenceNode(name=name, op_type="Softmax", attrs={"dim": module.dim}))

        else:
            # Generic fallback node
            graph.add_node(InferenceNode(name=name, op_type=type(module).__name__, attrs={"module": module}))

    def summary(self) -> str:
        """Format human-readable summary of nodes in the graph."""
        lines = [f"InferenceGraph (Total Nodes: {len(self.nodes)}):"]
        for idx, node in enumerate(self.nodes):
            lines.append(f"  [{idx}] {node}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
