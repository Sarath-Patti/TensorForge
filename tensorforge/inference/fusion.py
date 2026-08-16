"""Operator fusion passes for optimizing inference computation graphs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from tensorforge.inference.graph import InferenceGraph, InferenceNode


class OperatorFusionPass:
    """Graph transformation pass that identifies and collapses fusible operator sequences."""

    SUPPORTED_FUSIONS = {
        "ReLU": "relu",
        "Sigmoid": "sigmoid",
        "Tanh": "tanh",
        "Softmax": "softmax",
    }

    @classmethod
    def run(cls, graph: InferenceGraph) -> Tuple[InferenceGraph, Dict[str, Any]]:
        """Apply operator fusion transformations on an input InferenceGraph.

        Fuses supported patterns:
            - Linear + ReLU -> FusedLinear(activation='relu')
            - Linear + Sigmoid -> FusedLinear(activation='sigmoid')
            - Linear + Tanh -> FusedLinear(activation='tanh')
            - Linear + Softmax -> FusedLinear(activation='softmax', dim=...)

        Args:
            graph: Original unoptimized InferenceGraph.

        Returns:
            Tuple of (optimized_graph, fusion_statistics).
        """
        optimized_nodes: List[InferenceNode] = []
        fused_patterns: List[str] = []
        i = 0
        n = len(graph)

        while i < n:
            current_node = graph[i]

            # Check for Linear + Activation sequence
            if current_node.op_type == "Linear" and (i + 1) < n:
                next_node = graph[i + 1]
                if next_node.op_type in cls.SUPPORTED_FUSIONS:
                    act_name = cls.SUPPORTED_FUSIONS[next_node.op_type]
                    fused_name = f"{current_node.name}_{next_node.name}_fused"
                    fused_attrs = dict(current_node.attrs)
                    fused_attrs["activation"] = act_name
                    if "dim" in next_node.attrs:
                        fused_attrs["dim"] = next_node.attrs["dim"]

                    fused_node = InferenceNode(
                        name=fused_name,
                        op_type="FusedLinear",
                        params=dict(current_node.params),
                        attrs=fused_attrs,
                    )
                    optimized_nodes.append(fused_node)
                    fused_patterns.append(f"Linear+{next_node.op_type}")
                    i += 2
                    continue

            optimized_nodes.append(current_node)
            i += 1

        stats = {
            "original_nodes": n,
            "optimized_nodes": len(optimized_nodes),
            "fused_count": len(fused_patterns),
            "fused_patterns": fused_patterns,
        }

        return InferenceGraph(optimized_nodes), stats
