"""Static shape propagation engine for TensorForge inference graphs."""

from __future__ import annotations

from typing import List, Tuple
from tensorforge.inference.graph import InferenceGraph, InferenceNode
from tensorforge.utils.validation import ShapeError


class ShapePropagator:
    """Statically infers and validates tensor shapes across inference graph nodes without evaluation."""

    @classmethod
    def infer_node_shape(
        cls,
        node: InferenceNode,
        input_shape: Tuple[int, ...],
    ) -> Tuple[int, ...]:
        """Infer the output shape produced by an InferenceNode given its input shape.

        Args:
            node: Target InferenceNode.
            input_shape: Shape tuple of the incoming tensor.

        Returns:
            Computed output shape tuple.

        Raises:
            ShapeError: If the input shape is incompatible with node specifications.
        """
        if len(input_shape) == 0:
            raise ShapeError(f"Node '{node.name}' ({node.op_type}) received an invalid 0-dimensional input shape.")

        op_type = node.op_type

        if op_type in ("Linear", "FusedLinear"):
            in_features = node.attrs.get("in_features")
            out_features = node.attrs.get("out_features")

            if in_features is None or out_features is None:
                # Fallback to weight tensor shape inspection if attributes are absent
                weight = node.params.get("weight")
                if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                    out_features, in_features = weight.shape
                else:
                    raise ShapeError(f"Node '{node.name}' ({op_type}) is missing in_features/out_features metadata.")

            if input_shape[-1] != in_features:
                raise ShapeError(
                    f"Shape mismatch for node '{node.name}' ({op_type}): input has {input_shape[-1]} "
                    f"features, but operator expected {in_features} features (input shape: {input_shape})."
                )

            # Preserve batch dimensions: (..., in_features) -> (..., out_features)
            return (*input_shape[:-1], int(out_features))

        elif op_type in ("ReLU", "Sigmoid", "Tanh"):
            # Element-wise operations preserve exact input shape
            return tuple(input_shape)

        elif op_type == "Softmax":
            dim = node.attrs.get("dim", -1)
            ndim = len(input_shape)
            if dim < -ndim or dim >= ndim:
                raise ShapeError(f"Softmax dim {dim} out of range for tensor with ndim {ndim} (shape: {input_shape}).")
            return tuple(input_shape)

        else:
            # Generic operator fallback
            return tuple(input_shape)

    @classmethod
    def propagate(
        cls,
        graph: InferenceGraph,
        input_shape: Tuple[int, ...],
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Propagate shapes through every node in an InferenceGraph.

        Args:
            graph: InferenceGraph to propagate shapes through.
            input_shape: Initial model input shape.

        Returns:
            List of (node_input_shape, node_output_shape) pairs matching graph nodes.
        """
        shape_flow: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        current_shape = tuple(input_shape)

        for node in graph:
            out_shape = cls.infer_node_shape(node, current_shape)
            shape_flow.append((current_shape, out_shape))
            current_shape = out_shape

        return shape_flow
