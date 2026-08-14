"""Reverse-mode automatic differentiation execution engine for TensorForge.

Traverses the dynamic computation graph in reverse topological order, propagates
and accumulates gradients, and updates leaf tensor .grad attributes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple
from contextlib import ContextDecorator

if TYPE_CHECKING:
    from tensorforge.autograd.function import Node
    from tensorforge.tensor.tensor import Tensor


_GRAD_ENABLED: bool = True


def is_grad_enabled() -> bool:
    """Check if autograd gradient recording is currently active."""
    return _GRAD_ENABLED


class no_grad(ContextDecorator):
    """Context-manager and decorator that disables autograd gradient recording.

    Example:
        >>> with tf.no_grad():
        ...     y = x * 2  # y will have requires_grad=False and grad_fn=None
    """

    def __init__(self) -> None:
        self.prev: bool = True

    def __enter__(self) -> None:
        global _GRAD_ENABLED
        self.prev = _GRAD_ENABLED
        _GRAD_ENABLED = False

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self.prev


def _build_topological_order(root_node: Node) -> List[Node]:
    """Construct an iterative reverse topological ordering of DAG nodes starting from root.

    Uses a non-recursive post-order depth-first traversal to avoid Python call-stack
    recursion limits on deep graphs.

    Args:
        root_node: The starting root Node (e.g. loss.grad_fn).

    Returns:
        List of Nodes ordered from output root down toward leaf inputs.
    """
    order: list[Node] = []
    visited: set[Node] = set()

    # Stack holds (node, processed_flag)
    stack: list[tuple[Node, bool]] = [(root_node, False)]

    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue

        if node in visited:
            continue

        visited.add(node)
        stack.append((node, True))

        for parent_tensor in node.parents:
            p_node = getattr(parent_tensor, "grad_fn", None)
            if p_node is not None and p_node not in visited:
                stack.append((p_node, False))

    # Reversing post-order yields topological order (root -> inputs)
    return order[::-1]


def backward(
    root_tensor: Tensor,
    root_gradient: Optional[Tensor] = None,
    retain_graph: bool = False,
) -> None:
    """Compute gradients of root_tensor with respect to all leaf tensors in the graph.

    Args:
        root_tensor: Output tensor from which backpropagation begins.
        root_gradient: Initial gradient flowing into root_tensor. Required for non-scalar outputs.
        retain_graph: If False, graph references can be released after backward pass.

    Raises:
        RuntimeError: If called on a non-scalar without an explicit gradient, or on a tensor
            that does not require gradients.
        ValueError: If root_gradient shape does not match root_tensor shape.
    """
    from tensorforge.tensor.tensor import Tensor, ones, tensor

    if not root_tensor.requires_grad and root_tensor.grad_fn is None:
        raise RuntimeError("Cannot call backward on a tensor that does not require gradients.")

    # 1. Resolve initial root gradient
    if root_gradient is None:
        if root_tensor.numel != 1:
            raise RuntimeError(
                f"grad can be implicitly created only for scalar outputs (numel=1), "
                f"but tensor has shape {root_tensor.shape} (numel={root_tensor.numel}). "
                f"Provide an explicit gradient tensor: tensor.backward(gradient=...)"
            )
        current_root_grad = ones((), dtype=root_tensor.dtype)
    else:
        if not isinstance(root_gradient, Tensor):
            current_root_grad = tensor(root_gradient, dtype=root_tensor.dtype)
        else:
            current_root_grad = root_gradient

        if current_root_grad.shape != root_tensor.shape:
            raise ValueError(
                f"Initial gradient shape mismatch: expected {root_tensor.shape}, "
                f"got {current_root_grad.shape}"
            )

    # 2. Handle leaf root tensor edge case
    if root_tensor.is_leaf:
        if root_tensor.grad is None:
            root_tensor.grad = current_root_grad.clone()
        else:
            root_tensor.grad = root_tensor.grad + current_root_grad
        return

    root_node = root_tensor.grad_fn
    if root_node is None:
        return

    # 3. Build topological execution order
    topo_order = _build_topological_order(root_node)

    # 4. Map of accumulated gradient outputs for each Node
    grad_map: Dict[Node, Tensor] = {root_node: current_root_grad}

    # 5. Reverse propagation loop
    for node in topo_order:
        grad_out = grad_map.get(node)
        if grad_out is None:
            continue

        input_grads = node.backward(grad_out)

        for parent_tensor, in_grad, needs_grad in zip(node.parents, input_grads, node.needs_input_grad):
            if not needs_grad or in_grad is None:
                continue

            # Accumulate on leaf tensor
            if parent_tensor.is_leaf:
                if parent_tensor.grad is None:
                    parent_tensor.grad = in_grad.clone()
                else:
                    parent_tensor.grad = parent_tensor.grad + in_grad

            # Accumulate on parent grad_fn node
            p_node = getattr(parent_tensor, "grad_fn", None)
            if p_node is not None:
                if p_node not in grad_map:
                    grad_map[p_node] = in_grad.clone()
                else:
                    grad_map[p_node] = grad_map[p_node] + in_grad
