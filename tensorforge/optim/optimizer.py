"""Base Optimizer abstraction for TensorForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import numpy as np

from tensorforge.tensor.tensor import Tensor


class Optimizer(ABC):
    """Base class for all neural network parameter optimizers in TensorForge.

    Args:
        params: An iterable of :class:`~tensorforge.Tensor` or :class:`~tensorforge.nn.Parameter`
            instances to optimize.
        defaults: Default optimization hyperparameters.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        defaults: Dict[str, Any],
    ) -> None:
        if isinstance(params, Tensor):
            raise TypeError("params argument given to the optimizer should be an iterable of Tensors")

        param_list = list(params)
        if len(param_list) == 0:
            raise ValueError("optimizer got an empty parameter list")

        # Validate elements are Tensors
        for p in param_list:
            if not isinstance(p, Tensor):
                raise TypeError(f"optimizer can only optimize Tensors, but got {type(p).__name__}")

        self.defaults: Dict[str, Any] = defaults
        self.state: Dict[Tensor, Dict[str, Any]] = {}
        self.param_groups: List[Dict[str, Any]] = [{"params": param_list, **defaults}]

    def zero_grad(self) -> None:
        """Reset the gradients of all optimized parameters to None."""
        for group in self.param_groups:
            for p in group["params"]:
                p.zero_grad()

    @abstractmethod
    def step(self) -> None:
        """Perform a single optimization step (parameter update)."""
        raise NotImplementedError("Optimizer subclasses must implement step()")

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " (\n"
        for i, group in enumerate(self.param_groups):
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string
