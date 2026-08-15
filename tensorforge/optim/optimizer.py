"""Base Optimizer abstraction for TensorForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union
import numpy as np

from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import SerializationError


class _ParamStateDict(dict):
    """Dictionary mapping parameters to optimizer state using object identity."""

    def __getitem__(self, key: Any) -> Dict[str, Any]:
        k = id(key) if isinstance(key, Tensor) else key
        return super().__getitem__(k)

    def __setitem__(self, key: Any, value: Dict[str, Any]) -> None:
        k = id(key) if isinstance(key, Tensor) else key
        super().__setitem__(k, value)

    def __contains__(self, key: Any) -> bool:
        k = id(key) if isinstance(key, Tensor) else key
        return super().__contains__(k)

    def setdefault(self, key: Any, default: Any = None) -> Any:
        k = id(key) if isinstance(key, Tensor) else key
        return super().setdefault(k, default)

    def get(self, key: Any, default: Any = None) -> Any:
        k = id(key) if isinstance(key, Tensor) else key
        return super().get(k, default)


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
        self.state: Dict[Any, Dict[str, Any]] = _ParamStateDict()
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

    def state_dict(self) -> Dict[str, Any]:
        """Return the state of the optimizer as a dictionary.

        Returns:
            Dictionary containing:
                - 'state': A dict mapping parameter IDs (0, 1, ...) to state dicts.
                - 'param_groups': A list of param groups with 'params' mapped to parameter IDs.
        """
        # Create continuous integer mapping for all parameters across groups
        param_to_id: Dict[int, int] = {}
        all_params: List[Tensor] = []
        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in param_to_id:
                    param_to_id[id(p)] = len(all_params)
                    all_params.append(p)

        # Serialize state
        serialized_state: Dict[int, Dict[str, Any]] = {}
        for p_id, idx in param_to_id.items():
            if p_id in self.state:
                p_state = self.state[p_id]
                entry: Dict[str, Any] = {}
                for k, v in p_state.items():
                    if isinstance(v, np.ndarray):
                        entry[k] = v.copy()
                    elif isinstance(v, Tensor):
                        entry[k] = v.numpy().copy()
                    else:
                        entry[k] = v
                serialized_state[idx] = entry

        # Serialize param_groups
        serialized_groups: List[Dict[str, Any]] = []
        for group in self.param_groups:
            grp: Dict[str, Any] = {}
            for k, v in group.items():
                if k == "params":
                    grp["params"] = [param_to_id[id(p)] for p in v]
                else:
                    grp[k] = v
            serialized_groups.append(grp)

        return {
            "state": serialized_state,
            "param_groups": serialized_groups,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load an optimizer state dictionary.

        Args:
            state_dict: Optimizer state dictionary returned by :meth:`state_dict`.

        Raises:
            SerializationError: If state_dict structure or param_groups count is invalid.
        """
        if not isinstance(state_dict, dict) or "state" not in state_dict or "param_groups" not in state_dict:
            raise SerializationError("Invalid optimizer state_dict format (missing 'state' or 'param_groups')")

        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise SerializationError(
                f"Loaded state_dict contains {len(saved_groups)} param groups, "
                f"but optimizer has {len(self.param_groups)}"
            )

        # Build list of all current parameters in group order
        param_to_id: Dict[int, int] = {}
        id_to_param: Dict[int, Tensor] = {}
        all_params: List[Tensor] = []

        for group in self.param_groups:
            for p in group["params"]:
                if id(p) not in param_to_id:
                    idx = len(all_params)
                    param_to_id[id(p)] = idx
                    id_to_param[idx] = p
                    all_params.append(p)

        # Update param_group hyperparameters
        for current_group, saved_group in zip(self.param_groups, saved_groups):
            for k, v in saved_group.items():
                if k != "params":
                    current_group[k] = v

        # Restore parameter states
        saved_state = state_dict["state"]
        self.state.clear()

        for idx_key, p_state in saved_state.items():
            idx = int(idx_key)
            if idx not in id_to_param:
                continue
            p = id_to_param[idx]

            restored_state: Dict[str, Any] = {}
            for k, v in p_state.items():
                if isinstance(v, np.ndarray):
                    restored_state[k] = v.copy()
                elif isinstance(v, Tensor):
                    restored_state[k] = v.numpy().copy()
                else:
                    restored_state[k] = v

            self.state[p] = restored_state

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " (\n"
        for i, group in enumerate(self.param_groups):
            format_string += f"Parameter Group {i}\n"
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += f"    {key}: {group[key]}\n"
        format_string += ")"
        return format_string
