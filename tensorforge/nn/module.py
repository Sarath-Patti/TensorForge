"""Base Module class for neural network layers and containers in TensorForge."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Set, Tuple

from tensorforge.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules in TensorForge.

    Supports automatic parameter registration, hierarchical submodule management,
    train/eval mode switching, recursive gradient zeroing, and callable forward execution.
    """

    def __init__(self) -> None:
        """Initialize base Module internals."""
        self._parameters: OrderedDict[str, Optional[Parameter]] = OrderedDict()
        self._modules: OrderedDict[str, Optional[Module]] = OrderedDict()
        self._training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        """Intercept attribute assignments to automatically register Parameters and Modules."""
        params = self.__dict__.get("_parameters")
        modules = self.__dict__.get("_modules")

        if isinstance(value, Parameter):
            if params is None:
                raise AttributeError("Cannot assign Parameter before Module.__init__() call")
            if modules is not None and name in modules:
                del modules[name]
            params[name] = value
        elif isinstance(value, Module):
            if modules is None:
                raise AttributeError("Cannot assign Module before Module.__init__() call")
            if params is not None and name in params:
                del params[name]
            modules[name] = value
        else:
            if params is not None and name in params:
                del params[name]
            if modules is not None and name in modules:
                del modules[name]

        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        """Lookup parameters or submodules if not found directly in __dict__."""
        if "_parameters" in self.__dict__:
            params = self.__dict__["_parameters"]
            if name in params:
                return params[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __delattr__(self, name: str) -> None:
        """Remove attribute and unregister from parameters or modules dictionaries."""
        if name in self._parameters:
            del self._parameters[name]
        if name in self._modules:
            del self._modules[name]
        super().__delattr__(name)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Explicitly register a parameter to the module.

        Args:
            name: Name of parameter.
            param: Parameter object or None.
        """
        if param is not None and not isinstance(param, Parameter):
            raise TypeError(f"Cannot register {type(param).__name__} as parameter (must be Parameter or None)")
        self._parameters[name] = param

    def register_module(self, name: str, module: Optional[Module]) -> None:
        """Explicitly register a child submodule.

        Args:
            name: Name of submodule.
            module: Module object or None.
        """
        if module is not None and not isinstance(module, Module):
            raise TypeError(f"Cannot register {type(module).__name__} as submodule (must be Module or None)")
        self._modules[name] = module

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Return an iterator over module parameters.

        Args:
            recurse: If True, yields parameters of submodules recursively.

        Yields:
            Parameter objects.
        """
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        memo: Optional[Set[int]] = None,
    ) -> Iterator[Tuple[str, Parameter]]:
        """Return an iterator over module parameters, yielding both name and parameter.

        Args:
            prefix: Prefix to prepend to parameter names.
            recurse: If True, yields parameters of submodules recursively.
            memo: Optional set of parameter object ids to prevent duplicate yields.

        Yields:
            (name, parameter) tuples.
        """
        if memo is None:
            memo = set()

        for name, param in self._parameters.items():
            if param is not None and id(param) not in memo:
                memo.add(id(param))
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param

        if recurse:
            for mod_name, module in self._modules.items():
                if module is not None:
                    sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                    for sub_name, sub_param in module.named_parameters(
                        prefix=sub_prefix, recurse=True, memo=memo
                    ):
                        yield sub_name, sub_param

    def modules(self) -> Iterator[Module]:
        """Return an iterator over all modules in the network (self and all child submodules)."""
        for _, module in self.named_modules():
            yield module

    def named_modules(
        self,
        memo: Optional[Set[int]] = None,
        prefix: str = "",
    ) -> Iterator[Tuple[str, Module]]:
        """Return an iterator over all modules in the network, yielding both name and module."""
        if memo is None:
            memo = set()

        if id(self) not in memo:
            memo.add(id(self))
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                sub_prefix = f"{prefix}.{name}" if prefix else name
                for sub_name, sub_mod in module.named_modules(memo=memo, prefix=sub_prefix):
                    yield sub_name, sub_mod

    def zero_grad(self) -> None:
        """Reset gradients of all module parameters to None."""
        for param in self.parameters():
            param.zero_grad()

    def train(self, mode: bool = True) -> Module:
        """Set the module and all child submodules into training mode.

        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).

        Returns:
            self.
        """
        self._training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module into evaluation mode (equivalent to self.train(False)).

        Returns:
            self.
        """
        return self.train(False)

    @property
    def training(self) -> bool:
        """Whether module is currently in training mode."""
        return self._training

    def state_dict(
        self,
        destination: Optional[OrderedDict[str, Any]] = None,
        prefix: str = "",
        keep_vars: bool = False,
    ) -> OrderedDict[str, Any]:
        """Return a dictionary containing references to or copies of module parameters.

        Args:
            destination: Optional existing dictionary to populate.
            prefix: Prefix to prepend to parameter names.
            keep_vars: If True, returns original Parameter objects. If False, returns detached Tensor copies.

        Returns:
            OrderedDict mapping parameter full names to Tensor objects.
        """
        if destination is None:
            destination = OrderedDict()

        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if keep_vars:
                destination[name] = param
            else:
                destination[name] = param.detach().copy()

        return destination

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict: bool = True,
    ) -> Tuple[List[str], List[str]]:
        """Copy parameter values from a state dictionary into this module and its descendants.

        Copies values into existing physical Parameter storage in-place, preserving
        Parameter object identity, requires_grad, and leaf status without creating autograd nodes.

        Args:
            state_dict: A dictionary mapping parameter names to Tensor or array-like objects.
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned
                by this module's named_parameters().

        Returns:
            Tuple of (missing_keys, unexpected_keys).

        Raises:
            SerializationError: If strict=True and missing or unexpected keys are found,
                or if parameter shapes/dtypes are incompatible.
        """
        from tensorforge.autograd.engine import no_grad
        from tensorforge.tensor.tensor import Tensor
        from tensorforge.utils.validation import SerializationError
        import numpy as np

        local_params = OrderedDict(self.named_parameters(recurse=True))
        local_keys = set(local_params.keys())
        input_keys = set(state_dict.keys())

        missing_keys = sorted(list(local_keys - input_keys))
        unexpected_keys = sorted(list(input_keys - local_keys))

        if strict:
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing key(s) in state_dict: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected_keys}")
            if error_msgs:
                raise SerializationError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t"
                    + "\n\t".join(error_msgs)
                )

        with no_grad():
            for name, param in local_params.items():
                if name not in state_dict:
                    continue

                val = state_dict[name]
                if isinstance(val, Tensor):
                    val_arr = val.numpy()
                    val_shape = val.shape
                elif isinstance(val, np.ndarray):
                    val_arr = val
                    val_shape = val.shape
                else:
                    val_arr = np.asarray(val)
                    val_shape = val_arr.shape

                if param.shape != val_shape:
                    raise SerializationError(
                        f"Shape mismatch for parameter '{name}': expected {param.shape}, got {val_shape}"
                    )

                # In-place physical storage copy
                target_dtype = param.dtype.numpy_dtype
                converted_arr = val_arr.reshape(-1).astype(target_dtype, copy=False)
                np.copyto(param.storage.to_numpy(), converted_arr)

        return missing_keys, unexpected_keys

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Defines the forward computation executed at every call.

        Should be overridden by all subclasses.
        """
        raise NotImplementedError(f"Module [{type(self).__name__}] must implement forward()")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the module forward pass."""
        return self.forward(*args, **kwargs)

    def __repr__(self) -> str:
        """Pretty string representation of module and submodules."""
        child_lines = []
        for name, module in self._modules.items():
            if module is None:
                continue
            mod_str = repr(module)
            mod_str = "\n  ".join(mod_str.split("\n"))
            child_lines.append(f"({name}): {mod_str}")

        if not child_lines:
            return f"{type(self).__name__}()"

        lines_str = "\n  ".join(child_lines)
        return f"{type(self).__name__}(\n  {lines_str}\n)"
