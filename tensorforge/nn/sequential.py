"""Sequential container module for TensorForge neural networks."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, Sequence, Union

from tensorforge.nn.module import Module
from tensorforge.tensor.tensor import Tensor


class Sequential(Module):
    """A sequential container that executes child modules in the order they are added.

    Example:
        >>> model = Sequential(
        ...     Linear(784, 128),
        ...     ReLU(),
        ...     Linear(128, 10),
        ... )
        >>> output = model(x)
    """

    def __init__(self, *args: Any) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            modules = args[0]
            for idx, module in enumerate(modules):
                self.register_module(str(idx), module)
        elif len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.register_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.register_module(str(idx), module)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Pass input through each child module in sequential order."""
        current: Tensor = input_tensor
        for module in self._modules.values():
            if module is not None:
                current = module(current)
        return current

    def __len__(self) -> int:
        """Return number of submodules in the container."""
        return len(self._modules)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, Sequential]:
        """Access submodules by integer index or slice."""
        module_list = list(self._modules.values())
        if isinstance(idx, slice):
            return Sequential(OrderedDict(list(self._modules.items())[idx]))
        return module_list[idx]  # type: ignore

    def __iter__(self) -> Iterator[Module]:
        """Iterate over child modules."""
        for module in self._modules.values():
            if module is not None:
                yield module

    def __repr__(self) -> str:
        child_lines = []
        for name, module in self._modules.items():
            if module is None:
                continue
            mod_str = repr(module)
            mod_str = "\n  ".join(mod_str.split("\n"))
            child_lines.append(f"({name}): {mod_str}")

        if not child_lines:
            return "Sequential()"

        lines_str = "\n  ".join(child_lines)
        return f"Sequential(\n  {lines_str}\n)"
