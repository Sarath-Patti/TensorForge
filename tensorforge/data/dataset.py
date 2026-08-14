"""Dataset abstractions for TensorForge."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Union

from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import DimensionError


class Dataset(ABC):
    """Abstract base class representing a dataset.

    Subclasses must implement __len__ and __getitem__.
    """

    @abstractmethod
    def __getitem__(self, index: int) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Retrieve a sample by index."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return total number of samples in the dataset."""
        raise NotImplementedError


class TensorDataset(Dataset):
    """Dataset wrapping tensors where each sample is retrieved by indexing tensors along the first dimension.

    Args:
        *tensors: Tensors that have the same size in the first dimension.
    """

    def __init__(self, *tensors: Tensor) -> None:
        if len(tensors) == 0:
            raise ValueError("TensorDataset requires at least one tensor argument")

        first_len = tensors[0].shape[0] if tensors[0].ndim > 0 else 1
        for i, t in enumerate(tensors):
            if not isinstance(t, Tensor):
                raise TypeError(f"TensorDataset arguments must be Tensors, got {type(t).__name__}")
            t_len = t.shape[0] if t.ndim > 0 else 1
            if t_len != first_len:
                raise DimensionError(
                    f"Size mismatch between tensors: tensor 0 has length {first_len}, "
                    f"but tensor {i} has length {t_len}"
                )

        self.tensors: Tuple[Tensor, ...] = tuple(tensors)
        self._length: int = first_len

    def __getitem__(self, index: int) -> Tuple[Tensor, ...]:
        return tuple(t[index] for t in self.tensors)

    def __len__(self) -> int:
        return self._length
