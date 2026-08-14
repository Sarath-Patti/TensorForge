"""Memory storage abstraction for TensorForge.

Separates raw physical contiguous memory management and buffer operations from
tensor metadata (shape, strides). In v0.1, storage operates exclusively on
contiguous memory allocations. Designed for clean extensibility toward custom C++
memory allocators, pinned host memory, and device runtimes in later milestones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Final, Optional
import numpy as np

from tensorforge.tensor.dtype import DType, float32, to_dtype
from tensorforge.utils.validation import StorageError


class Storage(ABC):
    """Abstract base class representing contiguous physical memory allocation.

    Attributes:
        dtype: Data type of stored elements.
        numel: Total number of elements allocated in this storage.
        device: Target execution device ('cpu' in v0.1).
    """

    def __init__(self, dtype: DType, numel: int, device: str = "cpu") -> None:
        self._dtype: Final[DType] = dtype
        self._numel: Final[int] = numel
        self._device: Final[str] = device

    @property
    def dtype(self) -> DType:
        """Data type of the allocated memory."""
        return self._dtype

    @property
    def numel(self) -> int:
        """Total number of elements in this storage."""
        return self._numel

    @property
    def itemsize(self) -> int:
        """Bytes per element."""
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Total allocated memory in bytes."""
        return self._numel * self._dtype.itemsize

    @property
    def device(self) -> str:
        """Device name where memory is allocated."""
        return self._device

    @property
    @abstractmethod
    def data_ptr(self) -> Optional[int]:
        """Virtual memory address pointing to the start of the buffer.

        Used for low-level C++ runtime bindings and pointer arithmetic.
        """
        pass

    @abstractmethod
    def to_numpy(self) -> np.ndarray:
        """Return the underlying buffer as a 1D NumPy array."""
        pass

    @abstractmethod
    def copy(self) -> Storage:
        """Create a deep copy of this memory storage."""
        pass

    @abstractmethod
    def fill_(self, value: float | int) -> None:
        """In-place fill all storage elements with a scalar value."""
        pass

    def __len__(self) -> int:
        return self._numel

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} numel={self._numel} dtype={self._dtype} nbytes={self.nbytes}B device='{self._device}'>"


class NumPyStorage(Storage):
    """Concrete storage implementation backed by a contiguous NumPy buffer.

    Serves as the reference CPU storage implementation for v0.1.
    """

    def __init__(self, raw_array: np.ndarray, dtype: Optional[DType] = None, device: str = "cpu") -> None:
        if not isinstance(raw_array, np.ndarray):
            raise StorageError(f"NumPyStorage requires an ndarray, got {type(raw_array).__name__}")

        resolved_dtype = dtype if dtype is not None else to_dtype(raw_array.dtype)

        # Ensure array is contiguous and cast to target dtype if needed
        if raw_array.dtype != resolved_dtype.numpy_dtype:
            arr = np.ascontiguousarray(raw_array.astype(resolved_dtype.numpy_dtype, copy=False))
        elif not raw_array.flags.c_contiguous:
            arr = np.ascontiguousarray(raw_array)
        else:
            arr = raw_array

        super().__init__(dtype=resolved_dtype, numel=arr.size, device=device)
        self._data: np.ndarray = arr.reshape(-1)

    @classmethod
    def allocate(cls, numel: int, dtype: DType = float32, device: str = "cpu") -> NumPyStorage:
        """Allocate uninitialized or zero-initialized memory buffer of specified size.

        Args:
            numel: Total number of elements.
            dtype: Data type to allocate.
            device: Allocation device.

        Returns:
            A new NumPyStorage instance.
        """
        if numel < 0:
            raise StorageError(f"Cannot allocate storage with negative element count: {numel}")
        raw = np.empty(numel, dtype=dtype.numpy_dtype)
        return cls(raw, dtype=dtype, device=device)

    @classmethod
    def zeros(cls, numel: int, dtype: DType = float32, device: str = "cpu") -> NumPyStorage:
        """Allocate a zero-filled storage buffer."""
        if numel < 0:
            raise StorageError(f"Cannot allocate storage with negative element count: {numel}")
        raw = np.zeros(numel, dtype=dtype.numpy_dtype)
        return cls(raw, dtype=dtype, device=device)

    @classmethod
    def ones(cls, numel: int, dtype: DType = float32, device: str = "cpu") -> NumPyStorage:
        """Allocate a one-filled storage buffer."""
        if numel < 0:
            raise StorageError(f"Cannot allocate storage with negative element count: {numel}")
        raw = np.ones(numel, dtype=dtype.numpy_dtype)
        return cls(raw, dtype=dtype, device=device)

    @classmethod
    def from_array(cls, array: np.ndarray, dtype: Optional[DType] = None, copy: bool = True) -> NumPyStorage:
        """Create storage from an existing NumPy array."""
        target_dtype = dtype if dtype is not None else to_dtype(array.dtype)
        data = np.array(array, dtype=target_dtype.numpy_dtype, copy=copy, order="C")
        return cls(data, dtype=target_dtype)

    @property
    def data_ptr(self) -> int:
        """Memory address of the underlying NumPy buffer pointer."""
        return int(self._data.ctypes.data)

    @property
    def raw_data(self) -> np.ndarray:
        """Internal 1D NumPy array view."""
        return self._data

    def to_numpy(self) -> np.ndarray:
        """Return the contiguous 1D NumPy array representation."""
        return self._data

    def copy(self) -> NumPyStorage:
        """Return a deep copy of the storage buffer."""
        return NumPyStorage(self._data.copy(), dtype=self._dtype, device=self._device)

    def fill_(self, value: float | int) -> None:
        """In-place fill the entire buffer with value."""
        self._data.fill(value)
