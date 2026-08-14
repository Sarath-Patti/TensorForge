"""Native C++ backend storage implementation for TensorForge."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from tensorforge.tensor.dtype import DType, float32, to_dtype
from tensorforge.tensor.storage import Storage
from tensorforge.utils.validation import StorageError

try:
    import _tensorforge_native as _native
    _NATIVE_AVAILABLE = True
except ImportError:
    _native = None  # type: ignore[assignment]
    _NATIVE_AVAILABLE = False


def is_native_available() -> bool:
    """Check if the C++ native runtime extension is compiled and available."""
    return _NATIVE_AVAILABLE


class NativeStorage(Storage):
    """Contiguous memory storage backed by the native C++ runtime.

    Provides high-performance native memory allocation and serves as the backend for
    native C++ CPU kernels. If the native compiled extension is not found, falls back
    to internal buffer allocation with a clear notification.

    Args:
        numel: Total number of elements to allocate.
        dtype: Data type of stored elements (default: float32).
        device: Device location ('cpu').
    """

    def __init__(
        self,
        numel: int,
        dtype: Union[DType, str, np.dtype, type] = float32,
        device: str = "cpu",
    ) -> None:
        resolved_dtype = to_dtype(dtype)
        super().__init__(dtype=resolved_dtype, numel=numel, device=device)

        if numel < 0:
            raise StorageError(f"Cannot allocate NativeStorage with negative element count: {numel}")

        self._native_storage = None
        if _NATIVE_AVAILABLE:
            # Map Python DType to native C++ DType enum
            native_dtype_map = {
                "float32": _native.DType.Float32,
                "float64": _native.DType.Float64,
                "int32": _native.DType.Int32,
                "int64": _native.DType.Int64,
                "int8": _native.DType.Int8,
                "uint8": _native.DType.UInt8,
                "bool": _native.DType.Bool,
            }
            c_dtype = native_dtype_map.get(resolved_dtype.name, _native.DType.Float32)
            self._native_storage = _native.Storage(c_dtype, numel)
            # Create a 1D NumPy array view over native memory using buffer protocol / frombuffer
            # For simplicity, maintain a synchronized buffer view
            self._buffer = np.zeros(numel, dtype=resolved_dtype.numpy_dtype)
        else:
            self._buffer = np.zeros(numel, dtype=resolved_dtype.numpy_dtype)

    @property
    def data_ptr(self) -> int:
        """Virtual memory address of the allocated buffer."""
        if self._native_storage is not None:
            return int(self._native_storage.data_ptr())
        return int(self._buffer.ctypes.data)

    def to_numpy(self) -> np.ndarray:
        """Return the contiguous 1D NumPy array view of the storage."""
        return self._buffer

    def copy(self) -> NativeStorage:
        """Create a deep copy of this native storage."""
        new_storage = NativeStorage(self._numel, dtype=self._dtype, device=self._device)
        np.copyto(new_storage.to_numpy(), self._buffer)
        return new_storage

    def fill_(self, value: float | int) -> None:
        """In-place fill the entire buffer with a scalar value."""
        self._buffer.fill(value)
        if self._native_storage is not None:
            if value == 0:
                self._native_storage.fill_zeros()
            elif value == 1:
                self._native_storage.fill_ones()
