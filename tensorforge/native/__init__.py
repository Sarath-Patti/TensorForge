"""TensorForge Native C++ Runtime and Extension Subpackage."""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from tensorforge.tensor.native_storage import is_native_available

try:
    import _tensorforge_native as _native
except ImportError:
    _native = None


def native_add(a: Any, b: Any) -> Any:
    """Execute native element-wise addition kernel."""
    if _native is not None:
        return _native.native_add(a, b)
    raise RuntimeError("Native C++ runtime is not compiled. Build native extensions with CMake first.")


def native_sub(a: Any, b: Any) -> Any:
    """Execute native element-wise subtraction kernel."""
    if _native is not None:
        return _native.native_sub(a, b)
    raise RuntimeError("Native C++ runtime is not compiled. Build native extensions with CMake first.")


def native_mul(a: Any, b: Any) -> Any:
    """Execute native element-wise multiplication kernel."""
    if _native is not None:
        return _native.native_mul(a, b)
    raise RuntimeError("Native C++ runtime is not compiled. Build native extensions with CMake first.")


def native_matmul(a: Any, b: Any) -> Any:
    """Execute native CPU matrix multiplication kernel."""
    if _native is not None:
        return _native.native_matmul(a, b)
    raise RuntimeError("Native C++ runtime is not compiled. Build native extensions with CMake first.")


__all__ = [
    "is_native_available",
    "native_add",
    "native_sub",
    "native_mul",
    "native_matmul",
]
