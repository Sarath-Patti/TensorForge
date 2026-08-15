"""Native C++ compute backend integration for TensorForge."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from tensorforge.tensor.dtype import float32
from tensorforge.tensor.native_storage import is_native_available

try:
    import _tensorforge_native as _native
except ImportError:
    try:
        from tensorforge import _tensorforge_native as _native
    except ImportError:
        _native = None


def can_native_elementwise(a_dtype: object, a_shape: Tuple[int, ...], b_dtype: object, b_shape: Tuple[int, ...]) -> bool:
    """Check if two operands are eligible for native C++ element-wise kernel execution.

    Prerequisites:
        - Native extension is available.
        - Both operands are float32.
        - Both operands have identical shapes (no multi-dimensional broadcasting required).
    """
    if not is_native_available() or _native is None:
        return False
    if a_dtype != float32 or b_dtype != float32:
        return False
    return a_shape == b_shape


def can_native_matmul(a_dtype: object, a_ndim: int, a_shape: Tuple[int, ...], b_dtype: object, b_ndim: int, b_shape: Tuple[int, ...]) -> bool:
    """Check if matrix multiplication operands are eligible for native C++ matmul execution.

    Prerequisites:
        - Native extension is available.
        - Both operands are float32.
        - Both operands are strictly 2D matrices (M, K) x (K, N).
    """
    if not is_native_available() or _native is None:
        return False
    if a_dtype != float32 or b_dtype != float32:
        return False
    if a_ndim != 2 or b_ndim != 2:
        return False
    return a_shape[1] == b_shape[0]


def native_add(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise addition using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

    # Copy input arrays into native storage
    c_a = np.ctypeslib.as_array((ctypes_float := np.ctypeslib.ndpointer(dtype=np.float32, shape=a_arr.shape)).from_address(t_a.storage().data_ptr()))
    c_b = np.ctypeslib.as_array(ctypes_float.from_address(t_b.storage().data_ptr()))
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_add(t_a, t_b)
    out_arr = np.ctypeslib.as_array(ctypes_float.from_address(out_t.storage().data_ptr())).copy()
    return out_arr


def native_sub(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise subtraction using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

    ctypes_float = np.ctypeslib.ndpointer(dtype=np.float32, shape=a_arr.shape)
    c_a = np.ctypeslib.as_array(ctypes_float.from_address(t_a.storage().data_ptr()))
    c_b = np.ctypeslib.as_array(ctypes_float.from_address(t_b.storage().data_ptr()))
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_sub(t_a, t_b)
    out_arr = np.ctypeslib.as_array(ctypes_float.from_address(out_t.storage().data_ptr())).copy()
    return out_arr


def native_mul(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise multiplication using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

    ctypes_float = np.ctypeslib.ndpointer(dtype=np.float32, shape=a_arr.shape)
    c_a = np.ctypeslib.as_array(ctypes_float.from_address(t_a.storage().data_ptr()))
    c_b = np.ctypeslib.as_array(ctypes_float.from_address(t_b.storage().data_ptr()))
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_mul(t_a, t_b)
    out_arr = np.ctypeslib.as_array(ctypes_float.from_address(out_t.storage().data_ptr())).copy()
    return out_arr


def native_matmul(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute 2D float32 matrix multiplication using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape_a = _native.Shape(list(a_arr.shape))
    shape_b = _native.Shape(list(b_arr.shape))
    t_a = _native.Tensor(shape_a, _native.DType.Float32)
    t_b = _native.Tensor(shape_b, _native.DType.Float32)

    c_a = np.ctypeslib.as_array(np.ctypeslib.ndpointer(dtype=np.float32, shape=a_arr.shape).from_address(t_a.storage().data_ptr()))
    c_b = np.ctypeslib.as_array(np.ctypeslib.ndpointer(dtype=np.float32, shape=b_arr.shape).from_address(t_b.storage().data_ptr()))
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_matmul(t_a, t_b)
    out_shape = (a_arr.shape[0], b_arr.shape[1])
    out_arr = np.ctypeslib.as_array(np.ctypeslib.ndpointer(dtype=np.float32, shape=out_shape).from_address(out_t.storage().data_ptr())).copy()
    return out_arr
