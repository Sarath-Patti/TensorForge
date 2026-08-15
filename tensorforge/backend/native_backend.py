"""Native C++ compute backend integration for TensorForge."""

from __future__ import annotations

import ctypes
from typing import Optional, Tuple, Union
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


def _ptr_to_numpy(ptr: int, shape: Tuple[int, ...], dtype: Union[np.dtype, type, str]) -> np.ndarray:
    """Create a zero-copy NumPy array view from a raw virtual memory address.

    Args:
        ptr: Virtual memory address as an integer.
        shape: Desired tensor shape.
        dtype: NumPy data type.

    Returns:
        NumPy array view referencing the raw memory buffer.
    """
    resolved_dtype = np.dtype(dtype)
    numel = int(np.prod(shape))
    nbytes = numel * resolved_dtype.itemsize
    buf = (ctypes.c_char * nbytes).from_address(ptr)
    return np.frombuffer(buf, dtype=resolved_dtype).reshape(shape)


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


def can_native_qmatmul(a_ndim: int, a_shape: Tuple[int, ...], b_ndim: int, b_shape: Tuple[int, ...]) -> bool:
    """Check if quantized matrix multiplication operands are eligible for native C++ INT8 execution.

    Prerequisites:
        - Native extension is available and has `native_qmatmul`.
        - Both operands are strictly 2D matrices (M, K) x (K, N).
    """
    if not is_native_available() or _native is None:
        return False
    if not hasattr(_native, "native_qmatmul"):
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
    c_a = _ptr_to_numpy(t_a.storage().data_ptr(), a_arr.shape, np.float32)
    c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_add(t_a, t_b)
    out_arr = _ptr_to_numpy(out_t.storage().data_ptr(), a_arr.shape, np.float32).copy()
    return out_arr


def native_sub(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise subtraction using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

    c_a = _ptr_to_numpy(t_a.storage().data_ptr(), a_arr.shape, np.float32)
    c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_sub(t_a, t_b)
    out_arr = _ptr_to_numpy(out_t.storage().data_ptr(), a_arr.shape, np.float32).copy()
    return out_arr


def native_mul(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise multiplication using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

    c_a = _ptr_to_numpy(t_a.storage().data_ptr(), a_arr.shape, np.float32)
    c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_mul(t_a, t_b)
    out_arr = _ptr_to_numpy(out_t.storage().data_ptr(), a_arr.shape, np.float32).copy()
    return out_arr


def native_matmul(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute 2D float32 matrix multiplication using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape_a = _native.Shape(list(a_arr.shape))
    shape_b = _native.Shape(list(b_arr.shape))
    t_a = _native.Tensor(shape_a, _native.DType.Float32)
    t_b = _native.Tensor(shape_b, _native.DType.Float32)

    c_a = _ptr_to_numpy(t_a.storage().data_ptr(), a_arr.shape, np.float32)
    c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
    np.copyto(c_a, a_arr)
    np.copyto(c_b, b_arr)

    out_t = _native.native_matmul(t_a, t_b)
    out_shape = (a_arr.shape[0], b_arr.shape[1])
    out_arr = _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()
    return out_arr


def native_qmatmul(
    a_int8: np.ndarray,
    b_int8: np.ndarray,
    scale_a: float,
    zp_a: int,
    scale_b: float,
    zp_b: int,
) -> np.ndarray:
    """Execute 2D INT8 matrix multiplication using the native C++ kernel, returning float32 output."""
    if _native is None or not hasattr(_native, "native_qmatmul"):
        raise RuntimeError("Native C++ qmatmul is not loaded.")

    shape_a = _native.Shape(list(a_int8.shape))
    shape_b = _native.Shape(list(b_int8.shape))
    t_a = _native.Tensor(shape_a, _native.DType.Int8)
    t_b = _native.Tensor(shape_b, _native.DType.Int8)

    c_a = _ptr_to_numpy(t_a.storage().data_ptr(), a_int8.shape, np.int8)
    c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_int8.shape, np.int8)
    np.copyto(c_a, a_int8)
    np.copyto(c_b, b_int8)

    out_t = _native.native_qmatmul(t_a, t_b, float(scale_a), int(zp_a), float(scale_b), int(zp_b))
    out_shape = (a_int8.shape[0], b_int8.shape[1])
    out_arr = _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()
    return out_arr
