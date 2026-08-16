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
    """Check if two operands are eligible for native C++ element-wise kernel execution."""
    if not is_native_available() or _native is None:
        return False
    if a_dtype != float32 or b_dtype != float32:
        return False
    return a_shape == b_shape


def can_native_matmul(a_dtype: object, a_ndim: int, a_shape: Tuple[int, ...], b_dtype: object, b_ndim: int, b_shape: Tuple[int, ...]) -> bool:
    """Check if matrix multiplication operands are eligible for native C++ matmul execution."""
    if not is_native_available() or _native is None:
        return False
    if a_dtype != float32 or b_dtype != float32:
        return False
    if a_ndim != 2 or b_ndim != 2:
        return False
    return a_shape[1] == b_shape[0]


def can_native_qmatmul(a_ndim: int, a_shape: Tuple[int, ...], b_ndim: int, b_shape: Tuple[int, ...]) -> bool:
    """Check if quantized matrix multiplication operands are eligible for native C++ INT8 execution."""
    if not is_native_available() or _native is None:
        return False
    if not hasattr(_native, "native_qmatmul"):
        return False
    if a_ndim != 2 or b_ndim != 2:
        return False
    return a_shape[1] == b_shape[0]


def can_native_fused_linear(
    x_dtype: object,
    x_ndim: int,
    x_shape: Tuple[int, ...],
    w_shape: Tuple[int, ...],
) -> bool:
    """Check if Linear/FusedLinear operation is eligible for native C++ execution."""
    if not is_native_available() or _native is None:
        return False
    if not hasattr(_native, "native_fused_linear"):
        return False
    if x_dtype != float32:
        return False
    if x_ndim != 2 or len(w_shape) != 2:
        return False
    return x_shape[1] == w_shape[1]


def can_native_fused_qlinear_relu(
    x_ndim: int,
    x_shape: Tuple[int, ...],
    w_ndim: int,
    w_shape: Tuple[int, ...],
) -> bool:
    """Check if INT8 Fused Linear + ReLU is eligible for native C++ execution."""
    if not is_native_available() or _native is None:
        return False
    if not hasattr(_native, "native_fused_qlinear_relu"):
        return False
    if x_ndim != 2 or w_ndim != 2:
        return False
    return x_shape[1] == w_shape[1]


def native_add(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Execute float32 element-wise addition using the native C++ kernel."""
    if _native is None:
        raise RuntimeError("Native C++ backend is not loaded.")

    shape = _native.Shape(list(a_arr.shape))
    t_a = _native.Tensor(shape, _native.DType.Float32)
    t_b = _native.Tensor(shape, _native.DType.Float32)

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


# ============================================================================
# Fused Kernel Callers
# ============================================================================

def _prepare_fused_linear_args(
    x_arr: np.ndarray,
    w_arr: np.ndarray,
    b_arr: Optional[np.ndarray],
) -> Tuple[Any, Any, Optional[Any]]:
    """Helper to prepare native tensors for fused linear operations."""
    shape_x = _native.Shape(list(x_arr.shape))
    shape_w = _native.Shape(list(w_arr.shape))

    t_x = _native.Tensor(shape_x, _native.DType.Float32)
    t_w = _native.Tensor(shape_w, _native.DType.Float32)

    c_x = _ptr_to_numpy(t_x.storage().data_ptr(), x_arr.shape, np.float32)
    c_w = _ptr_to_numpy(t_w.storage().data_ptr(), w_arr.shape, np.float32)
    np.copyto(c_x, x_arr)
    np.copyto(c_w, w_arr)

    t_b = None
    if b_arr is not None:
        shape_b = _native.Shape(list(b_arr.shape))
        t_b = _native.Tensor(shape_b, _native.DType.Float32)
        c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
        np.copyto(c_b, b_arr)

    return t_x, t_w, t_b


def native_fused_linear(x_arr: np.ndarray, w_arr: np.ndarray, b_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Execute Fused Linear: out = x @ w.T + b."""
    if _native is None or not hasattr(_native, "native_fused_linear"):
        raise RuntimeError("Native fused_linear is not loaded.")

    t_x, t_w, t_b = _prepare_fused_linear_args(x_arr, w_arr, b_arr)
    out_t = _native.native_fused_linear(t_x, t_w, t_b)
    out_shape = (x_arr.shape[0], w_arr.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()


def native_fused_linear_relu(x_arr: np.ndarray, w_arr: np.ndarray, b_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Execute Fused Linear + ReLU: out = max(0, x @ w.T + b)."""
    if _native is None or not hasattr(_native, "native_fused_linear_relu"):
        raise RuntimeError("Native fused_linear_relu is not loaded.")

    t_x, t_w, t_b = _prepare_fused_linear_args(x_arr, w_arr, b_arr)
    out_t = _native.native_fused_linear_relu(t_x, t_w, t_b)
    out_shape = (x_arr.shape[0], w_arr.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()


def native_fused_linear_sigmoid(x_arr: np.ndarray, w_arr: np.ndarray, b_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Execute Fused Linear + Sigmoid: out = 1 / (1 + exp(-(x @ w.T + b)))."""
    if _native is None or not hasattr(_native, "native_fused_linear_sigmoid"):
        raise RuntimeError("Native fused_linear_sigmoid is not loaded.")

    t_x, t_w, t_b = _prepare_fused_linear_args(x_arr, w_arr, b_arr)
    out_t = _native.native_fused_linear_sigmoid(t_x, t_w, t_b)
    out_shape = (x_arr.shape[0], w_arr.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()


def native_fused_linear_tanh(x_arr: np.ndarray, w_arr: np.ndarray, b_arr: Optional[np.ndarray] = None) -> np.ndarray:
    """Execute Fused Linear + Tanh: out = tanh(x @ w.T + b)."""
    if _native is None or not hasattr(_native, "native_fused_linear_tanh"):
        raise RuntimeError("Native fused_linear_tanh is not loaded.")

    t_x, t_w, t_b = _prepare_fused_linear_args(x_arr, w_arr, b_arr)
    out_t = _native.native_fused_linear_tanh(t_x, t_w, t_b)
    out_shape = (x_arr.shape[0], w_arr.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()


def native_fused_linear_softmax(x_arr: np.ndarray, w_arr: np.ndarray, b_arr: Optional[np.ndarray] = None, dim: int = -1) -> np.ndarray:
    """Execute Fused Linear + Softmax: out = softmax(x @ w.T + b, dim=-1)."""
    if _native is None or not hasattr(_native, "native_fused_linear_softmax"):
        raise RuntimeError("Native fused_linear_softmax is not loaded.")

    t_x, t_w, t_b = _prepare_fused_linear_args(x_arr, w_arr, b_arr)
    out_t = _native.native_fused_linear_softmax(t_x, t_w, t_b, dim)
    out_shape = (x_arr.shape[0], w_arr.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()


def native_fused_qlinear_relu(
    x_int8: np.ndarray,
    w_int8: np.ndarray,
    b_arr: Optional[np.ndarray],
    scale_x: float,
    zp_x: int,
    scale_w: float,
    zp_w: int,
) -> np.ndarray:
    """Execute INT8 Fused Linear + ReLU."""
    if _native is None or not hasattr(_native, "native_fused_qlinear_relu"):
        raise RuntimeError("Native fused_qlinear_relu is not loaded.")

    shape_x = _native.Shape(list(x_int8.shape))
    shape_w = _native.Shape(list(w_int8.shape))

    t_x = _native.Tensor(shape_x, _native.DType.Int8)
    t_w = _native.Tensor(shape_w, _native.DType.Int8)

    c_x = _ptr_to_numpy(t_x.storage().data_ptr(), x_int8.shape, np.int8)
    c_w = _ptr_to_numpy(t_w.storage().data_ptr(), w_int8.shape, np.int8)
    np.copyto(c_x, x_int8)
    np.copyto(c_w, w_int8)

    t_b = None
    if b_arr is not None:
        shape_b = _native.Shape(list(b_arr.shape))
        t_b = _native.Tensor(shape_b, _native.DType.Float32)
        c_b = _ptr_to_numpy(t_b.storage().data_ptr(), b_arr.shape, np.float32)
        np.copyto(c_b, b_arr)

    out_t = _native.native_fused_qlinear_relu(
        t_x, t_w, t_b, float(scale_x), int(zp_x), float(scale_w), int(zp_w)
    )
    out_shape = (x_int8.shape[0], w_int8.shape[0])
    return _ptr_to_numpy(out_t.storage().data_ptr(), out_shape, np.float32).copy()
