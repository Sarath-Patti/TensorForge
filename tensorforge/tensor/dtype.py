"""Data type abstraction and registry for TensorForge.

Defines supported data types (float32, float64, int32, int64) and provides
type promotion, casting rules, and mapping to underlying numerical storage.
The abstraction is designed to cleanly accommodate future low-precision
(float16, bfloat16) and quantized (int8, int4) inference types.
"""

from __future__ import annotations

from typing import Any, Dict, Final, Optional
import numpy as np

from tensorforge.utils.validation import DTypeError


class DType:
    """Represents a data type for tensors in TensorForge.

    Attributes:
        name: Canonical name of the data type (e.g. 'float32', 'int64').
        itemsize: Number of bytes occupied by a single element.
        numpy_dtype: The corresponding NumPy dtype object.
        is_floating_point: Whether the type is a floating-point type.
        is_integer: Whether the type is an integer type.
        is_quantized: Whether the type is a quantized representation (e.g., int8, int4).
        c_type: C/C++ equivalent type name for future C++ runtime interop.
    """

    def __init__(
        self,
        name: str,
        itemsize: int,
        numpy_dtype: np.dtype,
        is_floating_point: bool = False,
        is_integer: bool = False,
        is_quantized: bool = False,
        c_type: str = "",
    ) -> None:
        self._name: Final[str] = name
        self._itemsize: Final[int] = itemsize
        self._numpy_dtype: Final[np.dtype] = np.dtype(numpy_dtype)
        self._is_floating_point: Final[bool] = is_floating_point
        self._is_integer: Final[bool] = is_integer
        self._is_quantized: Final[bool] = is_quantized
        self._c_type: Final[str] = c_type or name

    @property
    def name(self) -> str:
        """The canonical name of the data type."""
        return self._name

    @property
    def itemsize(self) -> int:
        """Size of a single element in bytes."""
        return self._itemsize

    @property
    def numpy_dtype(self) -> np.dtype:
        """The underlying NumPy dtype."""
        return self._numpy_dtype

    @property
    def is_floating_point(self) -> bool:
        """True if the data type is a floating point representation."""
        return self._is_floating_point

    @property
    def is_integer(self) -> bool:
        """True if the data type is an integer representation."""
        return self._is_integer

    @property
    def is_quantized(self) -> bool:
        """True if the data type represents a quantized format.

        Note (v0.1):
            In v0.1, types like int8 are supported as low-precision storage types
            to prepare for future quantization milestones (v0.7). Scale and
            zero-point quantization math or QuantizedTensor are not yet implemented.
        """
        return self._is_quantized

    @property
    def c_type(self) -> str:
        """C++ type definition string for future C++ inference runtime bindings."""
        return self._c_type

    def __repr__(self) -> str:
        return f"tensorforge.{self._name}"

    def __str__(self) -> str:
        return self._name

    def __eq__(self, other: object) -> bool:
        if isinstance(other, DType):
            return self._name == other._name
        if isinstance(other, (np.dtype, type, str)):
            try:
                return self == to_dtype(other)
            except DTypeError:
                return False
        return False

    def __hash__(self) -> int:
        return hash(self._name)


# Standard core supported data types for v0.1
float32 = DType("float32", 4, np.dtype(np.float32), is_floating_point=True, c_type="float")
float64 = DType("float64", 8, np.dtype(np.float64), is_floating_point=True, c_type="double")
int32 = DType("int32", 4, np.dtype(np.int32), is_integer=True, c_type="int32_t")
int64 = DType("int64", 8, np.dtype(np.int64), is_integer=True, c_type="int64_t")

# Extended types designed for future milestones (FP16, INT8, bool)
# Note: int8 is currently supported as a low-precision contiguous storage dtype
# to lay the groundwork for v0.7 quantization. Scale/zero-point quantization arithmetic
# and QuantizedTensor will be introduced in milestone v0.7.
float16 = DType("float16", 2, np.dtype(np.float16), is_floating_point=True, c_type="uint16_t")
int8 = DType("int8", 1, np.dtype(np.int8), is_integer=True, is_quantized=True, c_type="int8_t")
uint8 = DType("uint8", 1, np.dtype(np.uint8), is_integer=True, c_type="uint8_t")
bool_ = DType("bool", 1, np.dtype(np.bool_), c_type="bool")

# DType lookup registry
_DTYPE_REGISTRY: Dict[str, DType] = {
    "float32": float32,
    "float": float32,
    "fp32": float32,
    "float64": float64,
    "double": float64,
    "fp64": float64,
    "int32": int32,
    "int": int64,  # Python int maps to int64 by default
    "int64": int64,
    "long": int64,
    "float16": float16,
    "fp16": float16,
    "half": float16,
    "int8": int8,
    "uint8": uint8,
    "bool": bool_,
}

_NUMPY_DTYPE_MAP: Dict[np.dtype, DType] = {
    np.dtype(np.float32): float32,
    np.dtype(np.float64): float64,
    np.dtype(np.int32): int32,
    np.dtype(np.int64): int64,
    np.dtype(np.float16): float16,
    np.dtype(np.int8): int8,
    np.dtype(np.uint8): uint8,
    np.dtype(np.bool_): bool_,
}


def to_dtype(val: Any) -> DType:
    """Convert a string, Python type, NumPy dtype, or DType instance into a TensorForge DType.

    Args:
        val: Input representation of a data type.

    Returns:
        The corresponding canonical DType instance.

    Raises:
        DTypeError: If the input cannot be resolved to a supported TensorForge DType.
    """
    if isinstance(val, DType):
        return val

    if val is float:
        return float32
    if val is int:
        return int64
    if val is bool:
        return bool_

    if isinstance(val, str):
        cleaned = val.strip().lower()
        if cleaned in _DTYPE_REGISTRY:
            return _DTYPE_REGISTRY[cleaned]
        raise DTypeError(f"Unsupported dtype string '{val}'. Supported: float32, float64, int32, int64, float16, int8, uint8, bool")

    if isinstance(val, (np.dtype, type)):
        np_dt = np.dtype(val)
        if np_dt in _NUMPY_DTYPE_MAP:
            return _NUMPY_DTYPE_MAP[np_dt]
        raise DTypeError(f"Unsupported NumPy dtype '{val}'. Supported: float32, float64, int32, int64, float16, int8, uint8, bool")

    raise DTypeError(f"Cannot convert object of type '{type(val).__name__}' to DType")


def promote_dtypes(dt1: DType, dt2: DType) -> DType:
    """Determine the resulting data type when performing binary operations between two dtypes.

    Follows standard type promotion hierarchy:
    - If either is float64 -> float64
    - Else if either is float32 -> float32
    - Else if either is float16 -> float16
    - Else if either is int64 -> int64
    - Else if either is int32 -> int32
    - Else if either is int8 / uint8 -> largest integer
    - Default to float32 if mixed float/int.

    Args:
        dt1: First data type.
        dt2: Second data type.

    Returns:
        Promoted data type.
    """
    if dt1 == dt2:
        return dt1

    # Mixed floating and integer -> floating point
    if dt1.is_floating_point and not dt2.is_floating_point:
        return dt1
    if dt2.is_floating_point and not dt1.is_floating_point:
        return dt2

    # Both floating point
    if dt1.is_floating_point and dt2.is_floating_point:
        if dt1 == float64 or dt2 == float64:
            return float64
        if dt1 == float32 or dt2 == float32:
            return float32
        return float16

    # Both integer
    if dt1.is_integer and dt2.is_integer:
        if dt1 == int64 or dt2 == int64:
            return int64
        if dt1 == int32 or dt2 == int32:
            return int32
        return int8

    # Fallback to NumPy's result_type
    promoted_np = np.result_type(dt1.numpy_dtype, dt2.numpy_dtype)
    return to_dtype(promoted_np)
