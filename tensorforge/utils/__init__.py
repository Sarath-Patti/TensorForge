"""Utility modules for TensorForge."""

from tensorforge.utils.validation import (
    DimensionError,
    DTypeError,
    IndexError_,
    QuantizationError,
    ShapeError,
    StorageError,
    TensorForgeError,
    validate_axis,
    validate_broadcast_shapes,
    validate_matmul_shapes,
    validate_reshape_shape,
    validate_shape,
    validate_transpose_axes,
)

__all__ = [
    "TensorForgeError",
    "ShapeError",
    "DimensionError",
    "DTypeError",
    "IndexError_",
    "StorageError",
    "QuantizationError",
    "validate_shape",
    "validate_reshape_shape",
    "validate_broadcast_shapes",
    "validate_matmul_shapes",
    "validate_axis",
    "validate_transpose_axes",
]
