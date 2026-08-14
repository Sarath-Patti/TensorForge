"""Tensor data structures and operations for TensorForge."""

from tensorforge.tensor.dtype import (
    DType,
    bool_,
    float16,
    float32,
    float64,
    int8,
    int32,
    int64,
    promote_dtypes,
    to_dtype,
    uint8,
)
from tensorforge.tensor.operations import (
    add,
    matmul,
    mean,
    mul,
    neg,
    reshape,
    sub,
    sum,
    transpose,
    truediv,
)
from tensorforge.tensor.shape import (
    broadcast_shapes,
    broadcast_strides,
    compute_contiguous_strides,
    compute_numel,
    is_c_contiguous,
)
from tensorforge.tensor.native_storage import NativeStorage
from tensorforge.tensor.storage import NumPyStorage, Storage
from tensorforge.tensor.tensor import (
    Tensor,
    arange,
    from_numpy,
    ones,
    randn,
    tensor,
    zeros,
)

__all__ = [
    # Core Tensor & factories
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "from_numpy",
    # DTypes
    "DType",
    "float32",
    "float64",
    "int32",
    "int64",
    "float16",
    "int8",
    "uint8",
    "bool_",
    "to_dtype",
    "promote_dtypes",
    # Storage
    "Storage",
    "NumPyStorage",
    "NativeStorage",
    # Shape & Strides
    "compute_numel",
    "compute_contiguous_strides",
    "is_c_contiguous",
    "broadcast_shapes",
    "broadcast_strides",
    # Operations
    "add",
    "sub",
    "mul",
    "truediv",
    "neg",
    "matmul",
    "reshape",
    "transpose",
    "sum",
    "mean",
]
