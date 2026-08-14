"""TensorForge: A Memory-Aware Deep Learning Framework and Inference Engine."""

from tensorforge.autograd.engine import backward, is_grad_enabled, no_grad
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
    compute_contiguous_strides,
    compute_numel,
)
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
from tensorforge.utils.validation import (
    DimensionError,
    DTypeError,
    IndexError_,
    ShapeError,
    StorageError,
    TensorForgeError,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    # Core
    "Tensor",
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "from_numpy",
    # Autograd
    "backward",
    "no_grad",
    "is_grad_enabled",
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
    # Storage & Shape
    "Storage",
    "NumPyStorage",
    "compute_numel",
    "compute_contiguous_strides",
    "broadcast_shapes",
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
    # Errors
    "TensorForgeError",
    "ShapeError",
    "DimensionError",
    "DTypeError",
    "IndexError_",
    "StorageError",
]
