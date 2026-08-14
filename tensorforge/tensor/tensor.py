"""Core Tensor abstraction for TensorForge.

Encapsulates multi-dimensional array metadata (shape, strides, dtype) and links
to the underlying memory Storage. Provides intuitive arithmetic operator overloads,
indexing, numpy conversions, and metadata inspection.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.tensor.dtype import DType, float32, to_dtype
from tensorforge.tensor.shape import compute_contiguous_strides, compute_numel, is_c_contiguous
from tensorforge.tensor.storage import NumPyStorage, Storage
from tensorforge.utils.validation import (
    DimensionError,
    DTypeError,
    IndexError_,
    ShapeError,
    validate_shape,
)


class Tensor:
    """Multi-dimensional tensor for memory-aware numerical computing.

    Attributes:
        _storage: The contiguous physical memory buffer holding the numerical elements.
        _shape: Dimensions of the tensor.
        _strides: Strides in element counts for indexing dimensions.
        _dtype: Data type specification.
    """

    def __init__(
        self,
        data: Any,
        dtype: Optional[Union[DType, str, np.dtype, type]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        copy: bool = True,
    ) -> None:
        """Initialize a Tensor.

        Args:
            data: Input data (list, tuple, scalar, NumPy array, Storage, or another Tensor).
            dtype: Optional target data type.
            shape: Optional explicit shape override.
            copy: Whether to copy the underlying data buffer.
        """
        resolved_dtype: Optional[DType] = to_dtype(dtype) if dtype is not None else None

        if isinstance(data, Tensor):
            target_dt = resolved_dtype if resolved_dtype is not None else data.dtype
            self._shape: Tuple[int, ...] = shape if shape is not None else data.shape
            self._dtype: DType = target_dt
            if copy or target_dt != data.dtype:
                self._storage: Storage = NumPyStorage.from_array(data.numpy(), dtype=target_dt, copy=True)
            else:
                self._storage = data.storage
            self._strides: Tuple[int, ...] = compute_contiguous_strides(self._shape, itemsize=1)
            return

        if isinstance(data, Storage):
            self._storage = data if not copy else data.copy()
            self._dtype = self._storage.dtype
            if shape is None:
                self._shape = (self._storage.numel,)
            else:
                self._shape = validate_shape(shape)
                if compute_numel(self._shape) != self._storage.numel:
                    raise ShapeError(
                        f"Shape {self._shape} requires {compute_numel(self._shape)} elements, "
                        f"but storage has {self._storage.numel}"
                    )
            self._strides = compute_contiguous_strides(self._shape, itemsize=1)
            return

        # Handle NumPy array or nested Python lists / scalars
        if isinstance(data, np.ndarray):
            np_arr = data
        else:
            np_arr = np.array(data)

        # Default dtype inference: Python float -> float32, Python int -> int64
        if resolved_dtype is None:
            if np.issubdtype(np_arr.dtype, np.floating):
                resolved_dtype = float32
            elif np.issubdtype(np_arr.dtype, np.integer):
                resolved_dtype = to_dtype(np_arr.dtype)
            elif np.issubdtype(np_arr.dtype, np.bool_):
                resolved_dtype = to_dtype(np.bool_)
            else:
                resolved_dtype = to_dtype(np_arr.dtype)

        self._dtype = resolved_dtype
        cast_arr = np_arr.astype(self._dtype.numpy_dtype, copy=copy)

        if shape is not None:
            validated_shp = validate_shape(shape)
            if cast_arr.size != compute_numel(validated_shp):
                raise ShapeError(
                    f"Cannot create tensor with shape {validated_shp} from data of size {cast_arr.size}"
                )
            self._shape = validated_shp
        else:
            self._shape = tuple(cast_arr.shape)

        self._storage = NumPyStorage(cast_arr, dtype=self._dtype)
        self._strides = compute_contiguous_strides(self._shape, itemsize=1)

    # -------------------------------------------------------------------------
    # Metadata Properties
    # -------------------------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Tensor dimensions as a tuple of integers."""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of tensor dimensions."""
        return len(self._shape)

    @property
    def dtype(self) -> DType:
        """Data type of the tensor elements."""
        return self._dtype

    @property
    def numel(self) -> int:
        """Total number of elements in the tensor."""
        return compute_numel(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements (alias for numel)."""
        return self.numel

    @property
    def strides(self) -> Tuple[int, ...]:
        """Element strides along each dimension."""
        return self._strides

    @property
    def itemsize(self) -> int:
        """Size of a single element in bytes."""
        return self._dtype.itemsize

    @property
    def nbytes(self) -> int:
        """Total memory consumed by tensor elements in bytes."""
        return self.numel * self._dtype.itemsize

    @property
    def storage(self) -> Storage:
        """The underlying physical memory storage object."""
        return self._storage

    @property
    def is_contiguous(self) -> bool:
        """Whether the tensor memory layout is C-contiguous."""
        return is_c_contiguous(self._shape, self._strides, itemsize=1)

    @property
    def T(self) -> Tensor:
        """Transpose dimensions of the tensor, returning a new contiguous Tensor."""
        return self.transpose()

    # -------------------------------------------------------------------------
    # Conversions & Interop
    # -------------------------------------------------------------------------

    def numpy(self) -> np.ndarray:
        """Return the tensor data as a NumPy ndarray with matching shape and dtype."""
        return self._storage.to_numpy().reshape(self._shape)

    def tolist(self) -> Union[List[Any], float, int, bool]:
        """Convert the tensor to a nested Python list or scalar."""
        return self.numpy().tolist()

    def item(self) -> Union[float, int, bool]:
        """Extract the single scalar value of a 1-element tensor.

        Returns:
            Python scalar (float or int).

        Raises:
            DimensionError: If the tensor contains more or fewer than 1 element.
        """
        if self.numel != 1:
            raise DimensionError(f"item() can only be called on a single-element tensor, but tensor has {self.numel} elements")
        return self.numpy().item()

    def astype(self, dtype: Union[DType, str, np.dtype, type]) -> Tensor:
        """Cast tensor to a different data type.

        Args:
            dtype: Target data type.

        Returns:
            A new Tensor cast to the specified dtype.
        """
        target_dtype = to_dtype(dtype)
        if target_dtype == self._dtype:
            return self.clone()
        return Tensor(self.numpy(), dtype=target_dtype, copy=True)

    def clone(self) -> Tensor:
        """Create a deep copy of the tensor and its storage."""
        return Tensor(self.numpy(), dtype=self._dtype, copy=True)

    def contiguous(self) -> Tensor:
        """Return a contiguous in-memory copy of the tensor."""
        if self.is_contiguous:
            return self
        return Tensor(np.ascontiguousarray(self.numpy()), dtype=self._dtype)

    # -------------------------------------------------------------------------
    # Indexing & Slicing
    # -------------------------------------------------------------------------

    def __getitem__(self, index: Any) -> Tensor:
        """Access sub-tensors, slices, or scalar elements.

        Note (v0.1):
            In v0.1, indexing and slicing materializes and returns a new Tensor
            with its own contiguous memory storage (it does not return a shared-storage view).

        Returns:
            A new contiguous Tensor containing the indexed or sliced elements.

        Raises:
            IndexError_: If index is out of bounds.
        """
        try:
            arr = self.numpy()
            sliced = arr[index]
            return Tensor(sliced, dtype=self._dtype, copy=True)
        except IndexError as e:
            raise IndexError_(f"Index {index} is out of bounds for tensor with shape {self._shape}: {e}") from e

    def __setitem__(self, index: Any, value: Any) -> None:
        """Update tensor elements at the given index in-place.

        Args:
            index: Target index or slice.
            value: Scalar or Tensor/array to assign.
        """
        try:
            arr = self.numpy()
            if isinstance(value, Tensor):
                arr[index] = value.numpy().astype(self._dtype.numpy_dtype, copy=False)
            else:
                arr[index] = value
        except IndexError as e:
            raise IndexError_(f"Index {index} is out of bounds for tensor with shape {self._shape}: {e}") from e

    # -------------------------------------------------------------------------
    # Math & Structural Methods (delegating to operations)
    # -------------------------------------------------------------------------

    def reshape(self, *shape: Union[int, Sequence[int]]) -> Tensor:
        """Return a new contiguous tensor with the requested shape."""
        from tensorforge.tensor.operations import reshape as _reshape
        return _reshape(self, *shape)

    def transpose(self, *axes: Union[int, Sequence[int]]) -> Tensor:
        """Permute dimensions of the tensor, returning a new contiguous Tensor."""
        from tensorforge.tensor.operations import transpose as _transpose
        return _transpose(self, *axes)

    def sum(self, axis: Union[int, Sequence[int], None] = None, keepdims: bool = False) -> Tensor:
        """Compute the sum along the specified axis/axes."""
        from tensorforge.tensor.operations import sum as _sum
        return _sum(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Union[int, Sequence[int], None] = None, keepdims: bool = False) -> Tensor:
        """Compute the arithmetic mean along the specified axis/axes."""
        from tensorforge.tensor.operations import mean as _mean
        return _mean(self, axis=axis, keepdims=keepdims)

    # -------------------------------------------------------------------------
    # Operator Overloads
    # -------------------------------------------------------------------------

    def __add__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import add
        return add(self, other)

    def __radd__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import add
        return add(other, self)

    def __sub__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import sub
        return sub(self, other)

    def __rsub__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import sub
        return sub(other, self)

    def __mul__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import mul
        return mul(self, other)

    def __rmul__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import mul
        return mul(other, self)

    def __truediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import truediv
        return truediv(self, other)

    def __rtruediv__(self, other: Union[Tensor, float, int]) -> Tensor:
        from tensorforge.tensor.operations import truediv
        return truediv(other, self)

    def __matmul__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            return NotImplemented
        from tensorforge.tensor.operations import matmul
        return matmul(self, other)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        if not isinstance(other, Tensor):
            return NotImplemented
        from tensorforge.tensor.operations import matmul
        return matmul(other, self)

    def __neg__(self) -> Tensor:
        from tensorforge.tensor.operations import neg
        return neg(self)

    def __len__(self) -> int:
        if self.ndim == 0:
            raise DimensionError("len() of a 0-d tensor is undefined")
        return self._shape[0]

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if isinstance(other, Tensor):
            if self.shape != other.shape or self.dtype != other.dtype:
                return False
            return bool(np.array_equal(self.numpy(), other.numpy()))
        return False

    # -------------------------------------------------------------------------
    # String Representation
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        arr_str = np.array2string(
            self.numpy(),
            separator=", ",
            prefix="tensor(",
        )
        return f"tensor({arr_str}, dtype={self._dtype})"

    def __str__(self) -> str:
        return f"Tensor({self.numpy()}, shape={self._shape}, dtype={self._dtype})"


# -----------------------------------------------------------------------------
# Factory Functions
# -----------------------------------------------------------------------------

def tensor(
    data: Any,
    dtype: Optional[Union[DType, str, np.dtype, type]] = None,
    copy: bool = True,
) -> Tensor:
    """Create a Tensor from array-like numerical data.

    Args:
        data: Python list, scalar, tuple, or NumPy array.
        dtype: Optional target data type (e.g. float32, int64).
        copy: Whether to copy input data.

    Returns:
        A new TensorForge Tensor.
    """
    return Tensor(data, dtype=dtype, copy=copy)


def zeros(
    shape: Union[int, Sequence[int]],
    dtype: Union[DType, str, np.dtype, type] = float32,
) -> Tensor:
    """Create a tensor filled with zeros.

    Args:
        shape: Dimension sizes as separate ints or a sequence.
        dtype: Data type of the tensor (default: float32).

    Returns:
        A zero-initialized Tensor.
    """
    target_shape = validate_shape(shape)
    target_dtype = to_dtype(dtype)
    numel = compute_numel(target_shape)
    storage = NumPyStorage.zeros(numel, dtype=target_dtype)
    return Tensor(storage, shape=target_shape, copy=False)


def ones(
    shape: Union[int, Sequence[int]],
    dtype: Union[DType, str, np.dtype, type] = float32,
) -> Tensor:
    """Create a tensor filled with ones.

    Args:
        shape: Dimension sizes as separate ints or a sequence.
        dtype: Data type of the tensor (default: float32).

    Returns:
        A one-initialized Tensor.
    """
    target_shape = validate_shape(shape)
    target_dtype = to_dtype(dtype)
    numel = compute_numel(target_shape)
    storage = NumPyStorage.ones(numel, dtype=target_dtype)
    return Tensor(storage, shape=target_shape, copy=False)


def randn(
    *shape: Union[int, Sequence[int]],
    dtype: Union[DType, str, np.dtype, type] = float32,
) -> Tensor:
    """Create a tensor initialized with random standard normal distribution values.

    Args:
        *shape: Target shape as ints or a sequence.
        dtype: Target floating point data type.

    Returns:
        Randomly initialized Tensor.
    """
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        target_shape = validate_shape(shape[0])
    else:
        target_shape = validate_shape(shape)

    target_dtype = to_dtype(dtype)
    if not target_dtype.is_floating_point:
        raise DTypeError(f"randn requires a floating point dtype, got {target_dtype}")

    arr = np.random.randn(*target_shape).astype(target_dtype.numpy_dtype)
    return Tensor(arr, dtype=target_dtype, copy=False)


def arange(
    start: float | int,
    stop: Optional[float | int] = None,
    step: float | int = 1,
    dtype: Optional[Union[DType, str, np.dtype, type]] = None,
) -> Tensor:
    """Create a 1D tensor with values spanning a range [start, stop) with given step.

    Args:
        start: Start value (or stop value if stop is None).
        stop: Stop value (exclusive).
        step: Increment step.
        dtype: Optional data type.

    Returns:
        A 1D Tensor containing the sequence of numbers.
    """
    if stop is None:
        arr = np.arange(start, step=step)
    else:
        arr = np.arange(start, stop, step)

    target_dtype = to_dtype(dtype) if dtype is not None else None
    return Tensor(arr, dtype=target_dtype, copy=False)


def from_numpy(
    array: np.ndarray,
    dtype: Optional[Union[DType, str, np.dtype, type]] = None,
) -> Tensor:
    """Create a Tensor from an existing NumPy ndarray.

    If the input array is already C-contiguous and matches the target dtype,
    it is wrapped without unnecessary memory copying. If the array is
    non-contiguous (e.g. from slicing or transposition) or requires dtype
    casting, a contiguous copy is materialized to maintain TensorForge's
    contiguous memory invariant.

    Args:
        array: Input NumPy array.
        dtype: Optional data type to cast to.

    Returns:
        A Tensor backed by contiguous storage.
    """
    if not isinstance(array, np.ndarray):
        raise TypeError(f"from_numpy expected a numpy.ndarray, got {type(array).__name__}")
    return Tensor(array, dtype=dtype, copy=False)
