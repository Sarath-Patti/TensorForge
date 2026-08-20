"""QuantizedTensor abstraction for low-precision INT8 representation."""

from __future__ import annotations

from typing import Optional, Tuple, Union
import numpy as np

from tensorforge.tensor.dtype import DType, float32, int8, to_dtype
from tensorforge.tensor.storage import Storage
from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import QuantizationError, ShapeError


class QuantizedTensor:
    """Represents a low-precision quantized tensor with INT8 storage and affine parameters.

    In TensorForge v0.7, QuantizedTensor encapsulates contiguous physical INT8 storage
    along with linear scale and zero-point parameters:
        x_real ≈ (x_quant - zero_point) * scale

    Attributes:
        scale: Positive floating-point scaling factor.
        zero_point: Integer quantization zero-point offset.
        scheme: Quantization scheme ('symmetric' or 'asymmetric').
        orig_shape: Original multi-dimensional logical shape.
        orig_dtype: Original floating-point data type prior to quantization.
    """

    def __init__(
        self,
        qdata: Union[Tensor, np.ndarray],
        scale: float,
        zero_point: int = 0,
        scheme: str = "symmetric",
        orig_dtype: Union[DType, str] = float32,
        orig_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        if scale <= 0.0:
            raise QuantizationError(f"Quantization scale must be strictly positive, got: {scale}")

        norm_scheme = scheme.strip().lower()
        if norm_scheme not in ("symmetric", "asymmetric"):
            raise QuantizationError(f"Unsupported quantization scheme '{scheme}'. Supported: 'symmetric', 'asymmetric'.")

        self._scale: float = float(scale)
        self._zero_point: int = int(zero_point)
        self._scheme: str = norm_scheme
        self._orig_dtype: DType = to_dtype(orig_dtype)

        if isinstance(qdata, Tensor):
            if qdata.dtype != int8:
                # Convert underlying tensor to int8
                self._qtensor: Tensor = Tensor(qdata.numpy().astype(np.int8), dtype=int8)
            else:
                self._qtensor = qdata
        elif isinstance(qdata, np.ndarray):
            self._qtensor = Tensor(qdata.astype(np.int8), dtype=int8)
        else:
            self._qtensor = Tensor(np.asarray(qdata, dtype=np.int8), dtype=int8)

        self._orig_shape: Tuple[int, ...] = orig_shape if orig_shape is not None else self._qtensor.shape

    @property
    def shape(self) -> Tuple[int, ...]:
        """Logical multidimensional shape."""
        return self._orig_shape

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._orig_shape)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        return self._qtensor.numel

    @property
    def dtype(self) -> DType:
        """Quantized storage data type (int8)."""
        return int8

    @property
    def orig_dtype(self) -> DType:
        """Original unquantized floating-point data type."""
        return self._orig_dtype

    @property
    def scale(self) -> float:
        """Quantization scaling factor."""
        return self._scale

    @property
    def zero_point(self) -> int:
        """Quantization zero-point offset."""
        return self._zero_point

    @property
    def scheme(self) -> str:
        """Quantization scheme ('symmetric' or 'asymmetric')."""
        return self._scheme

    @property
    def storage(self) -> Storage:
        """Underlying contiguous physical memory storage."""
        return self._qtensor.storage

    @property
    def itemsize(self) -> int:
        """Number of bytes per element (1 byte for INT8)."""
        return 1

    @property
    def nbytes(self) -> int:
        """Total memory consumed by physical INT8 buffer in bytes."""
        return self._qtensor.nbytes

    def int_repr(self) -> Tensor:
        """Return the underlying integer Tensor (int8)."""
        return self._qtensor

    def numpy(self) -> np.ndarray:
        """Return the logical multidimensional NumPy int8 view."""
        return self._qtensor.numpy()

    def dequantize(self) -> Tensor:
        """Dequantize this INT8 tensor back to an FP32 Tensor."""
        from tensorforge.quantization.quantize import dequantize
        return dequantize(self)

    def transpose(self, *axes: Any) -> QuantizedTensor:
        """Transpose dimensions of the quantized tensor."""
        q_transposed = self._qtensor.transpose(*axes)
        return QuantizedTensor(
            qdata=q_transposed,
            scale=self._scale,
            zero_point=self._zero_point,
            scheme=self._scheme,
            orig_dtype=self._orig_dtype,
            orig_shape=q_transposed.shape,
        )

    def t(self) -> QuantizedTensor:
        """Transpose matrix (alias for 2D transpose)."""
        return self.transpose()

    @property
    def T(self) -> QuantizedTensor:
        """Transpose property matrix."""
        return self.transpose()

    def copy(self) -> QuantizedTensor:
        """Create a deep copy of this QuantizedTensor."""
        return QuantizedTensor(
            qdata=self._qtensor.copy(),
            scale=self._scale,
            zero_point=self._zero_point,
            scheme=self._scheme,
            orig_dtype=self._orig_dtype,
            orig_shape=self._orig_shape,
        )

    def __matmul__(self, other: Union[QuantizedTensor, Tensor]) -> Tensor:
        """Matrix multiplication using INT8 quantized operands."""
        from tensorforge.quantization.quantize import qmatmul
        return qmatmul(self, other)

    def __rmatmul__(self, other: Union[QuantizedTensor, Tensor]) -> Tensor:
        """Reverse matrix multiplication using INT8 quantized operands."""
        from tensorforge.quantization.quantize import qmatmul
        return qmatmul(other, self)

    def __repr__(self) -> str:
        shape_str = f"shape={self.shape}"
        scale_str = f"scale={self._scale:.6g}"
        zp_str = f"zero_point={self._zero_point}"
        scheme_str = f"scheme='{self._scheme}'"
        mem_str = f"memory={self.nbytes}B"
        return f"QuantizedTensor({self.numpy()!r}, {shape_str}, dtype=int8, {scale_str}, {zp_str}, {scheme_str}, {mem_str})"
