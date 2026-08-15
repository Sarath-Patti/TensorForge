"""Calibration utilities for computing optimal quantization parameters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import numpy as np

from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import QuantizationError


def _extract_numpy(val: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Helper to convert Tensor or array-like to floating-point NumPy array."""
    if isinstance(val, Tensor):
        return val.numpy().astype(np.float32)
    return np.asarray(val, dtype=np.float32)


def compute_quantization_params(
    min_val: float,
    max_val: float,
    scheme: str = "symmetric",
) -> Tuple[float, int]:
    """Compute scale and zero-point parameters given dynamic range [min_val, max_val].

    Args:
        min_val: Minimum observed real value.
        max_val: Maximum observed real value.
        scheme: Quantization scheme ('symmetric' or 'asymmetric').

    Returns:
        Tuple of (scale, zero_point).
    """
    norm_scheme = scheme.strip().lower()

    if norm_scheme == "symmetric":
        max_abs = max(abs(float(min_val)), abs(float(max_val)))
        if max_abs <= 1e-12:
            scale = 1.0
        else:
            scale = max_abs / 127.0
        zero_point = 0
        return scale, zero_point

    elif norm_scheme == "asymmetric":
        diff = float(max_val) - float(min_val)
        if diff <= 1e-12:
            scale = 1.0
            zero_point = 0
        else:
            scale = diff / 255.0
            raw_zp = np.round(-float(min_val) / scale) - 128.0
            zero_point = int(np.clip(raw_zp, -128, 127))
        return scale, zero_point

    else:
        raise QuantizationError(f"Unknown quantization scheme '{scheme}'. Supported: 'symmetric', 'asymmetric'.")


class Calibrator(ABC):
    """Abstract base class for calibration algorithms."""

    @abstractmethod
    def update(self, data: Union[Tensor, np.ndarray]) -> None:
        """Update calibration statistics with a new sample or batch of data."""
        pass

    @abstractmethod
    def compute_range(self) -> Tuple[float, float]:
        """Compute the calibrated dynamic range (min_val, max_val)."""
        pass

    def compute_params(self, scheme: str = "symmetric") -> Tuple[float, int]:
        """Compute quantization scale and zero_point for the selected scheme."""
        min_val, max_val = self.compute_range()
        return compute_quantization_params(min_val, max_val, scheme=scheme)

    @abstractmethod
    def reset(self) -> None:
        """Reset internal calibration accumulators."""
        pass


class MinMaxCalibrator(Calibrator):
    """Calibrator that tracks the absolute minimum and maximum observed values across batches."""

    def __init__(self) -> None:
        self.min_val: float = float("inf")
        self.max_val: float = float("-inf")
        self._observed: bool = False

    def update(self, data: Union[Tensor, np.ndarray]) -> None:
        arr = _extract_numpy(data)
        if arr.size == 0:
            return
        b_min = float(np.min(arr))
        b_max = float(np.max(arr))
        self.min_val = min(self.min_val, b_min)
        self.max_val = max(self.max_val, b_max)
        self._observed = True

    def compute_range(self) -> Tuple[float, float]:
        if not self._observed:
            return 0.0, 0.0
        return self.min_val, self.max_val

    def reset(self) -> None:
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self._observed = False


class MovingAverageCalibrator(Calibrator):
    """Calibrator that maintains exponential moving averages of batch min and max."""

    def __init__(self, momentum: float = 0.9) -> None:
        self.momentum: float = momentum
        self.min_val: float = 0.0
        self.max_val: float = 0.0
        self._observed: bool = False

    def update(self, data: Union[Tensor, np.ndarray]) -> None:
        arr = _extract_numpy(data)
        if arr.size == 0:
            return
        b_min = float(np.min(arr))
        b_max = float(np.max(arr))

        if not self._observed:
            self.min_val = b_min
            self.max_val = b_max
            self._observed = True
        else:
            self.min_val = self.momentum * self.min_val + (1.0 - self.momentum) * b_min
            self.max_val = self.momentum * self.max_val + (1.0 - self.momentum) * b_max

    def compute_range(self) -> Tuple[float, float]:
        if not self._observed:
            return 0.0, 0.0
        return self.min_val, self.max_val

    def reset(self) -> None:
        self.min_val = 0.0
        self.max_val = 0.0
        self._observed = False


class PercentileCalibrator(Calibrator):
    """Calibrator that estimates dynamic range using percentile clipping to reject extreme outliers."""

    def __init__(self, percentile: float = 99.99) -> None:
        self.percentile: float = percentile
        self._samples: List[np.ndarray] = []

    def update(self, data: Union[Tensor, np.ndarray]) -> None:
        arr = _extract_numpy(data)
        if arr.size > 0:
            self._samples.append(arr.ravel())

    def compute_range(self) -> Tuple[float, float]:
        if not self._samples:
            return 0.0, 0.0
        all_data = np.concatenate(self._samples)
        lower_p = (100.0 - self.percentile) / 2.0
        upper_p = 100.0 - lower_p
        min_val = float(np.percentile(all_data, lower_p))
        max_val = float(np.percentile(all_data, upper_p))
        return min_val, max_val

    def reset(self) -> None:
        self._samples.clear()


def calibrate_tensor(
    tensor: Union[Tensor, np.ndarray],
    scheme: str = "symmetric",
) -> Tuple[float, int]:
    """One-shot helper to compute quantization scale and zero_point from a tensor.

    Args:
        tensor: Floating-point Tensor or array.
        scheme: Quantization scheme ('symmetric' or 'asymmetric').

    Returns:
        Tuple of (scale, zero_point).
    """
    calibrator = MinMaxCalibrator()
    calibrator.update(tensor)
    return calibrator.compute_params(scheme=scheme)
