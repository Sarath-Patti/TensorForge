"""Evaluation metrics for measuring quantization fidelity and error."""

from __future__ import annotations

from typing import Dict, Union
import numpy as np

from tensorforge.tensor.tensor import Tensor


def _to_float32_array(val: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Helper to convert input to flat float32 NumPy array."""
    if isinstance(val, Tensor):
        return val.numpy().astype(np.float32)
    return np.asarray(val, dtype=np.float32)


def max_absolute_error(original: Union[Tensor, np.ndarray], dequantized: Union[Tensor, np.ndarray]) -> float:
    """Compute the maximum absolute error: max(|original - dequantized|)."""
    orig_arr = _to_float32_array(original)
    deq_arr = _to_float32_array(dequantized)
    return float(np.max(np.abs(orig_arr - deq_arr)))


def mean_absolute_error(original: Union[Tensor, np.ndarray], dequantized: Union[Tensor, np.ndarray]) -> float:
    """Compute the mean absolute error (MAE): mean(|original - dequantized|)."""
    orig_arr = _to_float32_array(original)
    deq_arr = _to_float32_array(dequantized)
    return float(np.mean(np.abs(orig_arr - deq_arr)))


def mean_squared_error(original: Union[Tensor, np.ndarray], dequantized: Union[Tensor, np.ndarray]) -> float:
    """Compute the mean squared error (MSE): mean((original - dequantized)^2)."""
    orig_arr = _to_float32_array(original)
    deq_arr = _to_float32_array(dequantized)
    return float(np.mean((orig_arr - deq_arr) ** 2))


def relative_error(
    original: Union[Tensor, np.ndarray],
    dequantized: Union[Tensor, np.ndarray],
    eps: float = 1e-8,
) -> float:
    """Compute normalized L2 relative error: ||original - dequantized||_2 / (||original||_2 + eps)."""
    orig_arr = _to_float32_array(original)
    deq_arr = _to_float32_array(dequantized)
    diff_norm = np.linalg.norm(orig_arr - deq_arr)
    orig_norm = np.linalg.norm(orig_arr)
    return float(diff_norm / (orig_norm + eps))


def quantization_snr(
    original: Union[Tensor, np.ndarray],
    dequantized: Union[Tensor, np.ndarray],
    eps: float = 1e-8,
) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB):
    10 * log10(sum(original^2) / (sum((original - dequantized)^2) + eps)).
    """
    orig_arr = _to_float32_array(original)
    deq_arr = _to_float32_array(dequantized)
    signal_power = np.sum(orig_arr ** 2)
    noise_power = np.sum((orig_arr - deq_arr) ** 2)
    return float(10.0 * np.log10(signal_power / (noise_power + eps)))


def compare_tensors(
    original: Union[Tensor, np.ndarray],
    dequantized: Union[Tensor, np.ndarray],
) -> Dict[str, float]:
    """Compute a comprehensive dictionary of all quantization error metrics.

    Returns:
        Dictionary with keys: 'max_abs_error', 'mean_abs_error', 'mean_sq_error', 'rel_error', 'sqnr_db'.
    """
    return {
        "max_abs_error": max_absolute_error(original, dequantized),
        "mean_abs_error": mean_absolute_error(original, dequantized),
        "mean_sq_error": mean_squared_error(original, dequantized),
        "rel_error": relative_error(original, dequantized),
        "sqnr_db": quantization_snr(original, dequantized),
    }
