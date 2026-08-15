"""NumPy reference compute backend for TensorForge."""

from __future__ import annotations

import numpy as np


def numpy_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute element-wise addition via NumPy."""
    return np.add(a, b)


def numpy_sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute element-wise subtraction via NumPy."""
    return np.subtract(a, b)


def numpy_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute element-wise multiplication via NumPy."""
    return np.multiply(a, b)


def numpy_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Execute matrix multiplication via NumPy."""
    return np.matmul(a, b)
