"""Tests for quantization evaluation metrics."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.quantization import (
    compare_tensors,
    max_absolute_error,
    mean_absolute_error,
    mean_squared_error,
    quantization_snr,
    relative_error,
)


def test_error_metrics_identical_tensors():
    t1 = tf.tensor([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)
    t2 = tf.tensor([1.0, 2.0, 3.0, 4.0], dtype=tf.float32)

    assert max_absolute_error(t1, t2) == 0.0
    assert mean_absolute_error(t1, t2) == 0.0
    assert mean_squared_error(t1, t2) == 0.0
    assert relative_error(t1, t2) == 0.0
    assert quantization_snr(t1, t2) > 80.0  # High SNR for identical tensors


def test_known_error_metrics():
    orig = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    pert = np.array([1.1, 1.9, 3.2, 3.8], dtype=np.float32)  # diffs: +0.1, -0.1, +0.2, -0.2

    # max error: 0.2
    assert pytest.approx(max_absolute_error(orig, pert)) == 0.2
    # MAE: (0.1 + 0.1 + 0.2 + 0.2) / 4 = 0.15
    assert pytest.approx(mean_absolute_error(orig, pert)) == 0.15
    # MSE: (0.01 + 0.01 + 0.04 + 0.04) / 4 = 0.025
    assert pytest.approx(mean_squared_error(orig, pert)) == 0.025


def test_compare_tensors_dictionary():
    t1 = tf.tensor([1.0, 2.0], dtype=tf.float32)
    t2 = tf.tensor([1.05, 1.95], dtype=tf.float32)

    summary = compare_tensors(t1, t2)
    assert "max_abs_error" in summary
    assert "mean_abs_error" in summary
    assert "mean_sq_error" in summary
    assert "rel_error" in summary
    assert "sqnr_db" in summary
    assert summary["max_abs_error"] > 0.0
