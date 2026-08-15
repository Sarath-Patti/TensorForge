"""Tests for dynamic range calibrators."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.quantization.calibration import (
    MinMaxCalibrator,
    MovingAverageCalibrator,
    PercentileCalibrator,
    calibrate_tensor,
)


def test_min_max_calibrator():
    calibrator = MinMaxCalibrator()
    batch1 = tf.tensor([-1.0, 2.0, 3.0], dtype=tf.float32)
    batch2 = tf.tensor([-5.0, 1.0, 4.0], dtype=tf.float32)

    calibrator.update(batch1)
    calibrator.update(batch2)

    min_v, max_v = calibrator.compute_range()
    assert min_v == -5.0
    assert max_v == 4.0

    scale, zp = calibrator.compute_params(scheme="symmetric")
    assert scale == 5.0 / 127.0
    assert zp == 0


def test_moving_average_calibrator():
    calibrator = MovingAverageCalibrator(momentum=0.5)
    batch1 = np.array([-10.0, 10.0], dtype=np.float32)
    batch2 = np.array([-20.0, 20.0], dtype=np.float32)

    calibrator.update(batch1)
    # After batch1: min = -10, max = 10
    min_v, max_v = calibrator.compute_range()
    assert min_v == -10.0
    assert max_v == 10.0

    calibrator.update(batch2)
    # After batch2: min = 0.5 * (-10) + 0.5 * (-20) = -15
    #               max = 0.5 * (10) + 0.5 * (20) = 15
    min_v, max_v = calibrator.compute_range()
    assert min_v == -15.0
    assert max_v == 15.0


def test_percentile_calibrator():
    calibrator = PercentileCalibrator(percentile=90.0)
    # Normal distribution with extreme outliers
    data = np.linspace(-10.0, 10.0, 1000, dtype=np.float32)
    data[0] = -1000.0  # outlier
    data[-1] = 1000.0  # outlier

    calibrator.update(data)
    min_v, max_v = calibrator.compute_range()
    # Percentile range should filter out +/- 1000 outliers
    assert min_v > -50.0
    assert max_v < 50.0


def test_calibrate_tensor_helper():
    t = tf.tensor([-2.54, 2.54], dtype=tf.float32)
    scale, zp = calibrate_tensor(t, scheme="symmetric")
    assert pytest.approx(scale, rel=1e-4) == 2.54 / 127.0
    assert zp == 0
