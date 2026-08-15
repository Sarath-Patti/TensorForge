"""Tests for QuantizedTensor and quantization/dequantization operations."""

import numpy as np
import pytest
import tensorforge as tf
from tensorforge.quantization import QuantizedTensor, dequantize, quantize, quantize_tensor
from tensorforge.utils.validation import QuantizationError


def test_symmetric_quantization_roundtrip():
    # Symmetric quantization: zero_point must be 0
    np_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    t = tf.tensor(np_data, dtype=tf.float32)

    q = quantize(t, scheme="symmetric")
    assert isinstance(q, QuantizedTensor)
    assert q.scheme == "symmetric"
    assert q.zero_point == 0
    assert q.dtype == tf.int8
    assert q.nbytes == len(np_data) * 1  # 1 byte per int8 element
    assert q.shape == (5,)

    deq = dequantize(q)
    assert isinstance(deq, tf.Tensor)
    assert deq.shape == (5,)
    assert deq.dtype == tf.float32

    # MAE for 5 values on [-2, 2] should be <= scale / 2 = (2/127)/2 ≈ 0.008
    np.testing.assert_allclose(deq.numpy(), np_data, atol=0.02)


def test_asymmetric_quantization_roundtrip():
    # Asymmetric affine quantization on positive interval [0.0, 10.0]
    np_data = np.linspace(0.0, 10.0, 100, dtype=np.float32)
    t = tf.tensor(np_data, dtype=tf.float32)

    q = quantize(t, scheme="asymmetric")
    assert q.scheme == "asymmetric"
    assert q.dtype == tf.int8
    assert q.shape == (100,)

    deq = dequantize(q)
    # Maximum error is bounded by scale = 10 / 255 ≈ 0.04
    np.testing.assert_allclose(deq.numpy(), np_data, atol=0.05)


def test_constant_tensor_quantization():
    # Tensor with identical values (x_min == x_max)
    t_const = tf.ones((4, 4), dtype=tf.float32) * 5.0
    q = quantize(t_const, scheme="symmetric")
    deq = dequantize(q)
    np.testing.assert_allclose(deq.numpy(), 5.0, atol=0.05)


def test_zero_tensor_quantization():
    # All-zero tensor
    t_zeros = tf.zeros((3, 3), dtype=tf.float32)
    q = quantize(t_zeros, scheme="symmetric")
    assert q.scale == 1.0
    assert q.zero_point == 0
    deq = dequantize(q)
    np.testing.assert_allclose(deq.numpy(), 0.0)


def test_clamping_behavior():
    # Outliers should be safely clamped to [-128, 127]
    np_data = np.array([-1000.0, 1000.0], dtype=np.float32)
    q = quantize_tensor(np_data, scale=1.0, zero_point=0, scheme="symmetric")
    q_arr = q.numpy()
    assert q_arr[0] == -128
    assert q_arr[1] == 127


def test_invalid_scale_raises_error():
    t = tf.tensor([1.0, 2.0], dtype=tf.float32)
    with pytest.raises(QuantizationError):
        quantize_tensor(t, scale=-0.5)
    with pytest.raises(QuantizationError):
        quantize_tensor(t, scale=0.0)


def test_quantized_tensor_copy():
    t = tf.tensor([1.0, 2.0, 3.0], dtype=tf.float32)
    q = quantize(t)
    q_copy = q.copy()
    assert q_copy.scale == q.scale
    assert q_copy.zero_point == q.zero_point
    np.testing.assert_array_equal(q.numpy(), q_copy.numpy())
