"""Tests for Backend Dispatcher and execution context."""

import pytest
import tensorforge as tf
from tensorforge.backend import (
    backend_context,
    get_backend,
    get_last_backend,
    is_native_available,
    set_backend,
)
from tensorforge.utils.validation import TensorForgeError


def test_default_backend():
    # NumPy must always be the default backend
    assert get_backend() == "numpy"


def test_set_backend_numpy():
    set_backend("numpy")
    assert get_backend() == "numpy"


def test_invalid_backend_raises_error():
    with pytest.raises(TensorForgeError, match="Invalid backend"):
        set_backend("invalid_backend_xyz")


def test_backend_context_manager():
    set_backend("numpy")
    assert get_backend() == "numpy"

    if is_native_available():
        with backend_context("native"):
            assert get_backend() == "native"
        assert get_backend() == "numpy"
    else:
        with pytest.raises(TensorForgeError):
            with backend_context("native"):
                pass


def test_last_backend_recording():
    set_backend("numpy")
    a = tf.tensor([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.tensor([4.0, 5.0, 6.0], dtype=tf.float32)
    _ = a + b
    assert get_last_backend() == "numpy"
