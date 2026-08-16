"""Tests for CPU thread pool configuration and execution."""

import pytest
from tensorforge.backend import get_num_threads, is_native_available, set_num_threads
from tensorforge.utils.validation import TensorForgeError


def test_thread_configuration():
    initial_threads = get_num_threads()
    assert initial_threads >= 1

    set_num_threads(2)
    assert get_num_threads() == 2

    set_num_threads(4)
    assert get_num_threads() == 4

    set_num_threads(1)
    assert get_num_threads() == 1

    # Restore initial
    set_num_threads(initial_threads)


def test_thread_configuration_invalid_raises():
    with pytest.raises(TensorForgeError, match="num_threads must be a positive integer"):
        set_num_threads(0)

    with pytest.raises(TensorForgeError, match="num_threads must be a positive integer"):
        set_num_threads(-2)


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_native_thread_pool_binding():
    import _tensorforge_native as _native

    assert hasattr(_native, "set_num_threads")
    assert hasattr(_native, "get_num_threads")

    _native.set_num_threads(3)
    assert _native.get_num_threads() == 3

    _native.set_num_threads(1)
    assert _native.get_num_threads() == 1
