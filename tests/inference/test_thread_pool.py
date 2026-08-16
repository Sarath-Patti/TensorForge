"""Tests for CPU thread pool configuration and execution."""

import unittest
from tensorforge.backend import get_num_threads, is_native_available, set_num_threads
from tensorforge.utils.validation import TensorForgeError


class TestThreadPool(unittest.TestCase):

    def test_thread_configuration(self):
        initial_threads = get_num_threads()
        self.assertGreaterEqual(initial_threads, 1)

        set_num_threads(2)
        self.assertEqual(get_num_threads(), 2)

        set_num_threads(4)
        self.assertEqual(get_num_threads(), 4)

        set_num_threads(1)
        self.assertEqual(get_num_threads(), 1)

        # Restore initial
        set_num_threads(initial_threads)

    def test_thread_configuration_invalid_raises(self):
        with self.assertRaises(TensorForgeError):
            set_num_threads(0)

        with self.assertRaises(TensorForgeError):
            set_num_threads(-2)

    def test_native_thread_pool_binding(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        try:
            import _tensorforge_native as _native
        except ImportError:
            from tensorforge import _tensorforge_native as _native

        self.assertTrue(hasattr(_native, "set_num_threads"))
        self.assertTrue(hasattr(_native, "get_num_threads"))

        _native.set_num_threads(3)
        self.assertEqual(_native.get_num_threads(), 3)

        _native.set_num_threads(1)
        self.assertEqual(_native.get_num_threads(), 1)


if __name__ == "__main__":
    unittest.main()
