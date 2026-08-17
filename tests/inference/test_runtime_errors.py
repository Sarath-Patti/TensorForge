"""Tests verifying exception hierarchy, error messages, and inheritance relationships."""

import unittest
from tensorforge.utils.validation import (
    RuntimeBusyError,
    RuntimeClosedError,
    RuntimeLimitError,
    RuntimeResourceError,
    RuntimeStateError,
    RuntimeTimeoutError,
    TensorForgeError,
    TensorForgeInputError,
)


class TestRuntimeErrors(unittest.TestCase):

    def test_exception_inheritance_hierarchy(self):
        # RuntimeStateError & RuntimeClosedError
        self.assertTrue(issubclass(RuntimeStateError, TensorForgeError))
        self.assertTrue(issubclass(RuntimeStateError, RuntimeError))
        self.assertTrue(issubclass(RuntimeClosedError, RuntimeStateError))

        # TensorForgeInputError
        self.assertTrue(issubclass(TensorForgeInputError, TensorForgeError))
        self.assertTrue(issubclass(TensorForgeInputError, ValueError))

        # RuntimeLimitError & RuntimeBusyError
        self.assertTrue(issubclass(RuntimeLimitError, TensorForgeError))
        self.assertTrue(issubclass(RuntimeLimitError, ValueError))
        self.assertTrue(issubclass(RuntimeBusyError, RuntimeLimitError))
        self.assertTrue(issubclass(RuntimeBusyError, RuntimeError))

        # RuntimeResourceError
        self.assertTrue(issubclass(RuntimeResourceError, TensorForgeError))
        self.assertTrue(issubclass(RuntimeResourceError, RuntimeError))

        # RuntimeTimeoutError
        self.assertTrue(issubclass(RuntimeTimeoutError, TensorForgeError))
        self.assertTrue(issubclass(RuntimeTimeoutError, TimeoutError))


if __name__ == "__main__":
    unittest.main()
