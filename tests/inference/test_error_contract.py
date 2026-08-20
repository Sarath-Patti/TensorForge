"""Unit tests for structured exception contracts in TensorForge v2.0."""

import os
import tempfile
import unittest

from tensorforge.inference import (
    CircuitBreakerOpenError,
    InferenceClient,
    InferenceServer,
    ModelNotFoundError,
    ServerClosedError,
    ServerError,
    TensorForgeInputError,
)
from tensorforge.serialization import save_model
import tensorforge.nn as nn


class TestErrorContract(unittest.TestCase):
    """Test suite verifying structured exception contracts across the inference stack."""

    def test_input_error_contract(self):
        with self.assertRaises(TensorForgeInputError):
            InferenceClient("invalid_server_object")

    def test_model_not_found_error_contract(self):
        with InferenceServer() as server:
            with InferenceClient(server) as client:
                with self.assertRaises(ModelNotFoundError):
                    client.predict("nonexistent_model", [1, 2, 3])

    def test_server_closed_error_contract(self):
        server = InferenceServer()
        server.close()
        with self.assertRaises(ServerClosedError):
            server.predict("model", [1, 2, 3])

    def test_exception_inheritance_hierarchy(self):
        self.assertTrue(issubclass(ModelNotFoundError, ServerError))
        self.assertTrue(issubclass(ServerClosedError, ServerError))
        self.assertTrue(issubclass(CircuitBreakerOpenError, ServerError))


if __name__ == "__main__":
    unittest.main()
