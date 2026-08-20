"""Unit tests for InferenceRequestContract in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceClient,
    InferenceRequest,
    InferenceRequestContract,
    InferenceServer,
    RetryConfig,
)
from tensorforge.serialization import save_model


class TestRequestContract(unittest.TestCase):
    """Test suite verifying InferenceRequestContract structure and execution."""

    def test_request_contract_dataclass(self):
        contract = InferenceRequestContract(
            model="classifier",
            inputs=np.ones((1, 4), dtype=np.float32),
            version="1",
            timeout_ms=100.0,
            retry_config=RetryConfig(max_retries=2),
        )
        self.assertEqual(contract.model, "classifier")
        self.assertEqual(contract.version, "1")
        self.assertEqual(contract.timeout_ms, 100.0)
        self.assertIsNotNone(contract.retry_config)

    def test_request_contract_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("classifier", model_path)
                with InferenceClient(server) as client:
                    contract = InferenceRequestContract(
                        model="classifier",
                        inputs=np.ones((1, 4), dtype=np.float32),
                    )
                    out = client.execute_contract(contract)
                    self.assertIsInstance(out, tf.Tensor)
                    self.assertEqual(out.shape, (1, 2))

    def test_inference_request_internal_structure(self):
        req = InferenceRequest(
            request_id="req-123",
            input_tensor=tf.ones((1, 4)),
            timeout_ms=50.0,
        )
        self.assertEqual(req.request_id, "req-123")
        self.assertFalse(req.is_cancelled)


if __name__ == "__main__":
    unittest.main()
