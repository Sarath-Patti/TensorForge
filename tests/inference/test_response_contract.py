"""Unit tests for inference response contracts in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceClient, InferenceFuture, InferenceServer
from tensorforge.serialization import save_model


class TestResponseContract(unittest.TestCase):
    """Test suite verifying return types and response structures for sync, async, and batch requests."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.tmpdir.name, "model.tfmodel")
        save_model(nn.Linear(4, 2), self.model_path)
        self.server = InferenceServer()
        self.server.load_model("model_a", self.model_path)
        self.client = InferenceClient(self.server)

    def tearDown(self):
        self.client.close()
        self.tmpdir.cleanup()

    def test_sync_response_contract(self):
        output = self.client.predict("model_a", np.ones((1, 4), dtype=np.float32))
        self.assertIsInstance(output, tf.Tensor)
        self.assertEqual(output.shape, (1, 2))

    def test_async_future_response_contract(self):
        future = self.client.submit("model_a", np.ones((1, 4), dtype=np.float32))
        self.assertIsInstance(future, InferenceFuture)
        output = future.result(timeout=2.0)
        self.assertIsInstance(output, tf.Tensor)

    def test_batch_response_contract(self):
        inputs = [np.ones((1, 4), dtype=np.float32) for _ in range(4)]
        outputs = self.client.predict_batch("model_a", inputs)
        self.assertIsInstance(outputs, list)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertIsInstance(out, tf.Tensor)


if __name__ == "__main__":
    unittest.main()
