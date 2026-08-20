"""Unit tests for InferenceClient and InferenceRequestContract in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceClient,
    InferenceFuture,
    InferenceRequestContract,
    InferenceServer,
)
from tensorforge.serialization import save_model


class TestClientAPI(unittest.TestCase):
    """Test suite verifying InferenceClient prediction, submission, batching, and health discovery."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.tmpdir.name, "model.tfmodel")
        model = nn.Linear(4, 2)
        save_model(model, self.model_path)

        self.server = InferenceServer()
        self.server.load_model("classifier", self.model_path, version="1")
        self.client = InferenceClient(self.server)

    def tearDown(self):
        self.client.close()
        self.tmpdir.cleanup()

    def test_client_synchronous_predict(self):
        inp = np.ones((1, 4), dtype=np.float32)
        out = self.client.predict("classifier", inp)
        self.assertIsInstance(out, tf.Tensor)
        self.assertEqual(out.shape, (1, 2))

    def test_client_execute_contract(self):
        contract = InferenceRequestContract(
            model="classifier",
            inputs=np.ones((1, 4), dtype=np.float32),
            version="1",
        )
        out = self.client.execute_contract(contract)
        self.assertIsInstance(out, tf.Tensor)
        self.assertEqual(out.shape, (1, 2))

    def test_client_async_submit(self):
        inp = np.ones((1, 4), dtype=np.float32)
        future = self.client.submit("classifier", inp)
        self.assertIsInstance(future, InferenceFuture)
        out = future.result()
        self.assertEqual(out.shape, (1, 2))

    def test_client_batch_prediction(self):
        inputs_list = [np.ones((1, 4), dtype=np.float32) for _ in range(3)]
        results = self.client.predict_batch("classifier", inputs_list)
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertEqual(res.shape, (1, 2))

    def test_client_discovery_methods(self):
        health = self.client.health()
        self.assertIn("status", health)
        self.assertEqual(health["status"], "healthy")

        models = self.client.models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "classifier")

        snapshot = self.client.performance_snapshot()
        self.assertIn("server", snapshot)
        self.assertEqual(snapshot["tensorforge_version"], tf.__version__)

    def test_client_context_manager(self):
        with InferenceClient(self.server) as client:
            res = client.predict("classifier", np.ones((1, 4), dtype=np.float32))
            self.assertEqual(res.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
