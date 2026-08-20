"""Unit tests for InferenceClient in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceClient,
    InferenceFuture,
    InferenceServer,
)
from tensorforge.serialization import save_model


class TestClient(unittest.TestCase):
    """Test suite verifying InferenceClient operations."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.model_path = os.path.join(self.tmpdir.name, "model.tfmodel")
        save_model(nn.Linear(4, 2), self.model_path)
        self.server = InferenceServer()
        self.server.load_model("classifier", self.model_path)
        self.client = InferenceClient(self.server)

    def tearDown(self):
        self.client.close()
        self.tmpdir.cleanup()

    def test_client_predict(self):
        out = self.client.predict("classifier", np.ones((1, 4), dtype=np.float32))
        self.assertIsInstance(out, tf.Tensor)
        self.assertEqual(out.shape, (1, 2))

    def test_client_predict_batch(self):
        inputs = [np.ones((1, 4), dtype=np.float32) for _ in range(3)]
        results = self.client.predict_batch("classifier", inputs)
        self.assertEqual(len(results), 3)
        for res in results:
            self.assertEqual(res.shape, (1, 2))

    def test_client_submit(self):
        future = self.client.submit("classifier", np.ones((1, 4), dtype=np.float32))
        self.assertIsInstance(future, InferenceFuture)
        res = future.result()
        self.assertEqual(res.shape, (1, 2))

    def test_client_health_and_stats(self):
        health = self.client.health()
        self.assertEqual(health["status"], "healthy")

        stats = self.client.stats()
        self.assertIn("submitted_requests", stats)

    def test_client_models(self):
        models = self.client.models()
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "classifier")

    def test_client_performance_snapshot(self):
        snapshot = self.client.performance_snapshot()
        self.assertIn("server", snapshot)

    def test_client_context_manager(self):
        with InferenceClient(self.server) as client:
            res = client.predict("classifier", np.ones((1, 4), dtype=np.float32))
            self.assertEqual(res.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
