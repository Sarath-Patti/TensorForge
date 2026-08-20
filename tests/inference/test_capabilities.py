"""Unit tests for capability and environment discovery in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceClient, InferenceRuntime, InferenceServer
from tensorforge.serialization import save_model


class TestCapabilities(unittest.TestCase):
    """Test suite verifying framework and server capability reporting."""

    def test_native_backend_capability_check(self):
        native_status = is_native_available()
        self.assertIsInstance(native_status, bool)

    def test_runtime_capabilities_summary(self):
        model = nn.Linear(4, 2)
        runtime = InferenceRuntime(model, {"name": "test_model"})
        summary = runtime.summary()

        self.assertIn("backend", summary)
        self.assertIn("num_threads", summary)
        self.assertIn("tensorforge_version", summary)
        self.assertEqual(summary["tensorforge_version"], tf.__version__)

    def test_server_and_client_capabilities_discovery(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("test_mod", model_path)
                with InferenceClient(server) as client:
                    health = client.health()
                    self.assertIn("status", health)
                    self.assertEqual(health["status"], "healthy")

                    models = client.models()
                    self.assertEqual(len(models), 1)
                    self.assertEqual(models[0]["name"], "test_mod")


if __name__ == "__main__":
    unittest.main()
