"""Unit tests for health monitoring contracts in TensorForge v2.0."""

import os
import tempfile
import unittest

import tensorforge.nn as nn
from tensorforge.inference import (
    HealthState,
    InferenceClient,
    InferenceRuntime,
    InferenceServer,
)
from tensorforge.serialization import save_model


class TestHealthContract(unittest.TestCase):
    """Test suite verifying HealthState enum and health reporting contracts."""

    def test_health_state_enum(self):
        self.assertEqual(HealthState.HEALTHY.value, "HEALTHY")
        self.assertEqual(HealthState.DEGRADED.value, "DEGRADED")
        self.assertEqual(HealthState.UNHEALTHY.value, "UNHEALTHY")

    def test_runtime_health_reporting(self):
        model = nn.Linear(4, 2)
        runtime = InferenceRuntime(model, {"name": "test_model"})
        h = runtime.health()
        self.assertEqual(h["status"], "healthy")
        self.assertEqual(h["prediction_count"], 0)
        self.assertEqual(h["error_count"], 0)

    def test_server_and_client_health_reporting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("model_x", model_path)
                with InferenceClient(server) as client:
                    h = client.health()
                    self.assertEqual(h["status"], "healthy")
                    self.assertEqual(h["server_state"], "RUNNING")
                    self.assertEqual(h["loaded_models_count"], 1)
                    self.assertIn("model_x:1", h["models"])


if __name__ == "__main__":
    unittest.main()
