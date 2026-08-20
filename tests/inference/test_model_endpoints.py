"""Unit tests for ModelEndpoint dataclass and endpoint metadata discovery in TensorForge v2.0."""

import os
import tempfile
import unittest

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    HealthState,
    InferenceClient,
    InferenceServer,
    ModelEndpoint,
    ModelLifecycleState,
)
from tensorforge.serialization import save_model


class TestModelEndpoints(unittest.TestCase):
    """Test suite verifying ModelEndpoint dataclass and model discovery contract."""

    def test_model_endpoint_dataclass(self):
        endpoint = ModelEndpoint(
            name="classifier",
            active_version="1",
            versions=["1", "2"],
            state=ModelLifecycleState.READY,
            health_state=HealthState.HEALTHY,
        )
        d = endpoint.to_dict()
        self.assertEqual(d["name"], "classifier")
        self.assertEqual(d["active_version"], "1")
        self.assertEqual(d["versions"], ["1", "2"])
        self.assertEqual(d["state"], "READY")
        self.assertEqual(d["health_state"], "HEALTHY")

    def test_server_and_client_endpoint_discovery_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("detector", model_path, version="1")
                server.load_model("detector", model_path, version="2", active=True)

                with InferenceClient(server) as client:
                    endpoints = client.models()
                    self.assertEqual(len(endpoints), 2)

                    # Inspect primary fields on active version endpoint
                    active_ep = [e for e in endpoints if e["is_active"]][0]
                    self.assertEqual(active_ep["name"], "detector")
                    self.assertEqual(active_ep["active_version"], "2")
                    self.assertIn("1", active_ep["versions"])
                    self.assertIn("2", active_ep["versions"])
                    self.assertEqual(active_ep["state"], "READY")
                    self.assertEqual(active_ep["health_state"], "HEALTHY")


if __name__ == "__main__":
    unittest.main()
