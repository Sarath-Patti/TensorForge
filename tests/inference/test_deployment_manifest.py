"""Unit tests for DeploymentManifest and InferenceServer bootstrapping in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    DeploymentManifest,
    InferenceServer,
    ModelDeploymentSpec,
    ServerConfig,
)
from tensorforge.serialization import save_model


class TestDeploymentManifest(unittest.TestCase):
    """Test suite verifying DeploymentManifest JSON export/import and server bootstrapping."""

    def test_manifest_dict_and_json_roundtrip(self):
        spec = ModelDeploymentSpec(
            name="test_model",
            path="/tmp/model.tfmodel",
            version="1",
            profile_type="LOW_LATENCY",
            active=True,
        )
        manifest = DeploymentManifest(
            name="test_deployment",
            server_config=ServerConfig(max_loaded_models=5),
            models=[spec],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "manifest.json")
            manifest.save_json(json_file)

            loaded_manifest = DeploymentManifest.load_json(json_file)
            self.assertEqual(loaded_manifest.name, "test_deployment")
            self.assertEqual(loaded_manifest.server_config.max_loaded_models, 5)
            self.assertEqual(len(loaded_manifest.models), 1)
            self.assertEqual(loaded_manifest.models[0].name, "test_model")
            self.assertEqual(loaded_manifest.models[0].profile_type, "LOW_LATENCY")

    def test_server_bootstrapping_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            spec = ModelDeploymentSpec(
                name="model_a",
                path=model_path,
                version="1",
                profile_type="BALANCED",
            )
            manifest = DeploymentManifest(models=[spec])

            with InferenceServer.from_manifest(manifest) as server:
                health = server.health()
                self.assertEqual(health["status"], "healthy")
                self.assertIn("model_a", health["models"])

                out = server.predict("model_a", np.ones((1, 4), dtype=np.float32))
                self.assertEqual(out.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
