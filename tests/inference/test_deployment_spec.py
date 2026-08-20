"""Unit tests for DeploymentSpec, DeploymentManifest, and server bootstrapping in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    DeploymentManifest,
    DeploymentSpec,
    InferenceServer,
    ModelDeploymentSpec,
    ServerConfig,
)
from tensorforge.serialization import save_model


class TestDeploymentSpec(unittest.TestCase):
    """Test suite verifying DeploymentSpec, DeploymentManifest, and server bootstrapping."""

    def test_deployment_spec_alias_parity(self):
        self.assertIs(DeploymentSpec, ModelDeploymentSpec)

    def test_deployment_spec_to_dict_and_from_dict(self):
        spec = DeploymentSpec(
            name="test_mod",
            path="/tmp/test.tfmodel",
            version="2",
            profile_type="LOW_LATENCY",
            active=True,
            metadata={"owner": "team_a"},
        )
        d = spec.to_dict()
        self.assertEqual(d["name"], "test_mod")
        self.assertEqual(d["version"], "2")
        self.assertEqual(d["profile_type"], "LOW_LATENCY")
        self.assertEqual(d["metadata"]["owner"], "team_a")

        reconstructed = DeploymentSpec.from_dict(d)
        self.assertEqual(reconstructed.name, spec.name)
        self.assertEqual(reconstructed.version, spec.version)

    def test_deployment_manifest_roundtrip_and_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            spec = DeploymentSpec(
                name="model_1",
                path=model_path,
                version="1",
                profile_type="BALANCED",
            )
            manifest = DeploymentManifest(
                name="cluster_1",
                server_config=ServerConfig(max_loaded_models=4),
                models=[spec],
            )

            manifest_file = os.path.join(tmpdir, "manifest.json")
            manifest.save_json(manifest_file)

            # Test bootstrapping from manifest file
            with InferenceServer.from_manifest(manifest_file) as server:
                h = server.health()
                self.assertEqual(h["status"], "healthy")
                self.assertEqual(h["loaded_models_count"], 1)

                out = server.predict("model_1", np.ones((1, 4), dtype=np.float32))
                self.assertEqual(out.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
