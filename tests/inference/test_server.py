"""Tests for InferenceServer construction, configuration, and basic functionality."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer, ServerConfig, ServerLifecycleState
from tensorforge.serialization import save_model


class TestServer(unittest.TestCase):

    def test_server_construction_and_lifecycle(self):
        server = InferenceServer(config=ServerConfig(auto_start=True))
        self.assertEqual(server.state, ServerLifecycleState.RUNNING)
        server.close()
        self.assertEqual(server.state, ServerLifecycleState.CLOSED)

    def test_server_context_manager(self):
        with InferenceServer() as server:
            self.assertEqual(server.state, ServerLifecycleState.RUNNING)
        self.assertEqual(server.state, ServerLifecycleState.CLOSED)

    def test_load_and_predict_single_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            with InferenceServer() as server:
                entry = server.load_model(name="classifier", path=model_path, version="1")
                self.assertEqual(entry.name, "classifier")
                self.assertEqual(entry.version, "1")

                x = tf.randn((2, 8))
                out = server.predict("classifier", x)
                self.assertEqual(out.shape, (2, 4))


if __name__ == "__main__":
    unittest.main()
