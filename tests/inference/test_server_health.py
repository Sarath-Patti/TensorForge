"""Tests for InferenceServer health API and model state monitoring."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestServerHealth(unittest.TestCase):

    def test_health_report_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("classifier", model_path)
                health = server.health()

                self.assertIn("status", health)
                self.assertIn("server_state", health)
                self.assertEqual(health["loaded_models_count"], 1)
                self.assertEqual(health["ready_models_count"], 1)
                self.assertIn("classifier:1", health["models"])


if __name__ == "__main__":
    unittest.main()
