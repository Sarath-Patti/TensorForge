"""Tests for InferenceServer performance snapshot aggregation and JSON export."""

import json
import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestServerObservability(unittest.TestCase):

    def test_performance_snapshot_and_export(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            json_export_path = os.path.join(tmpdir, "server_metrics.json")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("m1", model_path)
                _ = server.predict("m1", tf.randn((1, 4)))

                snapshot = server.performance_snapshot()
                self.assertIn("server", snapshot)
                self.assertIn("models", snapshot)
                self.assertIn("m1:1", snapshot["models"])
                self.assertEqual(snapshot["tensorforge_version"], "1.8.0")

                server.export_metrics(json_export_path)
                self.assertTrue(os.path.exists(json_export_path))

                with open(json_export_path, "r", encoding="utf-8") as f:
                    loaded_json = json.load(f)

                self.assertEqual(loaded_json["tensorforge_version"], "1.8.0")


if __name__ == "__main__":
    unittest.main()
