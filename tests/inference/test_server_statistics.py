"""Tests for InferenceServer aggregate and per-model statistics reporting."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestServerStatistics(unittest.TestCase):

    def test_stats_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("m1", model_path)
                _ = server.predict("m1", tf.randn((1, 4)))

                stats = server.stats()
                self.assertEqual(stats["total_models"], 1)
                self.assertEqual(stats["submitted_requests"], 1)
                self.assertEqual(stats["completed_requests"], 1)
                self.assertEqual(stats["tensorforge_version"], tf.__version__)
                self.assertIn("m1:1", stats["models"])


if __name__ == "__main__":
    unittest.main()
