"""Unit tests for statistics and performance metrics contracts in TensorForge v2.0."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceClient, InferenceServer
from tensorforge.serialization import save_model


class TestStatisticsContract(unittest.TestCase):
    """Test suite verifying server, client, and scheduler statistics contracts."""

    def test_statistics_and_snapshot_contracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("stat_model", model_path)
                with InferenceClient(server) as client:
                    _ = client.predict("stat_model", np.ones((1, 4), dtype=np.float32))

                    stats = client.stats()
                    self.assertEqual(stats["total_models"], 1)
                    self.assertEqual(stats["submitted_requests"], 1)
                    self.assertEqual(stats["completed_requests"], 1)
                    self.assertEqual(stats["tensorforge_version"], tf.__version__)

                    snapshot = client.performance_snapshot()
                    self.assertIn("server", snapshot)
                    self.assertIn("stats", snapshot["server"])
                    self.assertEqual(snapshot["server"]["stats"]["completed_requests"], 1)
                    self.assertIn("models", snapshot)
                    self.assertEqual(snapshot["tensorforge_version"], tf.__version__)


if __name__ == "__main__":
    unittest.main()
