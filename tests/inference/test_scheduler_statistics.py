"""Tests for scheduler health and statistics telemetry reporting."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler
from tensorforge.serialization import save_model


class TestSchedulerStatistics(unittest.TestCase):

    def test_health_and_stats_reporting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            with InferenceScheduler(runtime, max_batch_size=4, batch_timeout_ms=5.0) as scheduler:
                # Initial health
                h = scheduler.health()
                self.assertEqual(h["status"], "healthy")
                self.assertEqual(h["lifecycle_state"], "RUNNING")
                self.assertTrue(h["accepting_requests"])
                self.assertEqual(h["requests_submitted"], 0)

                # Process 4 predictions
                for _ in range(4):
                    _ = scheduler.predict(tf.randn((1, 8)))

                stats = scheduler.stats()
                self.assertEqual(stats["submitted_requests"], 4)
                self.assertEqual(stats["completed_requests"], 4)
                self.assertEqual(stats["failed_requests"], 0)
                self.assertEqual(stats["tensorforge_version"], tf.__version__)
                self.assertIn("config", stats)
                self.assertIn("health", stats)

            runtime.close()


if __name__ == "__main__":
    unittest.main()
