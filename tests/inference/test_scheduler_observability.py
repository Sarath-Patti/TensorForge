"""Tests for InferenceScheduler observability integration and snapshot generation."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    PerformanceSnapshot,
    SchedulerConfig,
)
from tensorforge.serialization import save_model


class TestSchedulerObservability(unittest.TestCase):

    def test_scheduler_performance_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            config = SchedulerConfig(max_batch_size=8, max_queue_size=32, batch_timeout_ms=5.0)

            with InferenceScheduler(runtime, config=config) as scheduler:
                # Dispatch 5 requests
                for _ in range(5):
                    _ = scheduler.predict(tf.randn((2, 8)))

                snapshot = scheduler.performance_snapshot()
                self.assertIsInstance(snapshot, PerformanceSnapshot)
                self.assertEqual(snapshot.requests.completed, 5)
                self.assertGreater(snapshot.batches.batches_formed, 0)
                self.assertEqual(snapshot.batches.samples_processed, 10)
                self.assertIsNotNone(snapshot.scheduler)
                self.assertEqual(snapshot.scheduler.max_batch_size, 8)
                self.assertEqual(snapshot.scheduler.max_queue_size, 32)

            runtime.close()


if __name__ == "__main__":
    unittest.main()
