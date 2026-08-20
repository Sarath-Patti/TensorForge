"""Tests for metrics reset functionality across collector, runtime, and scheduler."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler, MetricsCollector
from tensorforge.serialization import save_model


class TestMetricsReset(unittest.TestCase):

    def test_metrics_collector_reset(self):
        collector = MetricsCollector()
        collector.record_request_submitted()
        collector.record_request_completed(queue_wait_ms=1.0, exec_ms=2.0, e2e_ms=3.0, samples=1)
        collector.record_batch(batch_size=4)

        snapshot1 = collector.snapshot()
        self.assertEqual(snapshot1.requests.completed, 1)
        self.assertEqual(snapshot1.batches.batches_formed, 1)

        collector.reset()
        snapshot2 = collector.snapshot()
        self.assertEqual(snapshot2.requests.completed, 0)
        self.assertEqual(snapshot2.batches.batches_formed, 0)
        self.assertEqual(snapshot2.latency.execution.sample_count, 0)

    def test_runtime_metrics_reset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            _ = runtime.predict(tf.randn((1, 4)))
            self.assertEqual(runtime.performance_snapshot().requests.completed, 1)

            runtime.reset_metrics()
            self.assertEqual(runtime.performance_snapshot().requests.completed, 0)

            runtime.close()


if __name__ == "__main__":
    unittest.main()
