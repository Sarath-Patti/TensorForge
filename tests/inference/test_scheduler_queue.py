"""Tests for bounded scheduler queue, backpressure, and SchedulerQueueFullError enforcement."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler, SchedulerConfig
from tensorforge.serialization import save_model
from tensorforge.utils.validation import SchedulerQueueFullError


class TestSchedulerQueue(unittest.TestCase):

    def test_queue_capacity_and_backpressure_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            # Configure a very small queue (e.g. max_queue_size=2)
            config = SchedulerConfig(
                max_batch_size=10,
                max_queue_size=2,
                batch_timeout_ms=100.0,
            )
            scheduler = InferenceScheduler(runtime, config=config)

            # Enqueue up to capacity
            fut1 = scheduler.submit(tf.randn((1, 8)))
            fut2 = scheduler.submit(tf.randn((1, 8)))

            # 3rd submission exceeds max_queue_size -> immediate rejection
            with self.assertRaises(SchedulerQueueFullError):
                scheduler.submit(tf.randn((1, 8)))

            # Flush queue so previous requests finish
            scheduler.flush()
            _ = fut1.result(timeout=2.0)
            _ = fut2.result(timeout=2.0)

            scheduler.close()
            runtime.close()

    def test_pending_request_cancellation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            config = SchedulerConfig(
                max_batch_size=10,
                max_queue_size=10,
                batch_timeout_ms=500.0,  # long timeout so request stays pending
            )
            scheduler = InferenceScheduler(runtime, config=config)

            fut = scheduler.submit(tf.randn((1, 4)))
            # Cancel while pending
            cancelled = fut.cancel()
            self.assertTrue(cancelled)
            self.assertTrue(fut.done())

            scheduler.close()
            runtime.close()


if __name__ == "__main__":
    unittest.main()
