"""Tests for graceful scheduler shutdown, draining behavior, and idempotent close."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    SchedulerConfig,
    SchedulerLifecycleState,
)
from tensorforge.serialization import save_model
from tensorforge.utils.validation import SchedulerClosedError


class TestSchedulerShutdown(unittest.TestCase):

    def test_lifecycle_and_idempotent_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            scheduler = InferenceScheduler(runtime, max_batch_size=4, batch_timeout_ms=5.0)

            self.assertEqual(scheduler.lifecycle_state, SchedulerLifecycleState.RUNNING.value)
            self.assertTrue(scheduler.is_running)

            # Close scheduler
            scheduler.close()
            self.assertEqual(scheduler.lifecycle_state, SchedulerLifecycleState.CLOSED.value)
            self.assertTrue(scheduler.is_closed)

            # Repeated close is idempotent
            scheduler.close()
            self.assertTrue(scheduler.is_closed)

            # Submission after close raises SchedulerClosedError
            with self.assertRaises(SchedulerClosedError):
                scheduler.submit(tf.randn((1, 8)))

            with self.assertRaises(SchedulerClosedError):
                scheduler.predict(tf.randn((1, 8)))

            runtime.close()

    def test_draining_pending_requests_on_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            config = SchedulerConfig(
                max_batch_size=10,
                max_queue_size=10,
                batch_timeout_ms=100.0,
                drain_on_close=True,
            )
            scheduler = InferenceScheduler(runtime, config=config)

            fut1 = scheduler.submit(tf.randn((1, 8)))
            fut2 = scheduler.submit(tf.randn((1, 8)))

            # Close with drain=True: worker will process fut1 and fut2 before terminating
            scheduler.close(drain=True)

            out1 = fut1.result(timeout=2.0)
            out2 = fut2.result(timeout=2.0)

            self.assertEqual(out1.shape, (1, 2))
            self.assertEqual(out2.shape, (1, 2))

            runtime.close()


if __name__ == "__main__":
    unittest.main()
