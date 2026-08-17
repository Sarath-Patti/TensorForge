"""Tests for basic InferenceScheduler construction, configuration, and synchronous prediction."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceRuntime,
    InferenceScheduler,
    SchedulerConfig,
    SchedulingPolicy,
)
from tensorforge.serialization import save_model


class TestScheduler(unittest.TestCase):

    def test_scheduler_construction_and_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            config = SchedulerConfig(
                max_batch_size=16,
                max_queue_size=64,
                batch_timeout_ms=5.0,
                policy=SchedulingPolicy.FIFO,
            )
            scheduler = InferenceScheduler(runtime, config=config)

            self.assertEqual(scheduler.config.max_batch_size, 16)
            self.assertEqual(scheduler.config.max_queue_size, 64)
            self.assertEqual(scheduler.config.batch_timeout_ms, 5.0)
            self.assertEqual(scheduler.config.policy, SchedulingPolicy.FIFO)
            self.assertTrue(scheduler.is_running)
            self.assertFalse(scheduler.is_closed)

            scheduler.close()
            runtime.close()

    def test_synchronous_predict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            with InferenceScheduler(runtime, max_batch_size=8, batch_timeout_ms=2.0) as scheduler:
                x = tf.randn((2, 8))
                out = scheduler.predict(x)
                self.assertEqual(out.shape, (2, 2))

            runtime.close()

    def test_asynchronous_submit_and_future(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            with InferenceScheduler(runtime, max_batch_size=4, batch_timeout_ms=5.0) as scheduler:
                future = scheduler.submit(tf.randn((1, 4)))
                self.assertTrue(isinstance(future.request_id, str))
                out = future.result(timeout=2.0)
                self.assertEqual(out.shape, (1, 2))
                self.assertTrue(future.done())
                self.assertIsNone(future.exception())

            runtime.close()


if __name__ == "__main__":
    unittest.main()
