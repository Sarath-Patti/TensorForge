"""Tests for multi-threaded concurrent request submissions to InferenceScheduler."""

import concurrent.futures
import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler
from tensorforge.serialization import save_model


class TestSchedulerConcurrency(unittest.TestCase):

    def test_concurrent_producers_dynamic_batching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            with InferenceScheduler(runtime, max_batch_size=16, batch_timeout_ms=5.0) as scheduler:

                def worker(i: int):
                    x = tf.randn((1, 16))
                    out = scheduler.predict(x)
                    return out.shape

                # 30 concurrent producer threads submitting requests simultaneously
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(worker, i) for i in range(30)]
                    results = [f.result() for f in futures]

                for shape in results:
                    self.assertEqual(shape, (1, 4))

                stats = scheduler.stats()
                self.assertEqual(stats["completed_requests"], 30)
                self.assertEqual(stats["failed_requests"], 0)
                self.assertGreater(stats["batches_formed"], 0)

            runtime.close()


if __name__ == "__main__":
    unittest.main()
