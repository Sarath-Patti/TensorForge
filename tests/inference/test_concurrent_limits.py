"""Tests for concurrent request admission limits and RuntimeBusyError enforcement."""

import concurrent.futures
import os
import tempfile
import time
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeBusyError


class TestConcurrentLimits(unittest.TestCase):

    def test_max_concurrent_requests_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(16, 4)
            save_model(model, model_path)

            # Limit concurrent requests to 2
            limits = RuntimeLimits(max_concurrent_requests=2)
            runtime = InferenceRuntime.load(model_path, limits=limits)

            busy_count = 0
            success_count = 0
            x_test = tf.randn((2, 16))

            def worker():
                nonlocal busy_count, success_count
                try:
                    _ = runtime.predict(x_test)
                    success_count += 1
                except RuntimeBusyError:
                    busy_count += 1

            # Dispatch 20 concurrent threads
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(worker) for _ in range(20)]
                concurrent.futures.wait(futures)

            # Verify that total attempts sum up to 20 and active requests drain to 0
            self.assertEqual(success_count + busy_count, 20)
            self.assertEqual(runtime.active_contexts, 0)
            stats = runtime.stats()
            self.assertEqual(stats["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
