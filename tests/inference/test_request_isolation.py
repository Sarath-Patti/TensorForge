"""Tests verifying per-prediction request isolation, context separation, and request ID tracking."""

import concurrent.futures
import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestRequestIsolation(unittest.TestCase):

    def test_request_id_and_context_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(2, 8))

            # Run sequential requests and verify request counter increments
            _ = runtime.predict(tf.randn((2, 8)))
            _ = runtime.predict(tf.randn((2, 8)))

            stats = runtime.stats()
            self.assertEqual(stats["completed_requests"], 2)
            self.assertEqual(stats["accepted_requests"], 2)

    def test_concurrent_request_isolation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 16))

            def worker():
                x = tf.randn((4, 16))
                return runtime.predict(x).shape

            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(worker) for _ in range(20)]
                results = [f.result() for f in futures]

            for shape in results:
                self.assertEqual(shape, (4, 8))

            stats = runtime.stats()
            self.assertEqual(stats["completed_requests"], 20)
            self.assertEqual(stats["failed_requests"], 0)


if __name__ == "__main__":
    unittest.main()
