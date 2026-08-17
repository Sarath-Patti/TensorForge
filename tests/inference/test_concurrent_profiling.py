"""Tests verifying thread safety and determinism of RuntimeProfiler under concurrent multi-threaded predictions."""

import concurrent.futures
import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestConcurrentProfiling(unittest.TestCase):

    def test_concurrent_predictions_with_profiling_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 16))
            runtime.enable_profiling(detailed=True)

            num_workers = 6
            requests_per_worker = 10
            total_requests = num_workers * requests_per_worker

            # Fixed test input
            x_test = tf.randn((4, 16))
            ref_out = runtime.predict(x_test).numpy()

            def worker_task():
                results = []
                for _ in range(requests_per_worker):
                    out = runtime.predict(x_test)
                    results.append(out.numpy())
                return results

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker_task) for _ in range(num_workers)]
                all_results = [f.result() for f in futures]

            # Verify thread-safe telemetry aggregation
            lat = runtime.latency_stats()
            # Total predictions = 1 (ref) + total_requests
            self.assertEqual(lat["prediction_count"], total_requests + 1)
            self.assertEqual(runtime.prediction_count, total_requests + 1)
            self.assertEqual(runtime.error_count, 0)

            # Verify deterministic output across all workers
            for worker_res in all_results:
                for res in worker_res:
                    np.testing.assert_allclose(res, ref_out, atol=1e-5, rtol=1e-5)

            # Verify report generation is thread-safe
            report = runtime.profile()
            self.assertEqual(report.prediction_count, total_requests + 1)
            self.assertGreater(len(report.events), 0)


if __name__ == "__main__":
    unittest.main()
