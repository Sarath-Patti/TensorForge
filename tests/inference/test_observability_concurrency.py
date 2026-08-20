"""Tests verifying thread-safety of MetricsCollector under concurrent updates."""

import concurrent.futures
import unittest
from tensorforge.inference.observability import MetricsCollector


class TestObservabilityConcurrency(unittest.TestCase):

    def test_concurrent_metric_recording(self):
        collector = MetricsCollector(history_size=1000)

        def worker(thread_id: int):
            for i in range(20):
                collector.record_request_submitted(queue_depth=i)
                collector.record_request_completed(
                    queue_wait_ms=0.1 * i,
                    exec_ms=0.5 * i,
                    e2e_ms=0.6 * i,
                    samples=2,
                )
                collector.record_backend("native", is_fused=True)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, i) for i in range(10)]
            _ = [f.result() for f in futures]

        snapshot = collector.snapshot()
        self.assertEqual(snapshot.requests.submitted, 200)
        self.assertEqual(snapshot.requests.completed, 200)
        self.assertEqual(snapshot.backends.native_fused_count, 200)
        self.assertEqual(snapshot.latency.execution.sample_count, 200)


if __name__ == "__main__":
    unittest.main()
