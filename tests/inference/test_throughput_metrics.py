"""Tests for throughput rate calculations and measurement windows."""

import time
import unittest
from tensorforge.inference.observability import MetricsCollector, ThroughputStats


class TestThroughputMetrics(unittest.TestCase):

    def test_throughput_calculations(self):
        collector = MetricsCollector()

        # Record 10 completed requests with 4 samples each
        for _ in range(10):
            collector.record_request_submitted()
            collector.record_request_completed(queue_wait_ms=0.1, exec_ms=0.5, e2e_ms=0.6, samples=4)

        collector.record_batch(batch_size=40, configured_max_batch=64)

        snapshot = collector.snapshot()
        tp = snapshot.throughput

        self.assertIsInstance(tp, ThroughputStats)
        self.assertGreater(tp.requests_per_sec, 0.0)
        self.assertGreater(tp.samples_per_sec, 0.0)
        self.assertGreater(tp.batches_per_sec, 0.0)
        self.assertGreater(tp.window_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
