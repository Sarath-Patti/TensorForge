"""Tests for MetricsCollector tracking and PerformanceSnapshot data integrity."""

import unittest
from tensorforge.inference.observability import (
    BatchMetrics,
    LatencyHistogram,
    MetricsCollector,
    PerformanceSnapshot,
    RequestMetrics,
)


class TestObservability(unittest.TestCase):

    def test_metrics_collector_basic_lifecycle(self):
        collector = MetricsCollector(history_size=100)

        # 1. Record requests
        collector.record_request_submitted(queue_depth=2)
        collector.record_request_completed(queue_wait_ms=1.5, exec_ms=3.0, e2e_ms=4.5, samples=4)

        collector.record_request_submitted(queue_depth=1)
        collector.record_request_failed(exec_ms=2.0)

        collector.record_request_rejected(reason="limit")
        collector.record_request_cancelled()

        # 2. Record batch
        collector.record_batch(batch_size=8, configured_max_batch=16)

        # 3. Take snapshot
        snapshot = collector.snapshot()

        self.assertIsInstance(snapshot, PerformanceSnapshot)
        self.assertEqual(snapshot.requests.submitted, 2)
        self.assertEqual(snapshot.requests.completed, 1)
        self.assertEqual(snapshot.requests.failed, 1)
        self.assertEqual(snapshot.requests.rejected, 1)
        self.assertEqual(snapshot.requests.cancelled, 1)
        self.assertEqual(snapshot.batches.batches_formed, 1)
        self.assertEqual(snapshot.batches.samples_processed, 8)
        self.assertEqual(snapshot.batches.average_batch_size, 8.0)
        self.assertEqual(snapshot.batches.batch_utilization, 0.5)

    def test_snapshot_to_dict_and_json(self):
        collector = MetricsCollector()
        collector.record_request_submitted()
        collector.record_request_completed(queue_wait_ms=0.5, exec_ms=1.2, e2e_ms=1.7, samples=2)

        snapshot = collector.snapshot()
        d = snapshot.to_dict()

        self.assertIn("requests", d)
        self.assertIn("batches", d)
        self.assertIn("latency", d)
        self.assertIn("throughput", d)
        self.assertIn("backends", d)
        self.assertIn("compiler", d)
        self.assertIn("memory", d)
        self.assertEqual(d["tensorforge_version"], "1.8.0")

        json_str = snapshot.to_json()
        self.assertIsInstance(json_str, str)
        self.assertIn('"tensorforge_version": "1.8.0"', json_str)


if __name__ == "__main__":
    unittest.main()
