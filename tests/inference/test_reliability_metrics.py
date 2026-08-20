"""Unit tests for Reliability Metrics in PerformanceSnapshot in TensorForge v1.9 (Static Creation Only)."""

import unittest
from tensorforge.inference import (
    MetricsCollector,
    ReliabilityMetrics,
    PerformanceSnapshot,
)


class TestReliabilityMetrics(unittest.TestCase):
    """Test suite verifying reliability metrics aggregation and JSON serialization."""

    def test_reliability_metrics_recording(self):
        """Verify recording timeouts, cancellations, retries, and circuit open rejections."""
        collector = MetricsCollector()
        collector.record_request_submitted()
        collector.record_timeout()
        collector.record_retry()
        collector.record_circuit_open_rejection()

        snap = collector.snapshot()
        self.assertIsNotNone(snap.reliability)
        self.assertEqual(snap.reliability.timeouts, 1)
        self.assertEqual(snap.reliability.retries, 1)
        self.assertEqual(snap.reliability.circuit_open_rejections, 1)

        d = snap.to_dict()
        self.assertIn("reliability", d)
        self.assertEqual(d["reliability"]["timeouts"], 1)


if __name__ == "__main__":
    unittest.main()
