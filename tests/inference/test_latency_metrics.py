"""Tests for LatencyHistogram percentiles, bounding, and reservoir sampling."""

import unittest
from tensorforge.inference.observability import LatencyHistogram, LatencyStats


class TestLatencyMetrics(unittest.TestCase):

    def test_empty_histogram_safety(self):
        hist = LatencyHistogram(capacity=100)
        stats = hist.stats()

        self.assertIsInstance(stats, LatencyStats)
        self.assertEqual(stats.sample_count, 0)
        self.assertEqual(stats.min_ms, 0.0)
        self.assertEqual(stats.max_ms, 0.0)
        self.assertEqual(stats.mean_ms, 0.0)
        self.assertEqual(stats.p50_ms, 0.0)
        self.assertEqual(stats.p90_ms, 0.0)
        self.assertEqual(stats.p95_ms, 0.0)
        self.assertEqual(stats.p99_ms, 0.0)

    def test_percentile_calculations(self):
        hist = LatencyHistogram(capacity=1000)

        # Record values 1 to 100
        for i in range(1, 101):
            hist.record(float(i))

        stats = hist.stats()
        self.assertEqual(stats.sample_count, 100)
        self.assertEqual(stats.min_ms, 1.0)
        self.assertEqual(stats.max_ms, 100.0)
        self.assertAlmostEqual(stats.mean_ms, 50.5, places=2)
        self.assertAlmostEqual(stats.p50_ms, 50.5, places=1)
        self.assertAlmostEqual(stats.p90_ms, 90.1, places=0)
        self.assertAlmostEqual(stats.p95_ms, 95.05, places=0)
        self.assertAlmostEqual(stats.p99_ms, 99.01, places=0)

    def test_bounded_capacity(self):
        hist = LatencyHistogram(capacity=50)

        # Record 200 samples into capacity 50
        for i in range(200):
            hist.record(float(i))

        stats = hist.stats()
        self.assertEqual(stats.sample_count, 200)
        self.assertEqual(len(hist._samples), 50)
        self.assertEqual(stats.max_ms, 199.0)


if __name__ == "__main__":
    unittest.main()
