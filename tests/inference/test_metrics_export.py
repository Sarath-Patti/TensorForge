"""Tests for JSON export and serialization of PerformanceSnapshot."""

import json
import os
import tempfile
import unittest
from tensorforge.inference.observability import MetricsCollector


class TestMetricsExport(unittest.TestCase):

    def test_json_export_and_load(self):
        collector = MetricsCollector()
        collector.record_request_submitted()
        collector.record_request_completed(queue_wait_ms=0.5, exec_ms=2.5, e2e_ms=3.0, samples=4)
        collector.record_batch(batch_size=4, configured_max_batch=16)

        snapshot = collector.snapshot()

        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = os.path.join(tmpdir, "snapshot.json")
            snapshot.save_json(json_file, indent=2)

            self.assertTrue(os.path.exists(json_file))
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertIn("requests", data)
            self.assertIn("batches", data)
            self.assertIn("latency", data)
            self.assertEqual(data["requests"]["completed"], 1)
            self.assertEqual(data["tensorforge_version"], "1.7.0")


if __name__ == "__main__":
    unittest.main()
