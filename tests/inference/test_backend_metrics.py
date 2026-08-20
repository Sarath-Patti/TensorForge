"""Tests for backend execution accounting and fallback categorization."""

import unittest
from tensorforge.inference.observability import BackendMetrics, MetricsCollector


class TestBackendMetrics(unittest.TestCase):

    def test_backend_execution_and_fallback_tracking(self):
        collector = MetricsCollector()

        # Native fused executions
        collector.record_backend("native", is_fused=True, is_compiled=True)
        collector.record_backend("native", is_fused=True, is_compiled=True)

        # NumPy fallback execution
        collector.record_backend(
            "numpy",
            is_fused=False,
            is_compiled=False,
            was_fallback=True,
            fallback_reason="unsupported_dtype",
        )

        snapshot = collector.snapshot()
        be = snapshot.backends

        self.assertIsInstance(be, BackendMetrics)
        self.assertEqual(be.native_fused_count, 2)
        self.assertEqual(be.native_executed, 2)
        self.assertEqual(be.numpy_count, 1)
        self.assertEqual(be.compiled_count, 2)
        self.assertEqual(be.eager_count, 1)
        self.assertEqual(be.native_fallback, 1)
        self.assertEqual(be.fallback_reasons.get("unsupported_dtype"), 1)


if __name__ == "__main__":
    unittest.main()
