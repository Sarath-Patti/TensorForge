"""Unit tests for Retry Policy and Exponential Backoff in TensorForge v1.9 (Static Creation Only)."""

import unittest
from tensorforge.inference import (
    RetryConfig,
    compute_backoff_delay_sec,
    TensorForgeInputError,
    RequestCancelledError,
)


class TestRetryPolicy(unittest.TestCase):
    """Test suite verifying backoff delay math and retryable exception filtering."""

    def test_backoff_math(self):
        """Verify bounded exponential backoff delay calculations."""
        config = RetryConfig(base_delay_ms=10.0, max_delay_ms=100.0, backoff_factor=2.0)
        
        # Attempt 0: 10ms * (2^0) = 10ms -> 0.01s
        self.assertAlmostEqual(compute_backoff_delay_sec(0, config), 0.01)
        # Attempt 1: 10ms * (2^1) = 20ms -> 0.02s
        self.assertAlmostEqual(compute_backoff_delay_sec(1, config), 0.02)
        # Attempt 4: 10ms * (2^4) = 160ms -> capped at 100ms -> 0.1s
        self.assertAlmostEqual(compute_backoff_delay_sec(4, config), 0.1)

    def test_retryable_exception_filtering(self):
        """Verify client validation errors are excluded from retries."""
        config = RetryConfig()
        self.assertFalse(config.is_retryable(TensorForgeInputError("bad shape")))
        self.assertFalse(config.is_retryable(RequestCancelledError("cancelled")))
        self.assertTrue(config.is_retryable(RuntimeError("transient execution error")))


if __name__ == "__main__":
    unittest.main()
