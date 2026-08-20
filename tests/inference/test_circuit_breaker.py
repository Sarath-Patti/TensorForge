"""Unit tests for CircuitBreaker state machine in TensorForge v1.9 (Static Creation Only)."""

import time
import unittest
from tensorforge.inference import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitBreakerOpenError,
)


class TestCircuitBreaker(unittest.TestCase):
    """Test suite verifying circuit breaker transitions, thresholds, and probing."""

    def test_circuit_breaker_transitions(self):
        """Verify CLOSED -> OPEN -> HALF_OPEN -> CLOSED transition lifecycle."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_ms=10.0, half_open_max_requests=1)
        cb = CircuitBreaker(config)

        self.assertEqual(cb.state, CircuitState.CLOSED)
        self.assertTrue(cb.allow_request())

        # Record failures to reach threshold
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.CLOSED)
        cb.record_failure()
        self.assertEqual(cb.state, CircuitState.OPEN)

        # Requests should be rejected while OPEN
        self.assertFalse(cb.allow_request())

        # Wait for recovery timeout to elapse
        time.sleep(0.02)
        self.assertEqual(cb.state, CircuitState.HALF_OPEN)

        # Probing request allowed in HALF_OPEN
        self.assertTrue(cb.allow_request())

        # Record success to close circuit
        cb.record_success()
        self.assertEqual(cb.state, CircuitState.CLOSED)


if __name__ == "__main__":
    unittest.main()
