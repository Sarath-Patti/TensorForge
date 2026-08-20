"""Unit tests for RuntimeConfig and configuration dataclasses in TensorForge v2.0."""

import unittest

from tensorforge.inference import (
    CircuitBreakerConfig,
    RetryConfig,
    RuntimeConfig,
    RuntimeLimits,
    RuntimeProfile,
    SchedulerConfig,
)


class TestRuntimeConfig(unittest.TestCase):
    """Test suite verifying RuntimeConfig alias and configuration dataclasses."""

    def test_runtime_config_alias_parity(self):
        self.assertIs(RuntimeConfig, RuntimeProfile)

    def test_scheduler_config(self):
        config = SchedulerConfig(max_batch_size=8, batch_timeout_ms=10.0)
        d = config.to_dict()
        self.assertEqual(d["max_batch_size"], 8)
        self.assertEqual(d["batch_timeout_ms"], 10.0)

    def test_runtime_limits(self):
        limits = RuntimeLimits(max_batch_size=16, max_input_elements=1000)
        d = limits.to_dict()
        self.assertEqual(d["max_batch_size"], 16)
        self.assertEqual(d["max_input_elements"], 1000)

    def test_reliability_configs(self):
        cb_cfg = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_ms=1000.0)
        self.assertEqual(cb_cfg.failure_threshold, 5)

        retry_cfg = RetryConfig(max_retries=3, base_delay_ms=20.0)
        self.assertEqual(retry_cfg.max_retries, 3)


if __name__ == "__main__":
    unittest.main()
