"""Unit tests for RuntimeProfile and preset configuration resolution in TensorForge v2.0."""

import unittest

from tensorforge.inference import (
    RuntimeProfile,
    RuntimeProfileType,
    get_runtime_profile,
)
from tensorforge.utils.validation import TensorForgeInputError


class TestRuntimeProfiles(unittest.TestCase):
    """Test suite verifying preset runtime profiles."""

    def test_get_low_latency_profile(self):
        profile = get_runtime_profile("LOW_LATENCY")
        self.assertEqual(profile.profile_type, RuntimeProfileType.LOW_LATENCY)
        self.assertEqual(profile.scheduler_config.max_batch_size, 1)
        self.assertEqual(profile.scheduler_config.batch_timeout_ms, 0.5)

    def test_get_high_throughput_profile(self):
        profile = get_runtime_profile(RuntimeProfileType.HIGH_THROUGHPUT)
        self.assertEqual(profile.profile_type, RuntimeProfileType.HIGH_THROUGHPUT)
        self.assertEqual(profile.scheduler_config.max_batch_size, 64)

    def test_get_balanced_profile(self):
        profile = get_runtime_profile("balanced")
        self.assertEqual(profile.profile_type, RuntimeProfileType.BALANCED)
        self.assertEqual(profile.scheduler_config.max_batch_size, 16)

    def test_get_embedded_profile(self):
        profile = get_runtime_profile("EMBEDDED")
        self.assertEqual(profile.profile_type, RuntimeProfileType.EMBEDDED)
        self.assertEqual(profile.scheduler_config.max_batch_size, 4)

    def test_invalid_profile_raises_input_error(self):
        with self.assertRaises(TensorForgeInputError):
            get_runtime_profile("INVALID_PROFILE_NAME")

    def test_profile_to_dict(self):
        profile = get_runtime_profile("BALANCED")
        d = profile.to_dict()
        self.assertEqual(d["profile_type"], "BALANCED")
        self.assertIn("scheduler_config", d)
        self.assertIn("circuit_breaker_config", d)
        self.assertIn("retry_config", d)


if __name__ == "__main__":
    unittest.main()
