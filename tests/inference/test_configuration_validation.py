"""Unit tests for configuration validation in TensorForge v2.0."""

import unittest

from tensorforge.inference import (
    DeploymentManifest,
    RuntimeLimits,
    SchedulerConfig,
    ServerConfig,
)
from tensorforge.utils.validation import TensorForgeInputError


class TestConfigurationValidation(unittest.TestCase):
    """Test suite verifying validation logic for server, scheduler, and limit configurations."""

    def test_scheduler_config_validation(self):
        with self.assertRaises((TensorForgeInputError, ValueError)):
            SchedulerConfig(max_batch_size=0)

        with self.assertRaises((TensorForgeInputError, ValueError)):
            SchedulerConfig(batch_timeout_ms=-1.0)

    def test_runtime_limits_validation(self):
        with self.assertRaises((TensorForgeInputError, ValueError)):
            RuntimeLimits(max_batch_size=0)

        with self.assertRaises((TensorForgeInputError, ValueError)):
            RuntimeLimits(max_input_elements=-5)

    def test_manifest_load_nonexistent_file_validation(self):
        with self.assertRaises(TensorForgeInputError):
            DeploymentManifest.load_json("/nonexistent/file/path/manifest.json")


if __name__ == "__main__":
    unittest.main()
