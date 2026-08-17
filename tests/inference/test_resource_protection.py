"""Tests verifying workspace memory limits and protection against oversized allocations."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeLimitError


class TestResourceProtection(unittest.TestCase):

    def test_workspace_memory_protection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, 10),
            )
            save_model(model, model_path)

            # Restrict workspace memory to 100 bytes (smaller than model requires)
            limits = RuntimeLimits(max_workspace_bytes=100)
            runtime = InferenceRuntime.load(model_path, limits=limits).compile(input_shape=(8, 64))

            # Prediction should be rejected due to workspace requirement
            with self.assertRaises(RuntimeLimitError):
                runtime.predict(tf.randn((8, 64)))

            health = runtime.health()
            self.assertEqual(health["rejected_requests"], 1)
            self.assertEqual(health["resource_limit_violations"], 1)


if __name__ == "__main__":
    unittest.main()
