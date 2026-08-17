"""Tests verifying runtime recovery after failed or rejected inference requests."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeLimits
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeLimitError, TensorForgeInputError


class TestFailureRecovery(unittest.TestCase):

    def test_recovery_after_input_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(2, 8))

            # 1. Request A: Invalid input feature dimension (10 instead of 8) -> Fails
            with self.assertRaises(TensorForgeInputError):
                runtime.predict(tf.randn((2, 10)))

            self.assertEqual(runtime.active_contexts, 0)
            stats = runtime.stats()
            self.assertEqual(stats["input_validation_failures"], 1)

            # 2. Request B: Valid input -> Succeeds perfectly
            x_valid = tf.randn((2, 8))
            out_valid = runtime.predict(x_valid)
            self.assertEqual(out_valid.shape, (2, 4))
            self.assertEqual(runtime.active_contexts, 0)

    def test_recovery_after_resource_limit_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            limits = RuntimeLimits(max_batch_size=2)
            runtime = InferenceRuntime.load(model_path, limits=limits)

            # 1. Request A: Oversized batch (4 > 2) -> Fails
            with self.assertRaises(RuntimeLimitError):
                runtime.predict(tf.randn((4, 8)))

            # 2. Request B: Valid batch (2 <= 2) -> Succeeds
            out_valid = runtime.predict(tf.randn((2, 8)))
            self.assertEqual(out_valid.shape, (2, 4))
            self.assertEqual(runtime.active_contexts, 0)


if __name__ == "__main__":
    unittest.main()
