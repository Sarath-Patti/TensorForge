"""Tests verifying no_grad and zero autograd graph allocation during inference."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestInferenceNoGrad(unittest.TestCase):

    def test_inference_no_grad_and_leaf_guarantees(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 2),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)

            # Input with requires_grad=True
            x = tf.randn((4, 4), requires_grad=True)

            out = runtime.predict(x)

            # Output guarantees
            self.assertFalse(out.requires_grad)
            self.assertIsNone(out.grad_fn)
            self.assertTrue(out.is_leaf)

            # Model parameter guarantees
            for param in runtime.model.parameters():
                self.assertIsNone(param.grad)

    def test_inference_does_not_mutate_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            w_before = runtime.model.weight.numpy().copy()

            x = tf.randn((8, 4))
            for _ in range(10):
                _ = runtime.predict(x)

            w_after = runtime.model.weight.numpy()
            np.testing.assert_array_equal(w_before, w_after)


if __name__ == "__main__":
    unittest.main()
