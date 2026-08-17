"""Tests verifying comprehensive input validation rules (rank, feature size, non-finite values)."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model
from tensorforge.utils.validation import TensorForgeInputError


class TestInputValidation(unittest.TestCase):

    def test_scalar_and_0d_input_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)

            # Scalar tensor (0-dim)
            x_scalar = tf.Tensor(np.array(5.0, dtype=np.float32))
            with self.assertRaises(TensorForgeInputError):
                runtime.predict(x_scalar)

    def test_nan_and_inf_input_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)

            # NaN input
            x_nan = tf.from_numpy(np.array([[1.0, np.nan, 2.0, 3.0]], dtype=np.float32))
            with self.assertRaises(TensorForgeInputError):
                runtime.predict(x_nan)

            # Inf input
            x_inf = tf.from_numpy(np.array([[1.0, np.inf, 2.0, 3.0]], dtype=np.float32))
            with self.assertRaises(TensorForgeInputError):
                runtime.predict(x_inf)

    def test_feature_dimension_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)

            # Shape mismatch (6 instead of 8)
            with self.assertRaises(TensorForgeInputError):
                runtime.predict(tf.randn((2, 6)))


if __name__ == "__main__":
    unittest.main()
