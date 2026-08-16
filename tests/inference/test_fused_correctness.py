"""Tests verifying numerical correctness and parity of fused inference operators."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestFusedCorrectness(unittest.TestCase):

    def _check_fused_parity(self, model, in_features, atol=1e-5):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model.eval()
            save_model(model, model_path)

            # 1. Unfused Runtime
            runtime_unfused = InferenceRuntime.load(model_path)
            # 2. Optimized Fused Runtime
            runtime_fused = InferenceRuntime.load(model_path).optimize()

            self.assertTrue(runtime_fused.is_optimized)

            # Test single sample and batches
            for b in [1, 4, 16]:
                x = tf.randn((b, in_features))
                with tf.no_grad():
                    ref_out = model(x)
                unfused_out = runtime_unfused.predict(x)
                fused_out = runtime_fused.predict(x)

                np.testing.assert_allclose(unfused_out.numpy(), ref_out.numpy(), atol=atol, rtol=1e-5)
                np.testing.assert_allclose(fused_out.numpy(), ref_out.numpy(), atol=atol, rtol=1e-5)

    def test_fused_linear_relu_correctness(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self._check_fused_parity(model, in_features=8)

    def test_fused_linear_sigmoid_correctness(self):
        model = nn.Sequential(nn.Linear(6, 12), nn.Sigmoid())
        self._check_fused_parity(model, in_features=6)

    def test_fused_linear_tanh_correctness(self):
        model = nn.Sequential(nn.Linear(10, 8), nn.Tanh())
        self._check_fused_parity(model, in_features=10)

    def test_fused_linear_softmax_correctness(self):
        model = nn.Sequential(nn.Linear(8, 4), nn.Softmax(dim=-1))
        self._check_fused_parity(model, in_features=8)

    def test_fused_deep_mlp_correctness(self):
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
        self._check_fused_parity(model, in_features=16)


if __name__ == "__main__":
    unittest.main()
