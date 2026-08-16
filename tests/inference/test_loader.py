"""Tests for ModelLoader and architecture reconstruction."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.loader import ModelLoader
from tensorforge.serialization import save_model


class TestLoader(unittest.TestCase):

    def test_reconstruct_architecture_linear_and_activations(self):
        # Linear
        linear_cfg = {"type": "Linear", "in_features": 8, "out_features": 16, "bias": True, "dtype": "float32"}
        lin = ModelLoader.reconstruct_architecture(linear_cfg)
        self.assertIsInstance(lin, nn.Linear)
        self.assertEqual(lin.in_features, 8)
        self.assertEqual(lin.out_features, 16)
        self.assertIsNotNone(lin.bias)

        # Activations
        self.assertIsInstance(ModelLoader.reconstruct_architecture({"type": "ReLU"}), nn.ReLU)
        self.assertIsInstance(ModelLoader.reconstruct_architecture({"type": "Sigmoid"}), nn.Sigmoid)
        self.assertIsInstance(ModelLoader.reconstruct_architecture({"type": "Tanh"}), nn.Tanh)

        softmax_mod = ModelLoader.reconstruct_architecture({"type": "Softmax", "dim": -1})
        self.assertIsInstance(softmax_mod, nn.Softmax)
        self.assertEqual(softmax_mod.dim, -1)

    def test_reconstruct_architecture_sequential(self):
        seq_cfg = {
            "type": "Sequential",
            "layers": [
                {"module": {"type": "Linear", "in_features": 4, "out_features": 8, "bias": True}},
                {"module": {"type": "ReLU"}},
                {"module": {"type": "Linear", "in_features": 8, "out_features": 2, "bias": False}},
            ],
        }
        model = ModelLoader.reconstruct_architecture(seq_cfg)
        self.assertIsInstance(model, nn.Sequential)
        self.assertEqual(len(model), 3)
        self.assertIsInstance(model[0], nn.Linear)
        self.assertIsInstance(model[1], nn.ReLU)
        self.assertIsInstance(model[2], nn.Linear)
        self.assertIsNone(model[2].bias)

    def test_model_loader_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            original_model = nn.Sequential(
                nn.Linear(8, 16),
                nn.ReLU(),
                nn.Linear(16, 4),
            )

            save_model(original_model, model_path)
            loaded_model, state_dict, metadata, is_quantized = ModelLoader.load(model_path)

            self.assertFalse(is_quantized)
            self.assertEqual(len(loaded_model), 3)
            self.assertEqual(len(state_dict), 4)

            x = tf.randn((2, 8))
            with tf.no_grad():
                original_out = original_model(x)
                loaded_out = loaded_model(x)

            np.testing.assert_allclose(loaded_out.numpy(), original_out.numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
