"""Tests for ModelLoader and architecture reconstruction."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.loader import ModelLoader
from tensorforge.serialization import save_model


def test_reconstruct_architecture_linear_and_activations():
    # Linear
    linear_cfg = {"type": "Linear", "in_features": 8, "out_features": 16, "bias": True, "dtype": "float32"}
    lin = ModelLoader.reconstruct_architecture(linear_cfg)
    assert isinstance(lin, nn.Linear)
    assert lin.in_features == 8
    assert lin.out_features == 16
    assert lin.bias is not None

    # Activations
    assert isinstance(ModelLoader.reconstruct_architecture({"type": "ReLU"}), nn.ReLU)
    assert isinstance(ModelLoader.reconstruct_architecture({"type": "Sigmoid"}), nn.Sigmoid)
    assert isinstance(ModelLoader.reconstruct_architecture({"type": "Tanh"}), nn.Tanh)

    softmax_mod = ModelLoader.reconstruct_architecture({"type": "Softmax", "dim": -1})
    assert isinstance(softmax_mod, nn.Softmax)
    assert softmax_mod.dim == -1


def test_reconstruct_architecture_sequential():
    seq_cfg = {
        "type": "Sequential",
        "layers": [
            {"module": {"type": "Linear", "in_features": 4, "out_features": 8, "bias": True}},
            {"module": {"type": "ReLU"}},
            {"module": {"type": "Linear", "in_features": 8, "out_features": 2, "bias": False}},
        ],
    }
    model = ModelLoader.reconstruct_architecture(seq_cfg)
    assert isinstance(model, nn.Sequential)
    assert len(model) == 3
    assert isinstance(model[0], nn.Linear)
    assert isinstance(model[1], nn.ReLU)
    assert isinstance(model[2], nn.Linear)
    assert model[2].bias is None


def test_model_loader_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        original_model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        save_model(original_model, model_path)
        loaded_model, state_dict, metadata, is_quantized = ModelLoader.load(model_path)

        assert isinstance(loaded_model, nn.Sequential)
        assert len(loaded_model) == 3
        assert is_quantized is False
        assert loaded_model.training is False  # Must be in eval mode

        # Verify weights
        for (n1, p1), (n2, p2) in zip(original_model.named_parameters(), loaded_model.named_parameters()):
            assert n1 == n2
            np.testing.assert_array_equal(p1.numpy(), p2.numpy())
