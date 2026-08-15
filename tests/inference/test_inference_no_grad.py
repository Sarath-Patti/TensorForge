"""Tests verifying no_grad and zero autograd graph allocation during inference."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def test_inference_no_grad_and_leaf_guarantees():
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
        assert out.requires_grad is False
        assert out.grad_fn is None
        assert out.is_leaf is True

        # Model parameter guarantees
        for param in runtime.model.parameters():
            assert param.grad is None


def test_inference_does_not_mutate_weights():
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
