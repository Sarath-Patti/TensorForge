"""Tests for FP32 inference numerical parity and deterministic execution."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def test_fp32_prediction_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "mlp.tfmodel")

        model = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, 4),
            nn.Softmax(dim=-1),
        )
        model.eval()
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path)

        # Generate test inputs
        x_test = tf.randn((16, 6))

        with tf.no_grad():
            ref_out = model(x_test)

        runtime_out = runtime.predict(x_test)

        # Prediction parity target: bit-exact or near float32 precision
        np.testing.assert_allclose(runtime_out.numpy(), ref_out.numpy(), rtol=1e-6, atol=1e-6)


def test_repeated_inference_determinism():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2))
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path)
        x = tf.randn((4, 4))

        first_out = runtime.predict(x).numpy()

        for _ in range(5):
            next_out = runtime.predict(x).numpy()
            np.testing.assert_array_equal(first_out, next_out)
