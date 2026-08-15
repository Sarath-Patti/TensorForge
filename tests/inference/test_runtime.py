"""Tests for InferenceRuntime API and lifecycle."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def test_inference_runtime_load_and_predict():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "classifier.tfmodel")

        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        )
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path)
        assert isinstance(runtime, InferenceRuntime)
        assert runtime.is_quantized is False
        assert runtime.input_shape == (8,)
        assert runtime.output_shape == (3,)

        # Test single sample prediction (1D or 2D batch)
        x_np = np.random.randn(1, 8).astype(np.float32)
        out1 = runtime.predict(x_np)
        assert isinstance(out1, tf.Tensor)
        assert out1.shape == (1, 3)

        # Test batch prediction
        x_batch = tf.randn((10, 8))
        out_batch = runtime.predict_batch(x_batch)
        assert out_batch.shape == (10, 3)


def test_inference_runtime_summary_and_properties():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "linear.tfmodel")
        model = nn.Linear(4, 2)
        save_model(model, model_path, metadata={"author": "TensorForge Team"})

        runtime = InferenceRuntime.load(model_path)
        summary = runtime.summary()

        assert summary["model_type"] == "Linear"
        assert summary["num_parameters"] == 10  # 4*2 + 2
        assert summary["input_shape"] == (4,)
        assert summary["output_shape"] == (2,)
        assert summary["is_quantized"] is False
        assert "backend" in summary
        assert "architecture" in summary
