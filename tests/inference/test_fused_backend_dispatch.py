"""Tests for multi-backend execution, no-grad invariants, and INT8 fusion."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import backend_context, is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


def test_fused_runtime_summary_statistics():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).optimize()
        summary = runtime.summary()

        assert summary["is_optimized"] is True
        assert summary["original_nodes"] == 4
        assert summary["optimized_nodes"] == 2
        assert summary["fused_count"] == 2
        assert summary["fused_patterns"] == ["Linear+ReLU", "Linear+Softmax"]


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not compiled")
def test_fused_native_vs_numpy_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )
        save_model(model, model_path)

        runtime_np = InferenceRuntime.load(model_path, backend="numpy").optimize()
        runtime_native = InferenceRuntime.load(model_path, backend="native").optimize()

        x = tf.randn((10, 8))
        out_np = runtime_np.predict(x)
        out_native = runtime_native.predict(x)

        np.testing.assert_allclose(out_np.numpy(), out_native.numpy(), atol=1e-5, rtol=1e-5)


def test_fused_no_grad_and_immutability():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).optimize()
        w_before = runtime.model[0].weight.numpy().copy()

        x = tf.randn((4, 4), requires_grad=True)
        out = runtime.predict(x)

        assert out.requires_grad is False
        assert out.grad_fn is None
        assert out.is_leaf is True

        w_after = runtime.model[0].weight.numpy()
        np.testing.assert_array_equal(w_before, w_after)


def test_fused_int8_quantized_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "quantized_model.tfmodel")
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())

        q_weights = {name: quantize(param, scheme="symmetric") for name, param in model.named_parameters()}
        write_tfmodel_container(
            model_path,
            q_weights,
            metadata={"is_quantized": True},
            architecture=extract_module_architecture(model),
        )

        runtime = InferenceRuntime.load(model_path).optimize()
        assert runtime.is_quantized is True
        assert runtime.is_optimized is True

        x = tf.randn((4, 8))
        out = runtime.predict(x)
        assert out.shape == (4, 16)
        assert not np.any(np.isnan(out.numpy()))
