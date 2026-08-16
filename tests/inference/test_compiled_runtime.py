"""Tests for compiled InferenceRuntime execution, caching, and correctness."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


def test_runtime_compile_and_predict_parity():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path)
        assert runtime.is_compiled is False

        runtime.compile(input_shape=(8, 16))
        assert runtime.is_compiled is True
        assert runtime.execution_plan is not None
        assert runtime.workspace_size > 0

        # Predict with compiled runtime
        x = tf.randn((8, 16))
        with tf.no_grad():
            ref_out = model(x)
        compiled_out = runtime.predict(x)

        np.testing.assert_allclose(compiled_out.numpy(), ref_out.numpy(), atol=1e-5, rtol=1e-5)


def test_runtime_compiled_dynamic_batch_execution():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 2),
        )
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))

        # Predict with different batch sizes
        for b in [1, 4, 16, 32]:
            x = tf.randn((b, 8))
            with tf.no_grad():
                ref_out = model(x)
            compiled_out = runtime.predict(x)
            assert compiled_out.shape == (b, 2)
            np.testing.assert_allclose(compiled_out.numpy(), ref_out.numpy(), atol=1e-5, rtol=1e-5)


def test_runtime_compiled_summary():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Softmax(dim=-1),
        )
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
        summary = runtime.summary()

        assert summary["is_optimized"] is True
        assert summary["is_compiled"] is True
        assert summary["compiled_steps"] == 2
        assert summary["workspace_bytes"] > 0
        assert summary["tensorforge_version"] == "1.1.0"


@pytest.mark.skipif(not is_native_available(), reason="Native C++ extension not available")
def test_runtime_compiled_native_backend():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.Sigmoid(),
        )
        save_model(model, model_path)

        runtime_np = InferenceRuntime.load(model_path, backend="numpy").compile(input_shape=(4, 8))
        runtime_native = InferenceRuntime.load(model_path, backend="native").compile(input_shape=(4, 8))

        x = tf.randn((4, 8))
        out_np = runtime_np.predict(x)
        out_native = runtime_native.predict(x)

        np.testing.assert_allclose(out_np.numpy(), out_native.numpy(), atol=1e-5, rtol=1e-5)


def test_runtime_compiled_no_grad_and_immutability():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")
        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        save_model(model, model_path)

        runtime = InferenceRuntime.load(model_path).compile(input_shape=(2, 4))
        w_before = runtime.model[0].weight.numpy().copy()

        x = tf.randn((2, 4), requires_grad=True)
        out = runtime.predict(x)

        assert out.requires_grad is False
        assert out.grad_fn is None

        w_after = runtime.model[0].weight.numpy()
        np.testing.assert_array_equal(w_before, w_after)


def test_runtime_compiled_int8_quantized():
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

        runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
        assert runtime.is_quantized is True
        assert runtime.is_compiled is True

        x = tf.randn((4, 8))
        out = runtime.predict(x)
        assert out.shape == (4, 16)
        assert not np.any(np.isnan(out.numpy()))
