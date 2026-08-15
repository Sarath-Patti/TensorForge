"""Tests for QuantizedTensor serialization and low-precision state persistence."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.quantization import QuantizedTensor, dequantize, quantize
from tensorforge.serialization import (
    compute_model_size,
    load_state_dict_from_file,
    save_model,
)


def test_quantized_tensor_serialization_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "quantized_model.tfmodel")

        # 1. Create FP32 Model and Quantize Weights
        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 4),
        )

        q_state_dict = {}
        for name, param in model.named_parameters():
            q_state_dict[name] = quantize(param, scheme="symmetric")

        # 2. Save Quantized State Dict
        save_model(q_state_dict, model_path, metadata={"scheme": "symmetric", "is_quantized": True})

        # 3. Load Quantized State Dict
        loaded_sd, meta = load_state_dict_from_file(model_path)

        assert meta["user_metadata"]["is_quantized"] is True

        for name, q_orig in q_state_dict.items():
            assert name in loaded_sd
            q_loaded = loaded_sd[name]

            assert isinstance(q_loaded, QuantizedTensor)
            assert q_loaded.dtype == tf.int8
            assert q_loaded.shape == q_orig.shape
            assert pytest.approx(q_loaded.scale) == q_orig.scale
            assert q_loaded.zero_point == q_orig.zero_point
            assert q_loaded.scheme == q_orig.scheme

            np.testing.assert_array_equal(q_loaded.numpy(), q_orig.numpy())

            # Dequantized values must match
            np.testing.assert_allclose(q_loaded.dequantize().numpy(), q_orig.dequantize().numpy())


def test_quantized_model_size_comparison():
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.Linear(128, 32),
    )
    fp32_size = compute_model_size(model)

    q_state_dict = {}
    for name, param in model.named_parameters():
        q_state_dict[name] = quantize(param, scheme="symmetric")

    int8_size = compute_model_size(q_state_dict)

    assert int8_size["num_parameters"] == fp32_size["num_parameters"]
    assert fp32_size["total_bytes"] == int8_size["total_bytes"] * 4
    assert fp32_size["size_kb"] == pytest.approx(int8_size["size_kb"] * 4.0)
