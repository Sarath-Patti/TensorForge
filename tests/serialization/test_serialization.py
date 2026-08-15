"""Tests for .tfmodel serialization file formats and container validation."""

import os
import tempfile
import zipfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.serialization import (
    compute_model_size,
    load_model,
    load_state_dict_from_file,
    save_model,
)
from tensorforge.utils.validation import SerializationError


def test_save_and_load_model_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.tfmodel")

        # 1. Create and populate original model
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

        user_meta = {"task": "classification", "experiment_id": 42}
        save_model(model, model_path, metadata=user_meta)
        assert os.path.exists(model_path)

        # 2. Load into fresh model
        fresh_model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )
        loaded_meta = load_model(fresh_model, model_path)

        assert loaded_meta["user_metadata"]["task"] == "classification"
        assert loaded_meta["user_metadata"]["experiment_id"] == 42
        assert loaded_meta["format_version"] == "1.0"
        assert loaded_meta["library"] == "TensorForge"

        # 3. Verify forward outputs match exactly
        x = tf.randn((4, 8))
        with tf.no_grad():
            out_orig = model(x)
            out_fresh = fresh_model(x)

        np.testing.assert_allclose(out_orig.numpy(), out_fresh.numpy(), rtol=1e-6)


def test_load_state_dict_from_file_standalone():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "linear.tfmodel")
        model = nn.Linear(4, 2)
        save_model(model, model_path)

        sd, meta = load_state_dict_from_file(model_path)
        assert "weight" in sd
        assert "bias" in sd
        assert sd["weight"].shape == (2, 4)
        assert sd["bias"].shape == (2,)


def test_corrupted_model_file_handling():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Non-zip corrupted file
        bad_path = os.path.join(tmpdir, "corrupt.tfmodel")
        with open(bad_path, "wb") as f:
            f.write(b"NOT_A_ZIP_FILE_DATA")

        model = nn.Linear(4, 2)
        with pytest.raises(SerializationError, match="not a valid"):
            load_model(model, bad_path)

        # Zip missing metadata.json
        bad_zip_path = os.path.join(tmpdir, "missing_meta.tfmodel")
        with zipfile.ZipFile(bad_zip_path, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        with pytest.raises(SerializationError, match="missing 'metadata.json'"):
            load_model(model, bad_zip_path)


def test_compute_model_size():
    model = nn.Linear(10, 20, bias=True)  # 10*20 = 200 weights + 20 biases = 220 elements * 4 bytes = 880 bytes
    stats = compute_model_size(model)

    assert stats["num_parameters"] == 220
    assert stats["total_bytes"] == 880
    assert stats["size_kb"] == 880 / 1024.0
