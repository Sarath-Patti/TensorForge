"""Tests for training checkpoint save/load and training resume capabilities."""

import os
import tempfile
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
import tensorforge.optim as optim
from tensorforge.serialization import load_checkpoint, save_checkpoint


def test_checkpoint_save_and_load_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "checkpoint.tfckpt")

        # 1. Setup model, optimizer, and synthetic dataset
        model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

        # 2. Run initial training steps
        x = tf.randn((8, 4))
        y = tf.zeros((8, 2))
        criterion = nn.MSELoss()

        for step in range(3):
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # 3. Save checkpoint
        save_checkpoint(
            {
                "model": model,
                "optimizer": optimizer,
                "epoch": 2,
                "step": 3,
                "loss": loss.item() if hasattr(loss, "item") else float(loss.numpy()),
                "metrics": {"train_loss": 0.123},
                "user_metadata": {"author": "TensorForge Team"},
            },
            ckpt_path,
        )

        assert os.path.exists(ckpt_path)

        # 4. Create fresh model and optimizer and restore
        fresh_model = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        fresh_optimizer = optim.SGD(fresh_model.parameters(), lr=0.05, momentum=0.9)

        loaded_ckpt = load_checkpoint(ckpt_path)

        assert loaded_ckpt["epoch"] == 2
        assert loaded_ckpt["step"] == 3
        assert loaded_ckpt["user_metadata"]["author"] == "TensorForge Team"
        assert loaded_ckpt["metrics"]["train_loss"] == 0.123

        fresh_model.load_state_dict(loaded_ckpt["model_state_dict"])
        fresh_optimizer.load_state_dict(loaded_ckpt["optimizer_state_dict"])

        # 5. Verify resumed step trajectory matches
        with tf.no_grad():
            out1 = model(x)
            out2 = fresh_model(x)
        np.testing.assert_allclose(out1.numpy(), out2.numpy(), rtol=1e-6)

        # Perform next step on both models
        optimizer.zero_grad()
        loss1 = criterion(model(x), y)
        loss1.backward()
        optimizer.step()

        fresh_optimizer.zero_grad()
        loss2 = criterion(fresh_model(x), y)
        loss2.backward()
        fresh_optimizer.step()

        with tf.no_grad():
            next_out1 = model(x)
            next_out2 = fresh_model(x)
        np.testing.assert_allclose(next_out1.numpy(), next_out2.numpy(), rtol=1e-6)
