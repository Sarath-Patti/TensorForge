"""Tests for Module and Optimizer state_dict APIs and Tensor.copy semantics."""

from collections import OrderedDict
import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
import tensorforge.optim as optim
from tensorforge.utils.validation import SerializationError


def test_tensor_copy_independence_and_state_dict_isolation():
    # 1. Test Tensor.copy() basic properties
    t = tf.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, requires_grad=True)
    t_copy = t.copy()

    assert isinstance(t_copy, tf.Tensor)
    assert t_copy.shape == t.shape
    assert t_copy.dtype == t.dtype
    assert t_copy.requires_grad == t.requires_grad
    assert t_copy.is_leaf is True
    assert t_copy.grad_fn is None
    assert id(t) != id(t_copy)

    # 2. Modifying copy does not modify original
    t_copy_np = t_copy.numpy()
    t_copy_np[0, 0] = 999.0
    np.copyto(t_copy.storage.to_numpy(), t_copy_np.reshape(-1))

    assert t.numpy()[0, 0] == 1.0
    assert t_copy.numpy()[0, 0] == 999.0

    # 3. Test state_dict() returns copied tensors and does not share identity with parameters
    model = nn.Linear(4, 2)
    sd = model.state_dict()

    for name, param in model.named_parameters():
        assert name in sd
        copied_tensor = sd[name]
        assert id(copied_tensor) != id(param)
        assert copied_tensor.requires_grad is False  # Detached copy
        np.testing.assert_array_equal(copied_tensor.numpy(), param.numpy())

        # Modify state_dict tensor and verify parameter is unaffected
        copied_np = copied_tensor.numpy()
        copied_np.fill(1234.5)
        np.copyto(copied_tensor.storage.to_numpy(), copied_np.reshape(-1))

        assert not np.all(param.numpy() == 1234.5)


def test_module_state_dict_generation():
    model = nn.Sequential(
        nn.Linear(4, 8, bias=True),
        nn.ReLU(),
        nn.Linear(8, 2, bias=True),
    )

    sd = model.state_dict()
    assert isinstance(sd, (dict, OrderedDict))
    assert "0.weight" in sd
    assert "0.bias" in sd
    assert "2.weight" in sd
    assert "2.bias" in sd

    assert sd["0.weight"].shape == (8, 4)
    assert sd["0.bias"].shape == (8,)
    assert sd["2.weight"].shape == (2, 8)
    assert sd["2.bias"].shape == (2,)


def test_module_load_state_dict_in_place_preserves_identity():
    model1 = nn.Sequential(
        nn.Linear(4, 8),
        nn.Linear(8, 2),
    )
    model2 = nn.Sequential(
        nn.Linear(4, 8),
        nn.Linear(8, 2),
    )

    # Record parameter object IDs from model2
    p_ids_before = [id(p) for p in model2.parameters()]
    req_grad_before = [p.requires_grad for p in model2.parameters()]

    # Load state from model1 into model2
    model2.load_state_dict(model1.state_dict())

    p_ids_after = [id(p) for p in model2.parameters()]
    req_grad_after = [p.requires_grad for p in model2.parameters()]

    # Parameter object identities must remain identical (in-place buffer update)
    assert p_ids_before == p_ids_after
    assert req_grad_before == req_grad_after

    # Numerical values must match model1
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        assert n1 == n2
        np.testing.assert_array_equal(p1.numpy(), p2.numpy())
        assert p2.is_leaf is True
        assert p2.grad_fn is None


def test_module_load_state_dict_shape_mismatch_raises():
    model = nn.Sequential(nn.Linear(4, 8))
    corrupted_sd = {
        "0.weight": tf.randn((5, 5)),  # Expected (8, 4)
        "0.bias": tf.zeros((8,)),
    }

    with pytest.raises(SerializationError, match="Shape mismatch"):
        model.load_state_dict(corrupted_sd)


def test_module_load_state_dict_strictness():
    model = nn.Sequential(nn.Linear(4, 8, bias=True))
    full_sd = model.state_dict()

    # Missing key test
    partial_sd = {"0.weight": full_sd["0.weight"]}
    with pytest.raises(SerializationError, match="Missing key"):
        model.load_state_dict(partial_sd, strict=True)

    missing, unexpected = model.load_state_dict(partial_sd, strict=False)
    assert missing == ["0.bias"]
    assert unexpected == []

    # Unexpected key test
    extra_sd = dict(full_sd)
    extra_sd["extra.layer.weight"] = tf.zeros((2, 2))
    with pytest.raises(SerializationError, match="Unexpected key"):
        model.load_state_dict(extra_sd, strict=True)

    missing, unexpected = model.load_state_dict(extra_sd, strict=False)
    assert missing == []
    assert unexpected == ["extra.layer.weight"]


def test_optimizer_state_dict_roundtrip_adam():
    model = nn.Linear(4, 2)
    optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.85, 0.95), weight_decay=1e-4)

    # Perform a step to populate moments and step counters
    x = tf.randn((2, 4))
    out = model(x).sum()
    out.backward()
    optimizer.step()

    # Capture state_dict
    optim_sd = optimizer.state_dict()
    assert "state" in optim_sd
    assert "param_groups" in optim_sd
    assert optim_sd["param_groups"][0]["lr"] == 0.01
    assert optim_sd["param_groups"][0]["betas"] == (0.85, 0.95)

    # Create fresh model & optimizer
    model_fresh = nn.Linear(4, 2)
    model_fresh.load_state_dict(model.state_dict())
    optimizer_fresh = optim.Adam(model_fresh.parameters(), lr=0.001)

    # Restore optimizer state
    optimizer_fresh.load_state_dict(optim_sd)
    assert optimizer_fresh.param_groups[0]["lr"] == 0.01
    assert optimizer_fresh.param_groups[0]["betas"] == (0.85, 0.95)

    # Verify restored state for parameters
    for p_orig, p_fresh in zip(model.parameters(), model_fresh.parameters()):
        orig_s = optimizer.state[p_orig]
        fresh_s = optimizer_fresh.state[p_fresh]
        assert orig_s["step"] == fresh_s["step"]
        np.testing.assert_array_equal(orig_s["exp_avg"], fresh_s["exp_avg"])
        np.testing.assert_array_equal(orig_s["exp_avg_sq"], fresh_s["exp_avg_sq"])
