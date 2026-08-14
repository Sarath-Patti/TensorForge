"""Unit tests for Module base class in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, zeros
from tensorforge.nn import Linear, Module, Parameter


class SubModule(Module):
    """Simple child module for testing nested parameter registration."""

    def __init__(self) -> None:
        super().__init__()
        self.param_sub = Parameter([1.0, 2.0, 3.0])

    def forward(self, x: Tensor) -> Tensor:
        return x + self.param_sub


class CustomModel(Module):
    """Custom composite model for testing module hierarchy."""

    def __init__(self) -> None:
        super().__init__()
        self.w1 = Parameter([[1.0, 2.0], [3.0, 4.0]])
        self.b1 = Parameter([0.1, 0.2])
        self.sub = SubModule()
        self.non_param = "regular_attribute"

    def forward(self, x: Tensor) -> Tensor:
        h = x @ self.w1 + self.b1
        return self.sub(h)


class TestModule(unittest.TestCase):
    """Tests for Module parameter discovery, state tracking, and hierarchy traversal."""

    def test_parameter_registration_and_discovery(self):
        model = CustomModel()

        params = list(model.parameters())
        self.assertEqual(len(params), 3)

        named_params = dict(model.named_parameters())
        self.assertIn("w1", named_params)
        self.assertIn("b1", named_params)
        self.assertIn("sub.param_sub", named_params)
        self.assertIs(named_params["w1"], model.w1)
        self.assertIs(named_params["b1"], model.b1)
        self.assertIs(named_params["sub.param_sub"], model.sub.param_sub)

    def test_module_hierarchy(self):
        model = CustomModel()

        modules = list(model.modules())
        self.assertEqual(len(modules), 2)
        self.assertIs(modules[0], model)
        self.assertIs(modules[1], model.sub)

        named_mods = dict(model.named_modules())
        self.assertIn("", named_mods)
        self.assertIn("sub", named_mods)
        self.assertIs(named_mods[""], model)
        self.assertIs(named_mods["sub"], model.sub)

    def test_train_eval_modes(self):
        model = CustomModel()
        self.assertTrue(model.training)
        self.assertTrue(model.sub.training)

        model.eval()
        self.assertFalse(model.training)
        self.assertFalse(model.sub.training)

        model.train()
        self.assertTrue(model.training)
        self.assertTrue(model.sub.training)

    def test_zero_grad_recursive(self):
        model = CustomModel()
        x = tf.tensor([[1.0, 1.0]], dtype=float32)

        out = model(x)
        loss = out.sum()
        loss.backward()

        for p in model.parameters():
            self.assertIsNotNone(p.grad)

        model.zero_grad()
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_forward_not_implemented(self):
        base_mod = Module()
        with self.assertRaises(NotImplementedError):
            base_mod(tf.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
