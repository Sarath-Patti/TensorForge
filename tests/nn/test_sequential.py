"""Unit tests for Sequential container in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, randn
from tensorforge.nn import Linear, ReLU, Sequential, Sigmoid
from tests.autograd.test_utils import gradcheck


class TestSequential(unittest.TestCase):
    """Tests for Sequential module composition, parameter discovery, and end-to-end execution."""

    def test_sequential_construction_and_indexing(self):
        fc1 = Linear(10, 20)
        act = ReLU()
        fc2 = Linear(20, 5)

        model = Sequential(fc1, act, fc2)
        self.assertEqual(len(model), 3)
        self.assertIs(model[0], fc1)
        self.assertIs(model[1], act)
        self.assertIs(model[2], fc2)

    def test_sequential_parameters(self):
        model = Sequential(
            Linear(10, 20, bias=True),
            ReLU(),
            Linear(20, 2, bias=True),
        )

        params = list(model.parameters())
        # 2 layers with weight + bias = 4 parameters
        self.assertEqual(len(params), 4)

        named_params = dict(model.named_parameters())
        self.assertIn("0.weight", named_params)
        self.assertIn("0.bias", named_params)
        self.assertIn("2.weight", named_params)
        self.assertIn("2.bias", named_params)

    def test_sequential_forward_backward(self):
        model = Sequential(
            Linear(4, 8),
            ReLU(),
            Linear(8, 2),
        )

        x = randn(5, 4, requires_grad=True)
        out = model(x)

        self.assertEqual(out.shape, (5, 2))
        loss = out.sum()
        loss.backward()

        # All parameters must have gradients
        for p in model.parameters():
            self.assertIsNotNone(p.grad)

        # Input tensor must have gradients
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, (5, 4))

        # Test zero_grad
        model.zero_grad()
        for p in model.parameters():
            self.assertIsNone(p.grad)

    def test_sequential_gradcheck(self):
        model = Sequential(
            Linear(3, 4),
            Sigmoid(),
            Linear(4, 2),
        )
        x = randn(3, 3, requires_grad=True)
        self.assertTrue(gradcheck(lambda inp: model(inp).sum(), [x]))


if __name__ == "__main__":
    unittest.main()
