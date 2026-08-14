"""Unit tests for activation modules in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, randn, tensor
from tensorforge.nn import ReLU, Sigmoid, Softmax, Tanh
from tests.autograd.test_utils import gradcheck


class TestActivations(unittest.TestCase):
    """Tests for forward correctness and gradient propagation across activation modules."""

    def test_relu_forward_backward(self):
        relu = ReLU()
        x = tensor([-3.0, 0.0, 2.5, -0.5, 4.0], dtype=float32, requires_grad=True)

        out = relu(x)
        expected = np.array([0.0, 0.0, 2.5, 0.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected)

        loss = out.sum()
        loss.backward()
        expected_grad = np.array([0.0, 0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected_grad)

    def test_relu_gradcheck(self):
        relu = ReLU()
        # Avoid point at 0 for clean finite-difference checks
        x = tensor([1.5, -2.0, 3.2, -0.8], dtype=float32, requires_grad=True)
        self.assertTrue(gradcheck(lambda inp: relu(inp).sum(), [x]))

    def test_sigmoid_forward_backward(self):
        sigmoid = Sigmoid()
        x = tensor([0.0, 2.0, -2.0], dtype=float32, requires_grad=True)

        out = sigmoid(x)
        # sigmoid(0) = 0.5
        np.testing.assert_allclose(out.numpy()[0], 0.5, atol=1e-5)

        loss = out.sum()
        loss.backward()
        # dz/dx = s * (1 - s)
        s_np = out.numpy()
        np.testing.assert_allclose(x.grad.numpy(), s_np * (1.0 - s_np), atol=1e-5)

    def test_sigmoid_gradcheck(self):
        sigmoid = Sigmoid()
        x = tensor([0.5, -1.2, 2.0, -0.7], dtype=float32, requires_grad=True)
        self.assertTrue(gradcheck(lambda inp: sigmoid(inp).sum(), [x]))

    def test_tanh_forward_backward(self):
        tanh = Tanh()
        x = tensor([0.0, 1.0, -1.0], dtype=float32, requires_grad=True)

        out = tanh(x)
        expected = np.tanh(np.array([0.0, 1.0, -1.0], dtype=np.float32))
        np.testing.assert_allclose(out.numpy(), expected, atol=1e-5)

        loss = out.sum()
        loss.backward()
        t_np = out.numpy()
        np.testing.assert_allclose(x.grad.numpy(), 1.0 - t_np ** 2, atol=1e-5)

    def test_tanh_gradcheck(self):
        tanh = Tanh()
        x = tensor([-0.5, 0.8, -1.5, 1.2], dtype=float32, requires_grad=True)
        self.assertTrue(gradcheck(lambda inp: tanh(inp).sum(), [x]))

    def test_softmax_forward(self):
        softmax = Softmax(dim=-1)
        x = tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]], dtype=float32)

        out = softmax(x)
        # Sum of probabilities across dim=-1 must be 1.0
        np.testing.assert_allclose(out.sum(axis=-1).numpy(), [1.0, 1.0], atol=1e-5)
        # Row 1 with identical values must have equal probabilities 1/3
        np.testing.assert_allclose(out.numpy()[1], [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-5)

    def test_softmax_numerical_stability(self):
        softmax = Softmax(dim=-1)
        # Large logits that would overflow naive exp
        x = tensor([[1000.0, 1001.0, 1002.0]], dtype=float32)
        out = softmax(x)
        self.assertFalse(np.isnan(out.numpy()).any())
        self.assertFalse(np.isinf(out.numpy()).any())
        np.testing.assert_allclose(out.sum(axis=-1).numpy(), [1.0], atol=1e-5)

    def test_softmax_gradcheck(self):
        softmax = Softmax(dim=-1)
        x = tensor([[1.2, 0.5, -0.8], [0.1, 2.3, 1.0]], dtype=float32, requires_grad=True)
        # Test softmax followed by weighted sum reduction
        w = tensor([[1.0, 2.0, 3.0], [0.5, 1.5, 2.5]], dtype=float32)
        self.assertTrue(gradcheck(lambda inp: (softmax(inp) * w).sum(), [x]))


if __name__ == "__main__":
    unittest.main()
