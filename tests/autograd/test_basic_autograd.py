"""Unit tests for basic autograd operations, graph structure, and gradient accumulation."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, no_grad, tensor
from tests.autograd.test_utils import gradcheck


class TestBasicAutograd(unittest.TestCase):
    """Tests for core autograd behavior, leaf flags, and arithmetic backward rules."""

    def test_leaf_and_grad_fn_attributes(self):
        x = tensor([2.0, 3.0], requires_grad=True)
        self.assertTrue(x.is_leaf)
        self.assertTrue(x.requires_grad)
        self.assertIsNone(x.grad_fn)
        self.assertIsNone(x.grad)

        y = x * 2.0
        self.assertFalse(y.is_leaf)
        self.assertTrue(y.requires_grad)
        self.assertIsNotNone(y.grad_fn)
        self.assertEqual(y.grad_fn.name, "MulBackward")

        w = tensor([1.0, 2.0], requires_grad=False)
        self.assertTrue(w.is_leaf)
        self.assertFalse(w.requires_grad)

    def test_addition_backward(self):
        a = tensor([2.0, 3.0], requires_grad=True)
        b = tensor([4.0, 5.0], requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()

        self.assertIsNotNone(a.grad)
        self.assertIsNotNone(b.grad)
        np.testing.assert_allclose(a.grad.numpy(), [1.0, 1.0])
        np.testing.assert_allclose(b.grad.numpy(), [1.0, 1.0])

    def test_subtraction_backward(self):
        a = tensor([10.0, 20.0], requires_grad=True)
        b = tensor([3.0, 7.0], requires_grad=True)
        c = a - b
        loss = c.sum()
        loss.backward()

        np.testing.assert_allclose(a.grad.numpy(), [1.0, 1.0])
        np.testing.assert_allclose(b.grad.numpy(), [-1.0, -1.0])

    def test_multiplication_backward(self):
        a = tensor([2.0, -3.0], requires_grad=True)
        b = tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()

        np.testing.assert_allclose(a.grad.numpy(), [4.0, 5.0])
        np.testing.assert_allclose(b.grad.numpy(), [2.0, -3.0])

    def test_division_backward(self):
        a = tensor([6.0, 8.0], requires_grad=True)
        b = tensor([2.0, 4.0], requires_grad=True)
        c = a / b
        loss = c.sum()
        loss.backward()

        # dz/da = 1/b = [0.5, 0.25]
        # dz/db = -a / b^2 = [-6/4, -8/16] = [-1.5, -0.5]
        np.testing.assert_allclose(a.grad.numpy(), [0.5, 0.25])
        np.testing.assert_allclose(b.grad.numpy(), [-1.5, -0.5])

    def test_negation_backward(self):
        a = tensor([3.0, -5.0], requires_grad=True)
        b = -a
        loss = b.sum()
        loss.backward()

        np.testing.assert_allclose(a.grad.numpy(), [-1.0, -1.0])

    def test_scalar_operations_backward(self):
        x = tensor([2.0, 4.0], requires_grad=True)
        y = (x * 3.0 + 5.0) / 2.0
        loss = y.sum()
        loss.backward()

        # y = 1.5 * x + 2.5 -> dy/dx = 1.5
        np.testing.assert_allclose(x.grad.numpy(), [1.5, 1.5])

    def test_gradient_accumulation_multi_use(self):
        # y = x * x (x used twice in same operation)
        x = tensor([3.0], requires_grad=True)
        y = x * x
        y.backward()
        # dy/dx = 2 * 3 = 6.0
        np.testing.assert_allclose(x.grad.numpy(), [6.0])

        # Graph with branching paths: loss = x * x + 2 * x
        x.zero_grad()
        loss = x * x + x * 2.0
        loss.backward()
        # dloss/dx = 2*3 + 2 = 8.0
        np.testing.assert_allclose(x.grad.numpy(), [8.0])

    def test_successive_backward_accumulation(self):
        x = tensor([2.0], requires_grad=True)
        y1 = x * 3.0
        y1.backward()
        np.testing.assert_allclose(x.grad.numpy(), [3.0])

        y2 = x * 4.0
        y2.backward()
        # Gradients accumulate: 3 + 4 = 7
        np.testing.assert_allclose(x.grad.numpy(), [7.0])

        x.zero_grad()
        self.assertIsNone(x.grad)

    def test_detach(self):
        x = tensor([2.0, 3.0], requires_grad=True)
        y = x * 2.0
        d = y.detach()

        self.assertFalse(d.requires_grad)
        self.assertTrue(d.is_leaf)
        self.assertIsNone(d.grad_fn)

        z = d * 3.0
        z.sum().backward()
        # x should receive no gradients because d detached the graph
        self.assertIsNone(x.grad)

    def test_no_grad_context(self):
        x = tensor([2.0, 3.0], requires_grad=True)
        with no_grad():
            y = x * 4.0
            self.assertFalse(y.requires_grad)
            self.assertIsNone(y.grad_fn)

        # Outside no_grad context, tracking resumes
        z = x * 4.0
        self.assertTrue(z.requires_grad)
        self.assertIsNotNone(z.grad_fn)

    def test_error_cases(self):
        # 1. Non-scalar backward without gradient
        y = tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(RuntimeError):
            y.backward()

        # 2. Backward on tensor not requiring grad
        w = tensor([1.0, 2.0], requires_grad=False)
        with self.assertRaises(RuntimeError):
            w.backward()

        # 3. Shape mismatch on provided gradient
        x = tensor([1.0, 2.0], requires_grad=True)
        with self.assertRaises(ValueError):
            x.backward(gradient=tensor([1.0, 2.0, 3.0]))

    def test_gradcheck_composite_expression(self):
        def func(a: Tensor, b: Tensor) -> Tensor:
            return ((a * b + a) / (b + 1.0)).sum()

        a = tensor([[1.5, 2.5], [3.0, 4.0]], dtype=float32, requires_grad=True)
        b = tensor([[0.5, 1.2], [2.1, 0.8]], dtype=float32, requires_grad=True)
        self.assertTrue(gradcheck(func, [a, b]))


if __name__ == "__main__":
    unittest.main()
