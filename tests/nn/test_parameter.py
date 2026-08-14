"""Unit tests for Parameter abstraction in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, float64
from tensorforge.nn import Parameter


class TestParameter(unittest.TestCase):
    """Tests for Parameter class behavior and autograd integration."""

    def test_parameter_creation_and_defaults(self):
        p = Parameter([1.0, 2.0, 3.0], dtype=float32)
        self.assertTrue(isinstance(p, Tensor))
        self.assertTrue(isinstance(p, Parameter))
        self.assertTrue(p.requires_grad)
        self.assertTrue(p.is_leaf)
        self.assertIsNone(p.grad)
        self.assertEqual(p.shape, (3,))
        self.assertEqual(p.dtype, float32)

    def test_parameter_from_tensor(self):
        t = tf.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float64, requires_grad=False)
        p = Parameter(t)
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.shape, (2, 2))
        self.assertEqual(p.dtype, float64)
        np.testing.assert_allclose(p.numpy(), [[1.0, 2.0], [3.0, 4.0]])

    def test_parameter_operations_and_gradient(self):
        w = Parameter([[2.0, -1.0], [0.5, 3.0]], dtype=float32)
        x = tf.tensor([[1.0, 2.0]], dtype=float32, requires_grad=False)

        # Forward
        y = x @ w.T
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        self.assertEqual(w.grad.shape, (2, 2))
        # dy/dw: y0 = 1*w00 + 2*w01 -> dy0/dw0 = [1, 2], y1 = 1*w10 + 2*w11 -> dy1/dw1 = [1, 2]
        np.testing.assert_allclose(w.grad.numpy(), [[1.0, 2.0], [1.0, 2.0]])

        # Reset grad
        w.zero_grad()
        self.assertIsNone(w.grad)

    def test_parameter_repr(self):
        p = Parameter([1.0, 2.0], dtype=float32)
        rep = repr(p)
        self.assertTrue(rep.startswith("Parameter("))
        self.assertIn("requires_grad=True", rep)


if __name__ == "__main__":
    unittest.main()
