"""Unit tests for automatic differentiation through broadcasted operations."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, tensor, zeros
from tests.autograd.test_utils import gradcheck


class TestBroadcastGradients(unittest.TestCase):
    """Tests for gradient unbroadcasting reduction across various shape configurations."""

    def test_matrix_row_vector_broadcast(self):
        # x: (2, 3), b: (3,)
        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32, requires_grad=True)
        b = tensor([10.0, 20.0, 30.0], dtype=float32, requires_grad=True)

        y = x + b
        loss = y.sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (2, 3))
        self.assertEqual(b.grad.shape, (3,))
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        np.testing.assert_allclose(b.grad.numpy(), [2.0, 2.0, 2.0])

    def test_matrix_col_vector_broadcast(self):
        # x: (2, 3), col: (2, 1)
        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32, requires_grad=True)
        col = tensor([[10.0], [20.0]], dtype=float32, requires_grad=True)

        y = x * col
        loss = y.sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (2, 3))
        self.assertEqual(col.grad.shape, (2, 1))
        # dloss/dx = col broadcasted: [[10, 10, 10], [20, 20, 20]]
        np.testing.assert_allclose(x.grad.numpy(), [[10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])
        # dloss/dcol = sum(x, axis=1, keepdims=True): [[1+2+3], [4+5+6]] = [[6], [15]]
        np.testing.assert_allclose(col.grad.numpy(), [[6.0], [15.0]])

    def test_scalar_tensor_broadcast(self):
        # x: (3, 2), s: ()
        x = tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]], dtype=float32, requires_grad=True)
        s = tensor(3.0, dtype=float32, requires_grad=True)

        y = x * s
        loss = y.sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (3, 2))
        self.assertEqual(s.grad.shape, ())
        np.testing.assert_allclose(x.grad.numpy(), np.full((3, 2), 3.0))
        # s.grad = sum of all x elements = 2+4+6+8+10+12 = 42
        np.testing.assert_allclose(s.grad.numpy(), 42.0)

    def test_higher_dimensional_broadcast_gradcheck(self):
        # x: (2, 1, 4), y: (3, 4)
        def func(a: Tensor, b: Tensor) -> Tensor:
            return (a * b + (a / (b + 1.0))).sum()

        a = tensor(np.random.uniform(1.0, 3.0, (2, 1, 4)).astype(np.float32), requires_grad=True)
        b = tensor(np.random.uniform(1.0, 3.0, (3, 4)).astype(np.float32), requires_grad=True)

        self.assertTrue(gradcheck(func, [a, b]))


if __name__ == "__main__":
    unittest.main()
