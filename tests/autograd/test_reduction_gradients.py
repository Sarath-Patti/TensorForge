"""Unit tests for reduction (sum, mean) and structural (reshape, transpose) gradients."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, randn, tensor
from tests.autograd.test_utils import gradcheck


class TestReductionAndStructuralGradients(unittest.TestCase):
    """Tests for sum, mean, reshape, and transpose backward rules."""

    def test_sum_all_gradients(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32, requires_grad=True)
        loss = x.sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (2, 2))
        np.testing.assert_allclose(x.grad.numpy(), [[1.0, 1.0], [1.0, 1.0]])

    def test_sum_axis_gradients(self):
        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32, requires_grad=True)
        # Sum along axis 0 -> (3,)
        s0 = x.sum(axis=0)
        # Multiply by weight [10, 20, 30]
        w = tensor([10.0, 20.0, 30.0], dtype=float32)
        loss = (s0 * w).sum()
        loss.backward()

        # dz/dx should be broadcasted w across axis 0
        expected = np.array([[10.0, 20.0, 30.0], [10.0, 20.0, 30.0]], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_mean_all_gradients(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32, requires_grad=True)  # numel = 4
        loss = x.mean()
        loss.backward()

        self.assertEqual(x.grad.shape, (2, 2))
        np.testing.assert_allclose(x.grad.numpy(), np.full((2, 2), 0.25))

    def test_mean_axis_gradients(self):
        x = tensor([[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]], dtype=float32, requires_grad=True)
        # Mean along axis 1 -> (2,) where each mean averages 3 elements
        m1 = x.mean(axis=1, keepdims=False)
        w = tensor([3.0, 6.0], dtype=float32)
        loss = (m1 * w).sum()
        loss.backward()

        # Row 0 scaled by 3/3 = 1.0, Row 1 scaled by 6/3 = 2.0
        expected = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]], dtype=np.float32)
        np.testing.assert_allclose(x.grad.numpy(), expected)

    def test_reshape_gradients(self):
        x = tensor(np.arange(12, dtype=np.float32), requires_grad=True)
        r = x.reshape(3, 4)
        loss = (r * 2.0).sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (12,))
        np.testing.assert_allclose(x.grad.numpy(), np.full(12, 2.0))

    def test_transpose_gradients(self):
        x = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32, requires_grad=True)
        t = x.T  # (3, 2)
        w = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float32)
        loss = (t * w).sum()
        loss.backward()

        self.assertEqual(x.grad.shape, (2, 3))
        # dloss/dt = w -> dloss/dx = w.T
        np.testing.assert_allclose(x.grad.numpy(), w.numpy().T)

    def test_nd_transpose_gradcheck(self):
        def func(a: Tensor) -> Tensor:
            perm = a.transpose(1, 2, 0)
            return (perm * perm + perm).sum()

        np.random.seed(123)
        a = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        self.assertTrue(gradcheck(func, [a]))


if __name__ == "__main__":
    unittest.main()
