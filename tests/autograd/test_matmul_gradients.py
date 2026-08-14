"""Unit tests for matrix multiplication gradients across 1D, 2D, and batched tensors."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, tensor
from tests.autograd.test_utils import gradcheck


class TestMatmulGradients(unittest.TestCase):
    """Tests for matrix multiplication backward rules."""

    def test_2d_matmul_gradients(self):
        # A: (2, 3), B: (3, 2)
        a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32, requires_grad=True)
        b = tensor([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.5]], dtype=float32, requires_grad=True)

        c = a @ b  # (2, 2)
        loss = c.sum()
        loss.backward()

        # Analytical:
        # dC = ones(2, 2)
        # dA = dC @ B.T -> (2, 2) @ (2, 3) = (2, 3)
        # dB = A.T @ dC -> (3, 2) @ (2, 2) = (3, 2)
        dC = np.ones((2, 2), dtype=np.float32)
        expected_dA = dC @ b.numpy().T
        expected_dB = a.numpy().T @ dC

        self.assertEqual(a.grad.shape, (2, 3))
        self.assertEqual(b.grad.shape, (3, 2))
        np.testing.assert_allclose(a.grad.numpy(), expected_dA)
        np.testing.assert_allclose(b.grad.numpy(), expected_dB)

    def test_vector_matrix_matmul_gradients(self):
        # v: (3,), M: (3, 4)
        v = tensor([1.0, 2.0, 3.0], dtype=float32, requires_grad=True)
        m = tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)

        out = v @ m  # (4,)
        loss = out.sum()
        loss.backward()

        self.assertEqual(v.grad.shape, (3,))
        self.assertEqual(m.grad.shape, (3, 4))
        # dloss/dv = sum(M, axis=1)
        np.testing.assert_allclose(v.grad.numpy(), np.sum(m.numpy(), axis=1))
        # dloss/dm = ones(3, 1) @ v (outer product)
        np.testing.assert_allclose(m.grad.numpy(), np.outer(v.numpy(), np.ones(4, dtype=np.float32)))

    def test_matrix_vector_matmul_gradients(self):
        # M: (3, 4), v: (4,)
        m = tensor(np.arange(12, dtype=np.float32).reshape(3, 4), requires_grad=True)
        v = tensor([1.0, 2.0, 3.0, 4.0], dtype=float32, requires_grad=True)

        out = m @ v  # (3,)
        loss = out.sum()
        loss.backward()

        self.assertEqual(m.grad.shape, (3, 4))
        self.assertEqual(v.grad.shape, (4,))
        # dloss/dm = ones(3, 1) @ v.T
        np.testing.assert_allclose(m.grad.numpy(), np.outer(np.ones(3, dtype=np.float32), v.numpy()))
        # dloss/dv = sum(M, axis=0)
        np.testing.assert_allclose(v.grad.numpy(), np.sum(m.numpy(), axis=0))

    def test_dot_product_1d_gradients(self):
        # u: (4,), v: (4,)
        u = tensor([1.0, 2.0, 3.0, 4.0], dtype=float32, requires_grad=True)
        v = tensor([5.0, 6.0, 7.0, 8.0], dtype=float32, requires_grad=True)

        loss = u @ v  # ()
        loss.backward()

        self.assertEqual(u.grad.shape, (4,))
        self.assertEqual(v.grad.shape, (4,))
        np.testing.assert_allclose(u.grad.numpy(), v.numpy())
        np.testing.assert_allclose(v.grad.numpy(), u.numpy())

    def test_batched_matmul_gradcheck(self):
        def func(a: Tensor, b: Tensor) -> Tensor:
            return (a @ b).sum()

        np.random.seed(42)
        a = tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
        b = tensor(np.random.randn(2, 4, 5).astype(np.float32), requires_grad=True)

        self.assertTrue(gradcheck(func, [a, b]))


if __name__ == "__main__":
    unittest.main()
