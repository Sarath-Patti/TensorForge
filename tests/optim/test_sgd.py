"""Unit tests for SGD optimizer in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, tensor
from tensorforge.nn import Parameter
from tensorforge.optim import SGD


class TestSGD(unittest.TestCase):
    """Tests for SGD optimizer with vanilla updates, momentum, and weight decay."""

    def test_hyperparameter_validation(self):
        p = Parameter([1.0])
        with self.assertRaises(ValueError):
            SGD([p], lr=-0.01)
        with self.assertRaises(ValueError):
            SGD([p], lr=0.01, momentum=-0.1)
        with self.assertRaises(ValueError):
            SGD([p], lr=0.01, weight_decay=-0.001)

    def test_vanilla_sgd_step(self):
        w = Parameter([2.0, -3.0], dtype=float32)
        opt = SGD([w], lr=0.1)

        # Loss = sum(w^2) / 2 -> dL/dw = w
        loss = (w * w).sum() * 0.5
        loss.backward()
        np.testing.assert_allclose(w.grad.numpy(), [2.0, -3.0])

        opt.step()
        # w_new = w - lr * grad = [2.0 - 0.2, -3.0 - (-0.3)] = [1.8, -2.7]
        np.testing.assert_allclose(w.numpy(), [1.8, -2.7])

    def test_sgd_with_momentum(self):
        w = Parameter([1.0], dtype=float32)
        opt = SGD([w], lr=0.1, momentum=0.9)

        # Step 1: grad = 2.0
        # v1 = 2.0, w1 = 1.0 - 0.1 * 2.0 = 0.8
        w.grad = tensor([2.0], dtype=float32)
        opt.step()
        np.testing.assert_allclose(w.numpy(), [0.8])

        # Step 2: grad = 1.0
        # v2 = 0.9 * 2.0 + 1.0 = 2.8, w2 = 0.8 - 0.1 * 2.8 = 0.52
        w.grad = tensor([1.0], dtype=float32)
        opt.step()
        np.testing.assert_allclose(w.numpy(), [0.52])

    def test_sgd_with_weight_decay(self):
        w = Parameter([2.0], dtype=float32)
        opt = SGD([w], lr=0.1, weight_decay=0.05)

        # grad = 1.0
        # d_p = grad + weight_decay * w = 1.0 + 0.05 * 2.0 = 1.1
        # w_new = 2.0 - 0.1 * 1.1 = 1.89
        w.grad = tensor([1.0], dtype=float32)
        opt.step()
        np.testing.assert_allclose(w.numpy(), [1.89])

    def test_quadratic_optimization_convergence(self):
        # Minimize f(w) = (w - 3)^2
        w = Parameter([0.0], dtype=float32)
        opt = SGD([w], lr=0.2)

        for _ in range(50):
            opt.zero_grad()
            diff = w - 3.0
            loss = diff * diff
            loss.backward()
            opt.step()

        # w should converge close to 3.0
        np.testing.assert_allclose(w.numpy(), [3.0], atol=1e-3)

    def test_sgd_multidimensional_parameter(self):
        w = Parameter([[1.0, 2.0], [3.0, 4.0]], dtype=float32)
        opt = SGD([w], lr=0.1)
        w.grad = tensor([[0.5, 0.5], [1.0, 1.0]], dtype=float32)
        opt.step()
        self.assertEqual(w.shape, (2, 2))
        np.testing.assert_allclose(w.numpy(), [[0.95, 1.95], [2.9, 3.9]])


if __name__ == "__main__":
    unittest.main()

