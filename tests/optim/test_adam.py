"""Unit tests for Adam optimizer in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, tensor
from tensorforge.nn import Parameter
from tensorforge.optim import Adam


class TestAdam(unittest.TestCase):
    """Tests for Adam optimizer moment tracking, bias correction, and convergence."""

    def test_hyperparameter_validation(self):
        p = Parameter([1.0])
        with self.assertRaises(ValueError):
            Adam([p], lr=-0.001)
        with self.assertRaises(ValueError):
            Adam([p], betas=(1.0, 0.999))
        with self.assertRaises(ValueError):
            Adam([p], betas=(-0.1, 0.999))
        with self.assertRaises(ValueError):
            Adam([p], eps=-1e-8)
        with self.assertRaises(ValueError):
            Adam([p], weight_decay=-0.01)

    def test_adam_first_step_analytical(self):
        w = Parameter([1.0], dtype=float32)
        lr = 0.1
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        opt = Adam([w], lr=lr, betas=(beta1, beta2), eps=eps)

        # Grad = 2.0
        w.grad = tensor([2.0], dtype=float32)
        opt.step()

        # Analytical step 1:
        # m1 = 0.1 * 2.0 = 0.2, v1 = 0.001 * 4.0 = 0.004
        # m_hat = 0.2 / (1 - 0.9) = 2.0
        # v_hat = 0.004 / (1 - 0.999) = 4.0
        # update = 0.1 * 2.0 / (sqrt(4.0) + eps) = 0.2 / 2.0 = 0.1
        # w_new = 1.0 - 0.1 = 0.9
        np.testing.assert_allclose(w.numpy(), [0.9], atol=1e-5)

    def test_adam_quadratic_convergence(self):
        # Minimize f(w) = (w - 5)^2
        w = Parameter([0.0], dtype=float32)
        opt = Adam([w], lr=0.1)

        for _ in range(100):
            opt.zero_grad()
            diff = w - 5.0
            loss = diff * diff
            loss.backward()
            opt.step()

        # w should converge to 5.0
        np.testing.assert_allclose(w.numpy(), [5.0], atol=1e-2)

    def test_adam_multidimensional_parameter(self):
        w = Parameter([[1.0, 2.0], [3.0, 4.0]], dtype=float32)
        opt = Adam([w], lr=0.1)
        w.grad = tensor([[0.5, 0.5], [1.0, 1.0]], dtype=float32)
        opt.step()
        self.assertEqual(w.shape, (2, 2))
        self.assertEqual(w.storage.to_numpy().shape, (4,))


if __name__ == "__main__":
    unittest.main()

