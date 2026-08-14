"""Unit tests for loss function modules in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, tensor
from tensorforge.nn import CrossEntropyLoss, MSELoss
from tests.autograd.test_utils import gradcheck


class TestLosses(unittest.TestCase):
    """Tests for MSELoss and CrossEntropyLoss modules."""

    def test_mse_loss_forward_backward(self):
        mse_mean = MSELoss(reduction="mean")
        mse_sum = MSELoss(reduction="sum")
        mse_none = MSELoss(reduction="none")

        pred = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32, requires_grad=True)
        target = tensor([[1.5, 2.0], [2.0, 5.0]], dtype=float32)

        # diff = [-0.5, 0.0, 1.0, -1.0], sq = [0.25, 0.0, 1.0, 1.0]
        # sum = 2.25, mean = 2.25 / 4 = 0.5625
        loss_mean = mse_mean(pred, target)
        loss_sum = mse_sum(pred, target)
        loss_none = mse_none(pred, target)

        np.testing.assert_allclose(loss_mean.item(), 0.5625)
        np.testing.assert_allclose(loss_sum.item(), 2.25)
        np.testing.assert_allclose(loss_none.numpy(), [[0.25, 0.0], [1.0, 1.0]])

        # Test backward
        loss_mean.backward()
        # dloss/dpred = 2 * (pred - target) / 4 = [-0.25, 0.0, 0.5, -0.5]
        expected_grad = np.array([[-0.25, 0.0], [0.5, -0.5]], dtype=np.float32)
        np.testing.assert_allclose(pred.grad.numpy(), expected_grad)

    def test_mse_gradcheck(self):
        mse = MSELoss()
        pred = tensor([[2.0, -1.0], [0.5, 3.2]], dtype=float32, requires_grad=True)
        target = tensor([[1.0, 0.0], [1.5, 2.0]], dtype=float32)
        self.assertTrue(gradcheck(lambda p: mse(p, target), [pred]))

    def test_cross_entropy_batched(self):
        loss_fn = CrossEntropyLoss(reduction="mean")

        # 3 samples, 4 classes
        logits = tensor(
            [
                [2.0, 1.0, 0.1, -1.0],
                [0.5, 3.0, 1.2, 0.0],
                [-0.5, 0.2, 2.5, 1.0],
            ],
            dtype=float32,
            requires_grad=True,
        )
        targets = [0, 1, 2]  # Target class for each sample

        loss = loss_fn(logits, targets)
        self.assertFalse(np.isnan(loss.item()))
        self.assertTrue(loss.item() > 0)

        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (3, 4))
        # Sum of gradient for each row in cross-entropy is 0
        np.testing.assert_allclose(logits.grad.sum(axis=-1).numpy(), [0.0, 0.0, 0.0], atol=1e-5)

    def test_cross_entropy_single_sample(self):
        loss_fn = CrossEntropyLoss()
        logits = tensor([1.0, 2.0, 0.5], dtype=float32, requires_grad=True)
        target = 1

        loss = loss_fn(logits, target)
        self.assertTrue(loss.numel == 1)
        loss.backward()
        self.assertEqual(logits.grad.shape, (3,))

    def test_cross_entropy_numerical_stability(self):
        loss_fn = CrossEntropyLoss()
        # Huge logits that would overflow naive exp
        logits = tensor([[1000.0, 1001.0, 1002.0]], dtype=float32, requires_grad=True)
        targets = [2]

        loss = loss_fn(logits, targets)
        self.assertFalse(np.isnan(loss.item()))
        self.assertFalse(np.isinf(loss.item()))

    def test_cross_entropy_gradcheck(self):
        loss_fn = CrossEntropyLoss(reduction="mean")
        logits = tensor(
            [
                [1.5, 0.8, -0.5],
                [-0.2, 2.1, 1.0],
            ],
            dtype=float32,
            requires_grad=True,
        )
        targets = [1, 2]

        self.assertTrue(gradcheck(lambda inp: loss_fn(inp, targets), [logits]))


    def test_cross_entropy_invalid_targets(self):
        loss_fn = CrossEntropyLoss()
        logits = tensor([[1.0, 2.0, 3.0]], dtype=float32)
        # Class index 3 is out of bounds for 3 classes [0, 1, 2]
        with self.assertRaises(IndexError):
            loss_fn(logits, [3])
        # Negative class index
        with self.assertRaises(IndexError):
            loss_fn(logits, [-1])


if __name__ == "__main__":
    unittest.main()

