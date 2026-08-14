"""Unit tests for Optimizer base class in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, tensor
from tensorforge.nn import Parameter
from tensorforge.optim import Optimizer


class DummyOptimizer(Optimizer):
    """Dummy optimizer for testing base class mechanics."""

    def step(self) -> None:
        for group in self.param_groups:
            lr = group.get("lr", 0.1)
            for p in group["params"]:
                if p.grad is not None:
                    new_val = p.numpy() - lr * p.grad.numpy()
                    np.copyto(p.storage.to_numpy(), new_val.astype(p.dtype.numpy_dtype, copy=False))


class TestOptimizerBase(unittest.TestCase):
    """Tests for Optimizer base functionality, parameter registration, and zero_grad."""

    def test_optimizer_validation(self):
        # 1. Parameter passed as a single Tensor rather than iterable
        p = Parameter([1.0, 2.0])
        with self.assertRaises(TypeError):
            DummyOptimizer(p, defaults={"lr": 0.01})  # type: ignore

        # 2. Empty parameter list
        with self.assertRaises(ValueError):
            DummyOptimizer([], defaults={"lr": 0.01})

        # 3. Non-tensor element in parameter list
        with self.assertRaises(TypeError):
            DummyOptimizer([p, "not_a_tensor"], defaults={"lr": 0.01})  # type: ignore

    def test_parameter_preservation_and_zero_grad(self):
        w = Parameter([[1.0, 2.0], [3.0, 4.0]], dtype=float32)
        b = Parameter([0.5, -0.5], dtype=float32)
        opt = DummyOptimizer([w, b], defaults={"lr": 0.1})

        self.assertEqual(len(opt.param_groups[0]["params"]), 2)

        # Forward and backward
        x = tensor([[1.0, 1.0]], dtype=float32)
        y = x @ w + b
        loss = y.sum()
        loss.backward()

        self.assertIsNotNone(w.grad)
        self.assertIsNotNone(b.grad)

        # Step
        opt.step()

        # Check parameter identity & properties preserved
        self.assertTrue(isinstance(w, Parameter))
        self.assertTrue(isinstance(b, Parameter))
        self.assertTrue(w.requires_grad)
        self.assertTrue(w.is_leaf)
        self.assertEqual(w.shape, (2, 2))

        # Check zero_grad
        opt.zero_grad()
        self.assertIsNone(w.grad)
        self.assertIsNone(b.grad)


if __name__ == "__main__":
    unittest.main()
