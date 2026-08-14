"""Unit tests for Linear layer in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import Tensor, float32, randn, tensor
from tensorforge.nn import Linear
from tests.autograd.test_utils import gradcheck


class TestLinear(unittest.TestCase):
    """Tests for Linear layer forward transformations, parameter shapes, and autograd gradients."""

    def test_parameter_shapes(self):
        layer = Linear(in_features=4, out_features=3, bias=True)
        self.assertEqual(layer.weight.shape, (3, 4))
        self.assertIsNotNone(layer.bias)
        self.assertEqual(layer.bias.shape, (3,))

        layer_no_bias = Linear(in_features=4, out_features=3, bias=False)
        self.assertEqual(layer_no_bias.weight.shape, (3, 4))
        self.assertIsNone(layer_no_bias.bias)

    def test_single_and_batched_forward_shape(self):
        layer = Linear(in_features=5, out_features=2)

        # 1D input (5,) -> (2,)
        x_1d = randn(5)
        out_1d = layer(x_1d)
        self.assertEqual(out_1d.shape, (2,))

        # 2D batch (8, 5) -> (8, 2)
        x_2d = randn(8, 5)
        out_2d = layer(x_2d)
        self.assertEqual(out_2d.shape, (8, 2))

        # 3D tensor (3, 4, 5) -> (3, 4, 2)
        x_3d = randn(3, 4, 5)
        out_3d = layer(x_3d)
        self.assertEqual(out_3d.shape, (3, 4, 2))

    def test_linear_forward_numerical_correctness(self):
        layer = Linear(in_features=3, out_features=2, bias=True)
        # Set deterministic weights & bias
        layer.weight._storage = tf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32).storage
        layer.bias._storage = tf.tensor([0.5, -0.5], dtype=float32).storage

        x = tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 1.0]], dtype=float32)
        # Row 0: [1*1 + 0*2 + 1*3 + 0.5, 1*4 + 0*5 + 1*6 - 0.5] = [4.5, 9.5]
        # Row 1: [0*1 + 2*2 + 1*3 + 0.5, 0*4 + 2*5 + 1*6 - 0.5] = [7.5, 15.5]
        out = layer(x)
        expected = np.array([[4.5, 9.5], [7.5, 15.5]], dtype=np.float32)
        np.testing.assert_allclose(out.numpy(), expected)

    def test_linear_gradcheck(self):
        layer = Linear(in_features=3, out_features=2, bias=True)

        def func(inp: Tensor) -> Tensor:
            return layer(inp).sum()

        x = randn(4, 3, requires_grad=True)
        self.assertTrue(gradcheck(func, [x]))

        # Also check weight and bias gradients
        def func_wb(w: Tensor, b: Tensor) -> Tensor:
            return (x @ w.T + b).sum()

        self.assertTrue(gradcheck(func_wb, [layer.weight, layer.bias]))


if __name__ == "__main__":
    unittest.main()
