"""Tests for static shape propagation and validation."""

import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.graph import InferenceGraph
from tensorforge.inference.shapes import ShapePropagator
from tensorforge.utils.validation import ShapeError


class TestShapes(unittest.TestCase):

    def test_shape_propagation_linear(self):
        model = nn.Sequential(
            nn.Linear(16, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 10),
        )
        graph = InferenceGraph.from_module(model)

        shape_flow = ShapePropagator.propagate(graph, input_shape=(4, 16))
        self.assertEqual(len(shape_flow), 3)
        self.assertEqual(shape_flow[0], ((4, 16), (4, 64)))
        self.assertEqual(shape_flow[1], ((4, 64), (4, 32)))
        self.assertEqual(shape_flow[2], ((4, 32), (4, 10)))

    def test_shape_propagation_activations(self):
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Sigmoid(),
            nn.Tanh(),
            nn.Softmax(dim=-1),
        )
        graph = InferenceGraph.from_module(model)

        shape_flow = ShapePropagator.propagate(graph, input_shape=(2, 8))
        self.assertEqual(len(shape_flow), 5)
        self.assertEqual(shape_flow[0], ((2, 8), (2, 16)))
        self.assertEqual(shape_flow[1], ((2, 16), (2, 16)))
        self.assertEqual(shape_flow[2], ((2, 16), (2, 16)))
        self.assertEqual(shape_flow[3], ((2, 16), (2, 16)))
        self.assertEqual(shape_flow[4], ((2, 16), (2, 16)))

    def test_shape_propagation_incompatible_input_raises(self):
        model = nn.Sequential(nn.Linear(16, 32))
        graph = InferenceGraph.from_module(model)

        # Expected 16 features, got 12
        with self.assertRaises(ShapeError):
            ShapePropagator.propagate(graph, input_shape=(4, 12))

    def test_shape_propagation_zero_dim_raises(self):
        model = nn.Sequential(nn.Linear(16, 32))
        graph = InferenceGraph.from_module(model)

        with self.assertRaises(ShapeError):
            ShapePropagator.propagate(graph, input_shape=())


if __name__ == "__main__":
    unittest.main()
