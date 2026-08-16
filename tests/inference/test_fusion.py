"""Tests for inference graph extraction and operator fusion passes."""

import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.fusion import OperatorFusionPass
from tensorforge.inference.graph import InferenceGraph, InferenceNode


class TestFusion(unittest.TestCase):

    def test_fusion_linear_relu_pattern(self):
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
        )
        graph = InferenceGraph.from_module(model)
        self.assertEqual(len(graph), 4)

        opt_graph, stats = OperatorFusionPass.run(graph)
        self.assertEqual(len(opt_graph), 2)
        self.assertEqual(stats["fused_count"], 2)
        self.assertEqual(stats["fused_patterns"], ["Linear+ReLU", "Linear+ReLU"])
        self.assertEqual(opt_graph[0].op_type, "FusedLinear")
        self.assertEqual(opt_graph[0].attrs["activation"], "relu")
        self.assertEqual(opt_graph[1].op_type, "FusedLinear")
        self.assertEqual(opt_graph[1].attrs["activation"], "relu")

    def test_fusion_mixed_activations(self):
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Sigmoid(),
            nn.Linear(20, 15),
            nn.Tanh(),
            nn.Linear(15, 5),
            nn.Softmax(dim=-1),
        )
        graph = InferenceGraph.from_module(model)
        self.assertEqual(len(graph), 6)

        opt_graph, stats = OperatorFusionPass.run(graph)
        self.assertEqual(len(opt_graph), 3)
        self.assertEqual(stats["fused_count"], 3)
        self.assertEqual(stats["fused_patterns"], ["Linear+Sigmoid", "Linear+Tanh", "Linear+Softmax"])
        self.assertEqual(opt_graph[0].attrs["activation"], "sigmoid")
        self.assertEqual(opt_graph[1].attrs["activation"], "tanh")
        self.assertEqual(opt_graph[2].attrs["activation"], "softmax")

    def test_unsupported_patterns_remain_unfused(self):
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        graph = InferenceGraph.from_module(model)
        self.assertEqual(len(graph), 3)

        opt_graph, stats = OperatorFusionPass.run(graph)
        self.assertEqual(len(opt_graph), 2)
        self.assertEqual(stats["fused_count"], 1)
        self.assertEqual(stats["fused_patterns"], ["Linear+ReLU"])
        self.assertEqual(opt_graph[0].op_type, "Linear")
        self.assertEqual(opt_graph[1].op_type, "FusedLinear")


if __name__ == "__main__":
    unittest.main()
