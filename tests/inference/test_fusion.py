"""Tests for inference graph extraction and operator fusion passes."""

import numpy as np
import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.fusion import OperatorFusionPass
from tensorforge.inference.graph import InferenceGraph, InferenceNode


def test_fusion_linear_relu_pattern():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
        nn.ReLU(),
    )
    graph = InferenceGraph.from_module(model)
    assert len(graph) == 4

    opt_graph, stats = OperatorFusionPass.run(graph)
    assert len(opt_graph) == 2
    assert stats["fused_count"] == 2
    assert stats["fused_patterns"] == ["Linear+ReLU", "Linear+ReLU"]
    assert opt_graph[0].op_type == "FusedLinear"
    assert opt_graph[0].attrs["activation"] == "relu"
    assert opt_graph[1].op_type == "FusedLinear"
    assert opt_graph[1].attrs["activation"] == "relu"


def test_fusion_mixed_activations():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Sigmoid(),
        nn.Linear(20, 15),
        nn.Tanh(),
        nn.Linear(15, 5),
        nn.Softmax(dim=-1),
    )
    graph = InferenceGraph.from_module(model)
    assert len(graph) == 6

    opt_graph, stats = OperatorFusionPass.run(graph)
    assert len(opt_graph) == 3
    assert stats["fused_count"] == 3
    assert stats["fused_patterns"] == ["Linear+Sigmoid", "Linear+Tanh", "Linear+Softmax"]
    assert opt_graph[0].attrs["activation"] == "sigmoid"
    assert opt_graph[1].attrs["activation"] == "tanh"
    assert opt_graph[2].attrs["activation"] == "softmax"


def test_unsupported_patterns_remain_unfused():
    # Standalone Linear and unsupported sequences remain untouched
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.Linear(16, 8),  # Linear directly followed by Linear (not fusible)
        nn.ReLU(),
    )
    graph = InferenceGraph.from_module(model)
    assert len(graph) == 3

    opt_graph, stats = OperatorFusionPass.run(graph)
    assert len(opt_graph) == 2  # [Linear, FusedLinear(ReLU)]
    assert stats["fused_count"] == 1
    assert opt_graph[0].op_type == "Linear"
    assert opt_graph[1].op_type == "FusedLinear"
