"""Tests for static shape propagation and validation."""

import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.graph import InferenceGraph
from tensorforge.inference.shapes import ShapePropagator
from tensorforge.utils.validation import ShapeError


def test_shape_propagation_linear():
    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.Linear(64, 32),
        nn.Linear(32, 10),
    )
    graph = InferenceGraph.from_module(model)

    shape_flow = ShapePropagator.propagate(graph, input_shape=(4, 16))
    assert len(shape_flow) == 3
    assert shape_flow[0] == ((4, 16), (4, 64))
    assert shape_flow[1] == ((4, 64), (4, 32))
    assert shape_flow[2] == ((4, 32), (4, 10))


def test_shape_propagation_activations():
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Sigmoid(),
        nn.Tanh(),
        nn.Softmax(dim=-1),
    )
    graph = InferenceGraph.from_module(model)

    shape_flow = ShapePropagator.propagate(graph, input_shape=(2, 8))
    assert len(shape_flow) == 5
    assert shape_flow[0] == ((2, 8), (2, 16))
    assert shape_flow[1] == ((2, 16), (2, 16))
    assert shape_flow[2] == ((2, 16), (2, 16))
    assert shape_flow[3] == ((2, 16), (2, 16))
    assert shape_flow[4] == ((2, 16), (2, 16))


def test_shape_propagation_incompatible_input_raises():
    model = nn.Sequential(nn.Linear(16, 32))
    graph = InferenceGraph.from_module(model)

    # Expected 16 features, got 12
    with pytest.raises(ShapeError, match="input has 12 features, but operator expected 16"):
        ShapePropagator.propagate(graph, input_shape=(4, 12))


def test_shape_propagation_zero_dim_raises():
    model = nn.Sequential(nn.Linear(16, 32))
    graph = InferenceGraph.from_module(model)

    with pytest.raises(ShapeError, match="invalid 0-dimensional"):
        ShapePropagator.propagate(graph, input_shape=())
