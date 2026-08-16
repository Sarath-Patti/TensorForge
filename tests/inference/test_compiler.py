"""Tests for the Inference Compiler and ExecutionPlan generation."""

import pytest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.compiler import CompiledPlanCache, InferenceCompiler
from tensorforge.inference.fusion import OperatorFusionPass
from tensorforge.inference.graph import InferenceGraph
from tensorforge.quantization import quantize


def test_compiler_graph_to_execution_plan():
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 4),
        nn.Softmax(dim=-1),
    )
    raw_graph = InferenceGraph.from_module(model)
    opt_graph, _ = OperatorFusionPass.run(raw_graph)

    input_shape = (8, 16)
    plan = InferenceCompiler.compile(
        graph=opt_graph,
        input_shape=input_shape,
        backend="numpy",
        use_cache=False,
    )

    assert len(plan) == 2
    assert plan.input_shape == input_shape
    assert plan.output_shape == (8, 4)
    assert plan.target_backend == "numpy"
    assert plan.is_quantized is False

    step0 = plan[0]
    assert step0.op_type == "FusedLinear"
    assert step0.input_slot == -1  # External user input
    assert step0.output_slot == 0  # Workspace slot 0
    assert step0.input_shape == (8, 16)
    assert step0.output_shape == (8, 32)
    assert step0.backend_dispatch == "numpy_fused"
    assert step0.is_quantized is False

    step1 = plan[1]
    assert step1.op_type == "FusedLinear"
    assert step1.input_slot == 0  # From workspace slot 0
    assert step1.output_slot == 1  # Workspace slot 1
    assert step1.input_shape == (8, 32)
    assert step1.output_shape == (8, 4)
    assert step1.backend_dispatch == "numpy_fused"
    assert step1.is_quantized is False


def test_compiler_quantized_plan():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
    q_state = {name: quantize(param, scheme="symmetric") for name, param in model.named_parameters()}
    raw_graph = InferenceGraph.from_module(model, state_dict=q_state)
    opt_graph, _ = OperatorFusionPass.run(raw_graph)

    plan = InferenceCompiler.compile(opt_graph, input_shape=(4, 8), backend="numpy", is_quantized=True)
    assert plan.is_quantized is True
    assert len(plan) == 1
    assert plan[0].is_quantized is True


def test_compiler_cache():
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
    raw_graph = InferenceGraph.from_module(model)
    opt_graph, _ = OperatorFusionPass.run(raw_graph)

    plan1 = InferenceCompiler.compile(opt_graph, input_shape=(4, 8), backend="numpy", use_cache=True)
    plan2 = InferenceCompiler.compile(opt_graph, input_shape=(4, 8), backend="numpy", use_cache=True)

    # Identical configuration should return exact cached plan instance
    assert plan1 is plan2

    # Different shape should create a new plan
    plan3 = InferenceCompiler.compile(opt_graph, input_shape=(8, 8), backend="numpy", use_cache=True)
    assert plan3 is not plan1
    assert plan3.input_shape == (8, 8)


def test_compiler_deterministic_plan_summary():
    model = nn.Sequential(nn.Linear(10, 20), nn.Sigmoid())
    raw_graph = InferenceGraph.from_module(model)
    opt_graph, _ = OperatorFusionPass.run(raw_graph)

    plan = InferenceCompiler.compile(opt_graph, input_shape=(2, 10), backend="numpy")
    summary = plan.summary()

    assert "ExecutionPlan" in summary
    assert "FusedLinear" in summary
    assert "user_input -> slot_0" in summary
