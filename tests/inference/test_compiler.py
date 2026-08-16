"""Tests for the Inference Compiler and ExecutionPlan generation."""

import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference.compiler import CompiledPlanCache, InferenceCompiler
from tensorforge.inference.fusion import OperatorFusionPass
from tensorforge.inference.graph import InferenceGraph
from tensorforge.quantization import quantize


class TestCompiler(unittest.TestCase):

    def test_compiler_graph_to_execution_plan(self):
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

        self.assertEqual(len(plan), 2)
        self.assertEqual(plan.input_shape, input_shape)
        self.assertEqual(plan.output_shape, (8, 4))
        self.assertEqual(plan.target_backend, "numpy")
        self.assertFalse(plan.is_quantized)

        step0 = plan[0]
        self.assertEqual(step0.op_type, "FusedLinear")
        self.assertEqual(step0.input_slot, -1)  # External user input
        self.assertEqual(step0.output_slot, 0)  # Workspace slot 0
        self.assertEqual(step0.input_shape, (8, 16))
        self.assertEqual(step0.output_shape, (8, 32))
        self.assertEqual(step0.backend_dispatch, "numpy_fused")
        self.assertFalse(step0.is_quantized)

        step1 = plan[1]
        self.assertEqual(step1.op_type, "FusedLinear")
        self.assertEqual(step1.input_slot, 0)  # From workspace slot 0
        self.assertEqual(step1.output_slot, 1)  # Workspace slot 1
        self.assertEqual(step1.input_shape, (8, 32))
        self.assertEqual(step1.output_shape, (8, 4))
        self.assertEqual(step1.backend_dispatch, "numpy_fused")
        self.assertFalse(step1.is_quantized)

    def test_compiler_quantized_plan(self):
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        q_state = {name: quantize(param, scheme="symmetric") for name, param in model.named_parameters()}
        raw_graph = InferenceGraph.from_module(model, state_dict=q_state)
        opt_graph, _ = OperatorFusionPass.run(raw_graph)

        plan = InferenceCompiler.compile(
            graph=opt_graph,
            input_shape=(4, 8),
            is_quantized=True,
            use_cache=False,
        )

        self.assertEqual(len(plan), 1)
        self.assertTrue(plan[0].is_quantized)
        self.assertTrue(plan.is_quantized)

    def test_plan_cache_reuse(self):
        cache = CompiledPlanCache()
        model = nn.Sequential(nn.Linear(4, 8))
        graph = InferenceGraph.from_module(model)

        plan = InferenceCompiler.compile(graph, (2, 4), use_cache=False)
        graph_id = id(graph)

        self.assertIsNone(cache.get(graph_id, (2, 4), plan.dtype, plan.target_backend, False, plan.num_threads))

        cache.put(graph_id, (2, 4), plan.dtype, plan.target_backend, False, plan, plan.num_threads)
        cached = cache.get(graph_id, (2, 4), plan.dtype, plan.target_backend, False, plan.num_threads)
        self.assertIs(cached, plan)

        cache.clear()
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
