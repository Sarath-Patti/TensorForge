"""Tests for parallel InferenceRuntime execution, memory plan inspection, and immutability."""

import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.inference.memory import MemoryPlan
from tensorforge.serialization import save_model


class TestParallelRuntime(unittest.TestCase):

    def test_runtime_thread_configuration_and_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 4),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path, num_threads=2)
            self.assertEqual(runtime.num_threads, 2)

            runtime.set_num_threads(4)
            self.assertEqual(runtime.num_threads, 4)

            runtime.compile(input_shape=(8, 16))
            self.assertIsInstance(runtime.memory_plan, MemoryPlan)
            self.assertGreater(runtime.memory_plan.num_regions, 0)

            summary = runtime.summary()
            self.assertEqual(summary["num_threads"], 4)
            self.assertEqual(summary["workspace_regions"], runtime.memory_plan.num_regions)
            self.assertEqual(summary["reused_buffers"], runtime.memory_plan.num_reused_buffers)
            self.assertEqual(summary["tensorforge_version"], tf.__version__)

    def test_runtime_parallel_prediction_parity(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime_st = InferenceRuntime.load(model_path, backend="native", num_threads=1).compile(input_shape=(32, 32))
            runtime_mt = InferenceRuntime.load(model_path, backend="native", num_threads=4).compile(input_shape=(32, 32))

            x = tf.randn((32, 32))
            out_st = runtime_st.predict(x)
            out_mt = runtime_mt.predict(x)

            np.testing.assert_allclose(out_mt.numpy(), out_st.numpy(), atol=1e-5, rtol=1e-5)

    def test_runtime_parallel_no_grad_and_immutability(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).set_num_threads(4).compile(input_shape=(16, 16))
            w_before = runtime.model[0].weight.numpy().copy()

            x = tf.randn((16, 16), requires_grad=True)
            out = runtime.predict(x)

            self.assertFalse(out.requires_grad)
            self.assertIsNone(out.grad_fn)

            w_after = runtime.model[0].weight.numpy()
            np.testing.assert_array_equal(w_before, w_after)


if __name__ == "__main__":
    unittest.main()
