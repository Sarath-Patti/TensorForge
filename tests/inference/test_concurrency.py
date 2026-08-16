"""Tests for multi-threaded concurrency, workspace isolation, determinism, and context pooling."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import unittest
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model


class TestConcurrency(unittest.TestCase):

    def test_concurrent_predictions_same_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mlp.tfmodel")
            in_dim, hidden_dim, out_dim = 16, 32, 4

            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
                nn.Softmax(dim=-1),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(8, in_dim))

            np.random.seed(42)
            num_tasks = 20
            test_inputs = [np.random.randn(8, in_dim).astype(np.float32) for _ in range(num_tasks)]

            # Compute single-threaded reference outputs
            ref_outputs = [runtime.predict(x).numpy() for x in test_inputs]

            # Execute concurrently across 8 worker threads
            results = [None] * num_tasks
            with ThreadPoolExecutor(max_workers=8) as executor:
                future_to_idx = {executor.submit(runtime.predict, test_inputs[i]): i for i in range(num_tasks)}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result().numpy()

            # Check determinism and numerical bit-exact parity
            for i in range(num_tasks):
                np.testing.assert_allclose(results[i], ref_outputs[i], atol=1e-6, rtol=1e-6)

            # Context pool should have recycled all contexts (active count is 0)
            self.assertEqual(runtime.active_contexts, 0)
            self.assertGreaterEqual(runtime.prediction_count, num_tasks)

    def test_concurrent_predictions_mixed_batch_sizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mlp_mixed.tfmodel")
            in_dim, hidden_dim, out_dim = 16, 32, 4

            model = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(1, in_dim))

            batch_sizes = [1, 2, 4, 8, 16, 32, 1, 4, 8, 16]
            inputs = [np.random.randn(b, in_dim).astype(np.float32) for b in batch_sizes]
            ref_outputs = [runtime.predict(x).numpy() for x in inputs]

            results = [None] * len(batch_sizes)
            with ThreadPoolExecutor(max_workers=6) as executor:
                future_to_idx = {executor.submit(runtime.predict, inputs[i]): i for i in range(len(batch_sizes))}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result().numpy()

            for i in range(len(batch_sizes)):
                np.testing.assert_allclose(results[i], ref_outputs[i], atol=1e-6, rtol=1e-6)

            self.assertEqual(runtime.active_contexts, 0)

    def test_parameter_immutability_under_concurrency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mlp_immutable.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(4, 8))
            w_initial = runtime.model[0].weight.numpy().copy()

            inputs = [np.random.randn(4, 8).astype(np.float32) for _ in range(25)]

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(runtime.predict, x) for x in inputs]
                for f in as_completed(futures):
                    _ = f.result()

            w_final = runtime.model[0].weight.numpy()
            np.testing.assert_array_equal(w_initial, w_final)

    def test_concurrent_native_and_numpy_backend_parity(self):
        if not is_native_available():
            self.skipTest("Native C++ extension not compiled")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "mlp_parity.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.Tanh(),
            )
            save_model(model, model_path)

            runtime_numpy = InferenceRuntime.load(model_path, backend="numpy").compile(input_shape=(16, 16))
            runtime_native = InferenceRuntime.load(model_path, backend="native", num_threads=4).compile(input_shape=(16, 16))

            inputs = [np.random.randn(16, 16).astype(np.float32) for _ in range(10)]

            out_numpy = [runtime_numpy.predict(x).numpy() for x in inputs]

            out_native = [None] * len(inputs)
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_idx = {executor.submit(runtime_native.predict, inputs[i]): i for i in range(len(inputs))}
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    out_native[idx] = future.result().numpy()

            for i in range(len(inputs)):
                np.testing.assert_allclose(out_native[i], out_numpy[i], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
