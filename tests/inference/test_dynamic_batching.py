"""Tests verifying dynamic batch assembly, timeout flushing, and output demultiplexing."""

import os
import tempfile
import time
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler
from tensorforge.serialization import save_model


class TestDynamicBatching(unittest.TestCase):

    def test_dynamic_batch_assembly_and_demultiplexing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 4)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            with InferenceScheduler(runtime, max_batch_size=16, batch_timeout_ms=50.0) as scheduler:
                x_a = tf.randn((2, 8))
                x_b = tf.randn((4, 8))
                x_c = tf.randn((1, 8))

                fut_a = scheduler.submit(x_a)
                fut_b = scheduler.submit(x_b)
                fut_c = scheduler.submit(x_c)

                # Wait for all futures
                out_a = fut_a.result(timeout=2.0)
                out_b = fut_b.result(timeout=2.0)
                out_c = fut_c.result(timeout=2.0)

                # Verify individual output shapes
                self.assertEqual(out_a.shape, (2, 4))
                self.assertEqual(out_b.shape, (4, 4))
                self.assertEqual(out_c.shape, (1, 4))

                # Verify accuracy parity against direct individual predictions
                direct_a = runtime.predict(x_a)
                direct_b = runtime.predict(x_b)
                direct_c = runtime.predict(x_c)

                np.testing.assert_allclose(out_a.numpy(), direct_a.numpy(), rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(out_b.numpy(), direct_b.numpy(), rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(out_c.numpy(), direct_c.numpy(), rtol=1e-5, atol=1e-5)

            runtime.close()

    def test_batch_timeout_trigger(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            # max_batch_size = 100, but timeout = 5 ms -> single request will flush on timeout
            with InferenceScheduler(runtime, max_batch_size=100, batch_timeout_ms=5.0) as scheduler:
                t0 = time.perf_counter()
                out = scheduler.predict(tf.randn((1, 4)))
                t1 = time.perf_counter()

                self.assertEqual(out.shape, (1, 2))
                self.assertLess(t1 - t0, 1.0)  # should return promptly upon timeout

            runtime.close()


if __name__ == "__main__":
    unittest.main()
