"""Tests for scheduler failure recovery and fault isolation across dynamic batches."""

import os
import tempfile
import unittest
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceScheduler, RuntimeLimits
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeLimitError, TensorForgeInputError


class TestSchedulerFailureRecovery(unittest.TestCase):

    def test_batch_failure_isolation_and_subsequent_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            # Limit underlying runtime to max_batch_size=4
            runtime_limits = RuntimeLimits(max_batch_size=4)
            runtime = InferenceRuntime.load(model_path, limits=runtime_limits)

            with InferenceScheduler(runtime, max_batch_size=8, batch_timeout_ms=5.0) as scheduler:
                # 1. Dispatch oversized batch request that will cause runtime failure (batch 6 > limit 4)
                fut_fail = scheduler.submit(tf.randn((6, 8)))

                with self.assertRaises(RuntimeLimitError):
                    _ = fut_fail.result(timeout=2.0)

                # 2. Dispatch valid request (batch 2 <= limit 4) -> must succeed cleanly
                fut_succ = scheduler.submit(tf.randn((2, 8)))
                out_succ = fut_succ.result(timeout=2.0)
                self.assertEqual(out_succ.shape, (2, 2))

            runtime.close()


if __name__ == "__main__":
    unittest.main()
