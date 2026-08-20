"""Unit tests for request deadlines and cancellation in TensorForge v1.9 (Static Creation Only)."""

import time
import unittest
from tensorforge.inference import (
    InferenceScheduler,
    InferenceServer,
    InferenceRuntime,
    RequestState,
    RequestDeadlineExceededError,
    RequestCancelledError,
)
import tensorforge.nn as nn
from tensorforge.tensor.tensor import Tensor
import numpy as np


class TestReliabilityDeadlines(unittest.TestCase):
    """Test suite verifying monotonic deadlines and cancellation behavior."""

    def test_request_deadline_expiration(self):
        """Verify request expires when deadline is exceeded before execution."""
        model = nn.Linear(4, 4)
        runtime = InferenceRuntime(model, {"name": "test_model"})
        scheduler = InferenceScheduler(runtime)

        # Submit request with ultra-short deadline
        future = scheduler.submit(np.ones((1, 4), dtype=np.float32), timeout_ms=0.001)
        time.sleep(0.01)

        with self.assertRaises(RequestDeadlineExceededError):
            future.result()

        scheduler.close()

    def test_request_explicit_cancellation(self):
        """Verify explicit cancellation before batch processing."""
        model = nn.Linear(4, 4)
        runtime = InferenceRuntime(model, {"name": "test_model"})
        scheduler = InferenceScheduler(runtime)

        future = scheduler.submit(np.ones((1, 4), dtype=np.float32))
        cancelled = future.cancel()
        self.assertTrue(cancelled)

        with self.assertRaises(RequestCancelledError):
            future.result()

        scheduler.close()


if __name__ == "__main__":
    unittest.main()
