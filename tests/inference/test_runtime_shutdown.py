"""Tests verifying graceful runtime shutdown, idempotent close, and lifecycle states."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, RuntimeState
from tensorforge.serialization import save_model
from tensorforge.utils.validation import RuntimeClosedError


class TestRuntimeShutdown(unittest.TestCase):

    def test_lifecycle_states_and_idempotent_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(8, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            self.assertEqual(runtime.lifecycle_state, RuntimeState.READY.value)
            self.assertTrue(runtime.is_ready)
            self.assertFalse(runtime.is_closed)

            # Valid predict
            _ = runtime.predict(tf.randn((2, 8)))

            # Close runtime
            runtime.close()
            self.assertEqual(runtime.lifecycle_state, RuntimeState.CLOSED.value)
            self.assertFalse(runtime.is_ready)
            self.assertTrue(runtime.is_closed)

            # Repeated close is idempotent
            runtime.close()
            self.assertTrue(runtime.is_closed)

            # Predict after close raises RuntimeClosedError
            with self.assertRaises(RuntimeClosedError):
                runtime.predict(tf.randn((2, 8)))

    def test_context_manager_clean_exit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            with InferenceRuntime.load(model_path) as runtime:
                self.assertTrue(runtime.is_ready)
                _ = runtime.predict(tf.randn((2, 4)))

            self.assertTrue(runtime.is_closed)
            self.assertEqual(runtime.lifecycle_state, "CLOSED")


if __name__ == "__main__":
    unittest.main()
