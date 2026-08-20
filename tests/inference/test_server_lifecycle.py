"""Tests for InferenceServer lifecycle transitions, draining, and shutdown."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer, ServerLifecycleState
from tensorforge.serialization import save_model
from tensorforge.utils.validation import ServerClosedError


class TestServerLifecycle(unittest.TestCase):

    def test_rejection_after_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            server = InferenceServer()
            server.load_model("m1", model_path)

            server.close()
            self.assertEqual(server.state, ServerLifecycleState.CLOSED)

            with self.assertRaises(ServerClosedError):
                server.predict("m1", tf.randn((1, 4)))

            with self.assertRaises(ServerClosedError):
                server.load_model("m2", model_path)


if __name__ == "__main__":
    unittest.main()
