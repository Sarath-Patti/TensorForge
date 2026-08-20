"""Tests for InferenceServer request routing (predict and submit)."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceFuture, InferenceServer
from tensorforge.serialization import save_model


class TestServerRouting(unittest.TestCase):

    def test_sync_and_async_routing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_m1 = os.path.join(tmpdir, "m1.tfmodel")
            path_m2 = os.path.join(tmpdir, "m2.tfmodel")

            save_model(nn.Linear(4, 2), path_m1)
            save_model(nn.Linear(6, 3), path_m2)

            with InferenceServer() as server:
                server.load_model("model_a", path_m1, version="1")
                server.load_model("model_b", path_m2, version="1")

                # Sync predict
                out_a = server.predict("model_a", tf.randn((2, 4)))
                self.assertEqual(out_a.shape, (2, 2))

                # Async submit
                fut_b = server.submit("model_b", tf.randn((3, 6)))
                self.assertIsInstance(fut_b, InferenceFuture)
                out_b = fut_b.result(timeout=2.0)
                self.assertEqual(out_b.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
