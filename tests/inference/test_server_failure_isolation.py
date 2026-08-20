"""Tests verifying failure isolation between different models registered on the server."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model
from tensorforge.utils.validation import TensorForgeInputError


class TestServerFailureIsolation(unittest.TestCase):

    def test_failure_isolation_between_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_m1 = os.path.join(tmpdir, "m1.tfmodel")
            path_m2 = os.path.join(tmpdir, "m2.tfmodel")

            save_model(nn.Linear(4, 2), path_m1)
            save_model(nn.Linear(4, 2), path_m2)

            with InferenceServer() as server:
                server.load_model("m1", path_m1)
                server.load_model("m2", path_m2)

                # Invalid input for m1
                with self.assertRaises((TensorForgeInputError, ValueError)):
                    server.predict("m1", tf.randn((1, 10)))

                # m2 should remain unaffected and healthy!
                out2 = server.predict("m2", tf.randn((1, 4)))
                self.assertEqual(out2.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
