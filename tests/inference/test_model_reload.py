"""Tests for safe atomic model reloading."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestModelReload(unittest.TestCase):

    def test_reload_model_atomic_swap(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_v1 = os.path.join(tmpdir, "model_v1.tfmodel")
            path_v2 = os.path.join(tmpdir, "model_v2.tfmodel")

            save_model(nn.Linear(4, 2), path_v1)
            save_model(nn.Linear(4, 3), path_v2)

            with InferenceServer() as server:
                server.load_model("classifier", path_v1, version="1")
                out1 = server.predict("classifier", tf.randn((1, 4)))
                self.assertEqual(out1.shape, (1, 2))

                # Reload version 1 with path_v2
                reloaded = server.reload_model("classifier", path_v2, version="1")
                self.assertEqual(reloaded.version, "1")

                out2 = server.predict("classifier", tf.randn((1, 4)))
                self.assertEqual(out2.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
