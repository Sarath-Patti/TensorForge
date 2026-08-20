"""Tests for multi-version model management and active version switching."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestModelVersions(unittest.TestCase):

    def test_multi_version_loading_and_switching(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_v1 = os.path.join(tmpdir, "model_v1.tfmodel")
            path_v2 = os.path.join(tmpdir, "model_v2.tfmodel")

            save_model(nn.Linear(8, 4), path_v1)
            save_model(nn.Linear(8, 4), path_v2)

            with InferenceServer() as server:
                server.load_model("classifier", path_v1, version="1", active=True)
                server.load_model("classifier", path_v2, version="2", active=False)

                self.assertEqual(server.get_active_version("classifier"), "1")

                # Predict explicit version 2
                res_v2 = server.predict("classifier", tf.randn((1, 8)), version="2")
                self.assertEqual(res_v2.shape, (1, 4))

                # Switch active version to 2
                server.set_active_version("classifier", "2")
                self.assertEqual(server.get_active_version("classifier"), "2")

                # Default predict now uses v2
                res_active = server.predict("classifier", tf.randn((1, 8)))
                self.assertEqual(res_active.shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
