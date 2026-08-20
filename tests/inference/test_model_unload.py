"""Tests for model unloading and resource release."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model
from tensorforge.utils.validation import ModelNotFoundError


class TestModelUnload(unittest.TestCase):

    def test_unload_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            save_model(nn.Linear(4, 2), model_path)

            with InferenceServer() as server:
                server.load_model("m1", model_path, version="1")
                self.assertTrue(server.registry.has_model("m1", "1"))

                res = server.unload_model("m1", version="1")
                self.assertTrue(res)
                self.assertFalse(server.registry.has_model("m1", "1"))

                with self.assertRaises(ModelNotFoundError):
                    server.predict("m1", tf.randn((1, 4)))


if __name__ == "__main__":
    unittest.main()
