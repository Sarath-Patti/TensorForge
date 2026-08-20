"""Tests for server-level resource limit enforcement."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer, ServerConfig
from tensorforge.serialization import save_model
from tensorforge.utils.validation import ServerLimitError


class TestServerLimits(unittest.TestCase):

    def test_max_loaded_models_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_m1 = os.path.join(tmpdir, "m1.tfmodel")
            path_m2 = os.path.join(tmpdir, "m2.tfmodel")

            save_model(nn.Linear(4, 2), path_m1)
            save_model(nn.Linear(4, 2), path_m2)

            with InferenceServer(config=ServerConfig(max_loaded_models=1)) as server:
                server.load_model("m1", path_m1)
                with self.assertRaises(ServerLimitError):
                    server.load_model("m2", path_m2)


if __name__ == "__main__":
    unittest.main()
