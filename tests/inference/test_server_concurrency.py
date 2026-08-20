"""Tests verifying thread-safety of InferenceServer across concurrent threads."""

import concurrent.futures
import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceServer
from tensorforge.serialization import save_model


class TestServerConcurrency(unittest.TestCase):

    def test_concurrent_predictions_multi_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path_m1 = os.path.join(tmpdir, "m1.tfmodel")
            path_m2 = os.path.join(tmpdir, "m2.tfmodel")

            save_model(nn.Linear(8, 4), path_m1)
            save_model(nn.Linear(8, 2), path_m2)

            with InferenceServer() as server:
                server.load_model("m1", path_m1)
                server.load_model("m2", path_m2)

                def worker(model_name: str, expected_out_dim: int):
                    for _ in range(10):
                        x = tf.randn((2, 8))
                        out = server.predict(model_name, x)
                        assert out.shape == (2, expected_out_dim)

                with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                    futures = [
                        executor.submit(worker, "m1", 4),
                        executor.submit(worker, "m2", 2),
                        executor.submit(worker, "m1", 4),
                        executor.submit(worker, "m2", 2),
                    ]
                    _ = [f.result() for f in futures]


if __name__ == "__main__":
    unittest.main()
