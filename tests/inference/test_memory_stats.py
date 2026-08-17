"""Tests for memory telemetry, workspace metrics, region allocations, and context pooling statistics."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestMemoryStats(unittest.TestCase):

    def test_runtime_memory_telemetry(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(
                nn.Linear(16, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.Tanh(),
                nn.Linear(64, 4),
            )
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(8, 16))
            runtime.enable_profiling()

            report = runtime.profile()
            mem_stats = report.memory_stats

            self.assertGreater(mem_stats["workspace_bytes"], 0)
            self.assertGreater(mem_stats["num_regions"], 0)
            self.assertGreater(mem_stats["reused_buffers"], 0)
            self.assertEqual(mem_stats["active_contexts"], 0)

            mem_summary = report.memory_summary()
            self.assertIn("Workspace Memory:", mem_summary)
            self.assertIn("Memory Regions:", mem_summary)
            self.assertIn("Reused Buffers:", mem_summary)


if __name__ == "__main__":
    unittest.main()
