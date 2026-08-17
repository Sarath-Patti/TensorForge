"""Tests for compiler cache observability, hits/misses tracking, and compilation time measurements."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


class TestCompilerStats(unittest.TestCase):

    def test_compiler_cache_hit_and_miss_tracking(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            runtime.enable_profiling()

            # First compile: cache miss
            runtime.compile(input_shape=(4, 8), use_cache=True)
            stats1 = runtime.compiler_stats()
            self.assertEqual(stats1["cache_misses"], 1)
            self.assertEqual(stats1["cache_hits"], 0)
            self.assertGreater(stats1["total_compilation_time_ms"], 0.0)

            # Recompile identical shape with cache: cache hit
            runtime.compile(input_shape=(4, 8), use_cache=True)
            stats2 = runtime.compiler_stats()
            self.assertEqual(stats2["cache_misses"], 1)
            self.assertEqual(stats2["cache_hits"], 1)
            self.assertAlmostEqual(stats2["cache_hit_rate"], 0.5, places=2)

            report = runtime.profile()
            comp_summary = report.compiler_summary()
            self.assertIn("Cache Hits:", comp_summary)
            self.assertIn("Cache Misses:", comp_summary)


if __name__ == "__main__":
    unittest.main()
