"""Tests for scoped ProfileSession context manager and state restoration."""

import os
import tempfile
import unittest
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, ProfileSession
from tensorforge.serialization import save_model


class TestProfileSession(unittest.TestCase):

    def test_scoped_profile_session(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(2, 8))
            self.assertFalse(runtime.profiling_enabled)

            # Scoped session
            with runtime.profile_session(detailed=True) as session:
                self.assertTrue(runtime.profiling_enabled)
                out = runtime.predict(tf.randn((2, 8)))
                self.assertEqual(out.shape, (2, 4))

            # State restored after exit
            self.assertFalse(runtime.profiling_enabled)
            self.assertGreater(session.duration_ms, 0.0)
            self.assertGreater(len(session.events), 0)

            # Session summary is valid string
            summary = session.summary()
            self.assertIn("ProfileSession", summary)

            # Session report
            report = session.report()
            self.assertIsNotNone(report)
            self.assertEqual(len(report.events), len(session.events))

    def test_scoped_profile_session_restores_state_on_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path)
            self.assertFalse(runtime.profiling_enabled)

            with self.assertRaises(ValueError):
                with runtime.profile_session():
                    self.assertTrue(runtime.profiling_enabled)
                    raise ValueError("Intentional test error")

            # Profiling restored to disabled
            self.assertFalse(runtime.profiling_enabled)

    def test_nested_or_sequential_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.tfmodel")
            model = nn.Linear(4, 2)
            save_model(model, model_path)

            runtime = InferenceRuntime.load(model_path).compile(input_shape=(2, 4))

            with runtime.profile_session(detailed=True) as s1:
                _ = runtime.predict(tf.randn((2, 4)))
            count1 = len(s1.events)

            with runtime.profile_session(detailed=True) as s2:
                _ = runtime.predict(tf.randn((2, 4)))
            count2 = len(s2.events)

            self.assertEqual(count1, count2)
            self.assertEqual(len(s2.events), count2)


if __name__ == "__main__":
    unittest.main()
