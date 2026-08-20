"""Tests for ModelRegistry registration, active version resolution, and duplicates handling."""

import unittest
from tensorforge.inference.server import ModelEntry, ModelLifecycleState, ModelRegistry
from tensorforge.utils.validation import ModelAlreadyLoadedError, ModelNotFoundError, ModelVersionNotFoundError


class TestModelRegistry(unittest.TestCase):

    def test_registry_register_and_get(self):
        registry = ModelRegistry()
        entry1 = ModelEntry(name="detector", version="1.0", path="m1.tfmodel", state=ModelLifecycleState.READY)
        registry.register(entry1, set_active=True)

        self.assertTrue(registry.has_model("detector"))
        self.assertTrue(registry.has_model("detector", "1.0"))
        self.assertEqual(registry.get_active_version("detector"), "1.0")

        fetched = registry.get("detector")
        self.assertEqual(fetched.version, "1.0")

    def test_duplicate_registration_raises(self):
        registry = ModelRegistry()
        entry1 = ModelEntry(name="detector", version="1.0", path="m1.tfmodel")
        registry.register(entry1)

        entry2 = ModelEntry(name="detector", version="1.0", path="m2.tfmodel")
        with self.assertRaises(ModelAlreadyLoadedError):
            registry.register(entry2, overwrite=False)

    def test_unregistered_model_raises(self):
        registry = ModelRegistry()
        with self.assertRaises(ModelNotFoundError):
            registry.get("unknown_model")

        with self.assertRaises(ModelNotFoundError):
            registry.get_active_version("unknown_model")


if __name__ == "__main__":
    unittest.main()
