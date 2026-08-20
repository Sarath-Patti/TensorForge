"""Unit tests for Graceful Server Shutdown with Timeout in TensorForge v1.9 (Static Creation Only)."""

import unittest
from tensorforge.inference import InferenceServer, ServerConfig, ServerClosedError
import numpy as np


class TestGracefulShutdown(unittest.TestCase):
    """Test suite verifying server shutdown deadline and request rejection."""

    def test_shutdown_rejects_new_requests(self):
        """Verify closed/draining server rejects new predictions immediately."""
        server = InferenceServer(ServerConfig(auto_start=True))
        server.close(timeout_ms=100.0)

        with self.assertRaises(ServerClosedError):
            server.predict("nonexistent", np.ones((1, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
