"""Unit tests for DataLoader in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, int64, tensor
from tensorforge.data import DataLoader, TensorDataset


class TestDataLoader(unittest.TestCase):
    """Tests for DataLoader batching, shuffling determinism, and drop_last behavior."""

    def setUp(self):
        # 10 samples with 3 features and integer labels
        self.x = tensor(np.arange(30, dtype=np.float32).reshape(10, 3))
        self.y = tensor(np.arange(10, dtype=np.int64))
        self.dataset = TensorDataset(self.x, self.y)

    def test_sequential_batching_drop_last_false(self):
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False, drop_last=False)
        self.assertEqual(len(loader), 3)  # 4 + 4 + 2

        batches = list(loader)
        self.assertEqual(len(batches), 3)

        # Batch 0: shape (4, 3) and (4,)
        b0_x, b0_y = batches[0]
        self.assertEqual(b0_x.shape, (4, 3))
        self.assertEqual(b0_y.shape, (4,))
        self.assertEqual(b0_y.dtype, int64)
        np.testing.assert_allclose(b0_y.numpy(), [0, 1, 2, 3])

        # Batch 2 (last incomplete): shape (2, 3) and (2,)
        b2_x, b2_y = batches[2]
        self.assertEqual(b2_x.shape, (2, 3))
        self.assertEqual(b2_y.shape, (2,))
        np.testing.assert_allclose(b2_y.numpy(), [8, 9])

    def test_sequential_batching_drop_last_true(self):
        loader = DataLoader(self.dataset, batch_size=4, shuffle=False, drop_last=True)
        self.assertEqual(len(loader), 2)  # drops last 2 samples

        batches = list(loader)
        self.assertEqual(len(batches), 2)
        for b_x, b_y in batches:
            self.assertEqual(b_x.shape, (4, 3))
            self.assertEqual(b_y.shape, (4,))

    def test_shuffling_determinism_with_seed(self):
        loader1 = DataLoader(self.dataset, batch_size=5, shuffle=True, seed=42)
        loader2 = DataLoader(self.dataset, batch_size=5, shuffle=True, seed=42)

        batches1 = list(loader1)
        batches2 = list(loader2)

        for (b1_x, b1_y), (b2_x, b2_y) in zip(batches1, batches2):
            np.testing.assert_allclose(b1_x.numpy(), b2_x.numpy())
            np.testing.assert_allclose(b1_y.numpy(), b2_y.numpy())


if __name__ == "__main__":
    unittest.main()
