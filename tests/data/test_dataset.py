"""Unit tests for Dataset and TensorDataset in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, int64, tensor
from tensorforge.data import Dataset, TensorDataset
from tensorforge.utils.validation import DimensionError


class TestDataset(unittest.TestCase):
    """Tests for Dataset base class and TensorDataset implementation."""

    def test_tensor_dataset_construction_and_indexing(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=float32)
        y = tensor([0, 1, 0], dtype=int64)

        dataset = TensorDataset(x, y)
        self.assertEqual(len(dataset), 3)

        # Index 0
        sample_x, sample_y = dataset[0]
        self.assertEqual(sample_x.shape, (2,))
        self.assertEqual(sample_y.shape, ())
        np.testing.assert_allclose(sample_x.numpy(), [1.0, 2.0])
        self.assertEqual(sample_y.item(), 0)

        # Index 2
        sample_x2, sample_y2 = dataset[2]
        np.testing.assert_allclose(sample_x2.numpy(), [5.0, 6.0])
        self.assertEqual(sample_y2.item(), 0)

    def test_tensor_dataset_length_mismatch(self):
        x = tensor([[1.0, 2.0], [3.0, 4.0]], dtype=float32)  # length 2
        y = tensor([0, 1, 0], dtype=int64)                   # length 3

        with self.assertRaises(DimensionError):
            TensorDataset(x, y)


if __name__ == "__main__":
    unittest.main()
