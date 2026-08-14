"""Unit tests for training and evaluation metrics in TensorForge."""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import float32, int64, tensor
from tensorforge.training import accuracy


class TestMetrics(unittest.TestCase):
    """Tests for classification accuracy metric."""

    def test_accuracy_with_logits(self):
        # 4 samples, 3 classes
        logits = tensor(
            [
                [10.0, 1.0, 0.0],   # pred: 0, target: 0 (correct)
                [0.0, 5.0, 1.0],    # pred: 1, target: 1 (correct)
                [2.0, 8.0, 3.0],    # pred: 1, target: 2 (incorrect)
                [0.1, 0.2, 4.0],    # pred: 2, target: 2 (correct)
            ],
            dtype=float32,
        )
        targets = tensor([0, 1, 2, 2], dtype=int64)

        # 3 out of 4 correct -> 0.75
        acc = accuracy(logits, targets)
        self.assertAlmostEqual(acc, 0.75)

    def test_accuracy_with_labels(self):
        preds = tensor([0, 1, 2, 0], dtype=int64)
        targets = [0, 1, 1, 0]  # 3 out of 4 match -> 0.75
        acc = accuracy(preds, targets)
        self.assertAlmostEqual(acc, 0.75)


if __name__ == "__main__":
    unittest.main()
