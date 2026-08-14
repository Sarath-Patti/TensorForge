"""Evaluation and training metrics for TensorForge."""

from __future__ import annotations

from typing import Sequence, Union
import numpy as np

from tensorforge.tensor.tensor import Tensor


def accuracy(
    predictions: Tensor,
    targets: Union[Tensor, Sequence[int], np.ndarray],
) -> float:
    """Compute classification accuracy between model predictions (logits or class indices) and targets.

    Args:
        predictions: Unnormalized class logits of shape (N, C) or predicted labels of shape (N,).
        targets: True integer class labels of shape (N,).

    Returns:
        Accuracy score in range [0.0, 1.0].
    """
    if isinstance(targets, Tensor):
        targets_np = targets.numpy()
    elif isinstance(targets, np.ndarray):
        targets_np = targets
    else:
        targets_np = np.array(targets)

    if predictions.ndim > 1:
        pred_labels = predictions.argmax(axis=-1).numpy()
    else:
        pred_labels = predictions.numpy()

    return float(np.mean(pred_labels == targets_np))
