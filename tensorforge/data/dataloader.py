"""DataLoader abstraction for mini-batch generation in TensorForge."""

from __future__ import annotations

from typing import Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np

from tensorforge.data.dataset import Dataset
from tensorforge.tensor.tensor import Tensor


class DataLoader:
    """Combines a dataset and a sampler, providing an iterable over the given dataset.

    Supports mini-batching, deterministic shuffling, and optional dropping of incomplete batches.

    Args:
        dataset: Dataset from which to load the data.
        batch_size: Number of samples per batch to load (default: 1).
        shuffle: Set to True to have the data reshuffled at every epoch (default: False).
        drop_last: Set to True to drop the last incomplete batch if dataset size is not divisible by batch_size.
        seed: Optional random seed for reproducible shuffling.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.dataset: Dataset = dataset
        self.batch_size: int = int(batch_size)
        self.shuffle: bool = bool(shuffle)
        self.drop_last: bool = bool(drop_last)
        self.seed: Optional[int] = seed
        self._rng: Optional[np.random.RandomState] = (
            np.random.RandomState(seed) if seed is not None else None
        )

    def __len__(self) -> int:
        """Return total number of batches in the DataLoader."""
        num_samples = len(self.dataset)
        if self.drop_last:
            return num_samples // self.batch_size
        return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Union[Tensor, Tuple[Tensor, ...]]]:
        """Yield batched tensors for each mini-batch."""
        num_samples = len(self.dataset)
        indices = np.arange(num_samples)

        if self.shuffle:
            if self._rng is not None:
                self._rng.shuffle(indices)
            else:
                np.random.shuffle(indices)

        for i in range(0, num_samples, self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue

            # Fetch samples
            samples = [self.dataset[int(idx)] for idx in batch_indices]

            # Collate samples into batch tensors
            if isinstance(samples[0], tuple):
                num_fields = len(samples[0])
                batch_fields: List[Tensor] = []
                for f_idx in range(num_fields):
                    field_samples = [s[f_idx] for s in samples]
                    sample_arrays = [s.numpy() for s in field_samples]
                    stacked_arr = np.stack(sample_arrays, axis=0)
                    field_dtype = field_samples[0].dtype
                    batch_fields.append(
                        Tensor(stacked_arr, dtype=field_dtype, copy=False)
                    )
                yield tuple(batch_fields)
            else:
                sample_arrays = [s.numpy() for s in samples]
                stacked_arr = np.stack(sample_arrays, axis=0)
                field_dtype = samples[0].dtype
                yield Tensor(stacked_arr, dtype=field_dtype, copy=False)
