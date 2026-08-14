"""Lightweight profiling and timing utilities for TensorForge."""

from __future__ import annotations

import time
from typing import Optional


class profile:
    """Context manager for timing execution of code blocks and kernel operations.

    Example:
        >>> with profile("matmul"):
        ...     c = a @ b
        [matmul] Elapsed: 12.345 ms (0.012345 s)
    """

    def __init__(self, name: str = "operation", verbose: bool = True) -> None:
        self.name: str = name
        self.verbose: bool = verbose
        self.elapsed_sec: float = 0.0
        self.elapsed_ms: float = 0.0
        self._start_time: float = 0.0

    def __enter__(self) -> profile:
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        end_time = time.perf_counter()
        self.elapsed_sec = end_time - self._start_time
        self.elapsed_ms = self.elapsed_sec * 1000.0
        if self.verbose:
            print(f"[{self.name}] Elapsed: {self.elapsed_ms:.3f} ms ({self.elapsed_sec:.6f} s)")
