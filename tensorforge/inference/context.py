"""Per-prediction Execution Context and Workspace Pool for thread-safe concurrent inference."""

from __future__ import annotations

import collections
from contextlib import contextmanager
import threading
import time
from typing import Any, Dict, Generator, List, Optional
import numpy as np

from tensorforge.tensor.tensor import Tensor
from tensorforge.utils.validation import ExecutionContextError


class ExecutionContext:
    """An isolated, per-prediction execution workspace and buffer manager.

    Ensures that concurrent threads never share mutable intermediate activation slots
    or temporary workspace memory during inference execution.

    Args:
        context_id: Unique numerical identifier for this execution context.
    """

    def __init__(self, context_id: int) -> None:
        self.context_id: int = context_id
        self.slots: Dict[int, Tensor] = {}
        self.is_in_use: bool = False
        self.created_at: float = time.time()
        self.last_used_at: float = self.created_at
        self.prediction_count: int = 0
        self.arena: Optional[Any] = None

    def reset(self) -> None:
        """Reset the execution context state, clearing all intermediate buffers."""
        self.slots.clear()
        if self.arena is not None and hasattr(self.arena, "reset"):
            self.arena.reset()
        self.is_in_use = False

    def get_slot(self, slot_id: int) -> Optional[Tensor]:
        """Fetch an intermediate tensor stored in the specified slot."""
        return self.slots.get(slot_id)

    def set_slot(self, slot_id: int, tensor: Tensor) -> None:
        """Store an intermediate tensor into the specified slot."""
        self.slots[slot_id] = tensor

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(id={self.context_id}, in_use={self.is_in_use}, "
            f"slots={len(self.slots)}, preds={self.prediction_count})"
        )


class ExecutionContextPool:
    """Thread-safe object pool managing reusable ExecutionContext instances.

    Uses a fast, synchronized LIFO queue to reuse idle execution contexts across
    concurrent inference worker threads, avoiding memory allocation churn.

    Args:
        max_pool_size: Maximum number of idle contexts to retain in the pool.
    """

    def __init__(self, max_pool_size: int = 128) -> None:
        self._max_pool_size: int = max_pool_size
        self._available_contexts: collections.deque[ExecutionContext] = collections.deque()
        self._all_contexts: List[ExecutionContext] = []
        self._lock: threading.Lock = threading.Lock()
        self._context_counter: int = 0
        self._active_count: int = 0

    def get_context(self) -> ExecutionContext:
        """Acquire an idle ExecutionContext from the pool, or allocate a new one.

        Returns:
            An isolated, clean ExecutionContext instance marked as in-use.
        """
        with self._lock:
            if self._available_contexts:
                ctx = self._available_contexts.pop()
            else:
                self._context_counter += 1
                ctx = ExecutionContext(context_id=self._context_counter)
                self._all_contexts.append(ctx)

            ctx.is_in_use = True
            ctx.last_used_at = time.time()
            ctx.prediction_count += 1
            self._active_count += 1
            return ctx

    def release_context(self, ctx: ExecutionContext) -> None:
        """Return a previously acquired ExecutionContext back to the pool.

        Args:
            ctx: The ExecutionContext to release.
        """
        ctx.reset()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            if len(self._available_contexts) < self._max_pool_size:
                self._available_contexts.append(ctx)

    @contextmanager
    def acquire(self) -> Generator[ExecutionContext, None, None]:
        """Context-manager wrapper for RAII acquisition and release of an ExecutionContext.

        Yields:
            Clean ExecutionContext instance.
        """
        ctx = self.get_context()
        try:
            yield ctx
        finally:
            self.release_context(ctx)

    def clear(self) -> None:
        """Release and clear all managed execution contexts."""
        with self._lock:
            for ctx in self._all_contexts:
                ctx.reset()
                ctx.arena = None
            self._available_contexts.clear()
            self._all_contexts.clear()
            self._active_count = 0

    @property
    def active_count(self) -> int:
        """Number of execution contexts currently checked out and in active use."""
        with self._lock:
            return self._active_count

    @property
    def total_count(self) -> int:
        """Total number of execution contexts created in this pool."""
        with self._lock:
            return len(self._all_contexts)

    @property
    def idle_count(self) -> int:
        """Number of available idle execution contexts ready for reuse."""
        with self._lock:
            return len(self._available_contexts)

    def __repr__(self) -> str:
        return (
            f"ExecutionContextPool(active={self.active_count}, idle={self.idle_count}, "
            f"total={self.total_count})"
        )
