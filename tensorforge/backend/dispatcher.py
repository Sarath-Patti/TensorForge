"""Backend dispatch, execution, and CPU thread management for TensorForge."""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Optional

from tensorforge.tensor.native_storage import is_native_available
from tensorforge.utils.validation import TensorForgeError

_ACTIVE_BACKEND: str = "numpy"
_LAST_BACKEND: str = "numpy"
_NUM_THREADS: int = 4


def get_backend() -> str:
    """Return the currently active compute backend ('numpy' or 'native')."""
    return _ACTIVE_BACKEND


def set_backend(name: str) -> None:
    """Set the active compute backend for tensor operations.

    Args:
        name: Name of the backend ('numpy' or 'native').

    Raises:
        TensorForgeError: If an invalid backend name is given or if 'native' is requested
            when the C++ native extension is not available.
    """
    global _ACTIVE_BACKEND
    norm_name = name.strip().lower()
    if norm_name not in ("numpy", "native"):
        raise TensorForgeError(f"Invalid backend '{name}'. Supported backends: 'numpy', 'native'.")

    if norm_name == "native" and not is_native_available():
        raise TensorForgeError(
            "Native C++ backend is not available. Please compile native extensions using CMake."
        )

    _ACTIVE_BACKEND = norm_name


def get_last_backend() -> str:
    """Return the backend that executed the most recent dispatched operation."""
    return _LAST_BACKEND


def set_last_backend(name: str) -> None:
    """Record which backend executed the last operation (internal use)."""
    global _LAST_BACKEND
    _LAST_BACKEND = name


def set_num_threads(num_threads: int) -> None:
    """Set the number of CPU threads used for parallel native inference execution.

    Args:
        num_threads: Number of worker threads (must be >= 1).

    Raises:
        TensorForgeError: If num_threads < 1.
    """
    global _NUM_THREADS
    if not isinstance(num_threads, int) or num_threads < 1:
        raise TensorForgeError(f"num_threads must be a positive integer >= 1, got {num_threads}.")

    _NUM_THREADS = num_threads
    if is_native_available():
        try:
            import _tensorforge_native as _native
            _native.set_num_threads(num_threads)
        except (ImportError, AttributeError):
            try:
                from tensorforge import _tensorforge_native as _native
                _native.set_num_threads(num_threads)
            except (ImportError, AttributeError):
                pass


def get_num_threads() -> int:
    """Get the currently configured CPU thread count."""
    if is_native_available():
        try:
            import _tensorforge_native as _native
            return int(_native.get_num_threads())
        except (ImportError, AttributeError):
            try:
                from tensorforge import _tensorforge_native as _native
                return int(_native.get_num_threads())
            except (ImportError, AttributeError):
                pass
    return _NUM_THREADS


class backend_context(ContextDecorator):
    """Context manager for temporarily selecting a compute backend.

    Example:
        >>> with backend_context("native"):
        ...     c = a @ b
    """

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        self.prev_backend: Optional[str] = None

    def __enter__(self) -> str:
        self.prev_backend = get_backend()
        set_backend(self.backend_name)
        return self.backend_name

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.prev_backend is not None:
            set_backend(self.prev_backend)
