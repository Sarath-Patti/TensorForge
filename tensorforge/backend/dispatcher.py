"""Backend dispatch and execution management for TensorForge."""

from __future__ import annotations

from contextlib import ContextDecorator
from typing import Optional

from tensorforge.tensor.native_storage import is_native_available
from tensorforge.utils.validation import TensorForgeError

_ACTIVE_BACKEND: str = "numpy"
_LAST_BACKEND: str = "numpy"


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


class backend_context(ContextDecorator):
    """Context manager for temporarily selecting a compute backend.

    Example:
        >>> with backend_context("native"):
        ...     c = a @ b
    """

    def __init__(self, backend_name: str) -> None:
        self.target_backend = backend_name
        self.prev_backend = get_backend()

    def __enter__(self) -> backend_context:
        self.prev_backend = get_backend()
        set_backend(self.target_backend)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        set_backend(self.prev_backend)
