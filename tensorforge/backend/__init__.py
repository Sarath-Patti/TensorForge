"""Compute backend management and dispatch package for TensorForge."""

from tensorforge.backend.dispatcher import (
    backend_context,
    get_backend,
    get_last_backend,
    is_native_available,
    set_backend,
    set_last_backend,
)

__all__ = [
    "set_backend",
    "get_backend",
    "get_last_backend",
    "set_last_backend",
    "backend_context",
    "is_native_available",
]
