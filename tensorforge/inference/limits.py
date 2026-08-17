"""Runtime limits, lifecycle states, and resource constraints for production inference."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional


class RuntimeState(str, Enum):
    """Lifecycle states for the TensorForge Inference Runtime."""

    CREATED = "CREATED"
    READY = "READY"
    CLOSED = "CLOSED"


class RuntimeLimits:
    """Configurable resource and admission limits for production inference workloads.

    Provides admission control safeguards against oversized inputs, excessive workspace memory,
    concurrency overload, and runaway prediction latencies.

    Args:
        max_batch_size: Maximum allowed batch size per prediction request.
        max_input_elements: Maximum total number of elements in the input tensor.
        max_workspace_bytes: Maximum planned or allocated workspace memory in bytes.
        max_prediction_time_ms: Optional soft timeout threshold in milliseconds for prediction duration.
        max_concurrent_requests: Maximum number of simultaneous active prediction requests.

    Raises:
        ValueError: If any provided limit value is non-positive or of an invalid type.
    """

    def __init__(
        self,
        max_batch_size: Optional[int] = None,
        max_input_elements: Optional[int] = None,
        max_workspace_bytes: Optional[int] = None,
        max_prediction_time_ms: Optional[float] = None,
        max_concurrent_requests: Optional[int] = None,
    ) -> None:
        if max_batch_size is not None:
            if not isinstance(max_batch_size, int) or max_batch_size < 1:
                raise ValueError(f"max_batch_size must be a positive integer, got {max_batch_size}")

        if max_input_elements is not None:
            if not isinstance(max_input_elements, int) or max_input_elements < 1:
                raise ValueError(f"max_input_elements must be a positive integer, got {max_input_elements}")

        if max_workspace_bytes is not None:
            if not isinstance(max_workspace_bytes, int) or max_workspace_bytes < 0:
                raise ValueError(f"max_workspace_bytes must be a non-negative integer, got {max_workspace_bytes}")

        if max_prediction_time_ms is not None:
            if not isinstance(max_prediction_time_ms, (int, float)) or max_prediction_time_ms <= 0:
                raise ValueError(f"max_prediction_time_ms must be a positive number, got {max_prediction_time_ms}")

        if max_concurrent_requests is not None:
            if not isinstance(max_concurrent_requests, int) or max_concurrent_requests < 1:
                raise ValueError(f"max_concurrent_requests must be a positive integer, got {max_concurrent_requests}")

        self.max_batch_size: Optional[int] = max_batch_size
        self.max_input_elements: Optional[int] = max_input_elements
        self.max_workspace_bytes: Optional[int] = max_workspace_bytes
        self.max_prediction_time_ms: Optional[float] = float(max_prediction_time_ms) if max_prediction_time_ms is not None else None
        self.max_concurrent_requests: Optional[int] = max_concurrent_requests

    def to_dict(self) -> Dict[str, Any]:
        """Convert limits configuration to a dictionary representation."""
        return {
            "max_batch_size": self.max_batch_size,
            "max_input_elements": self.max_input_elements,
            "max_workspace_bytes": self.max_workspace_bytes,
            "max_prediction_time_ms": self.max_prediction_time_ms,
            "max_concurrent_requests": self.max_concurrent_requests,
        }

    def copy(self) -> RuntimeLimits:
        """Create a deep copy of this RuntimeLimits instance."""
        return RuntimeLimits(
            max_batch_size=self.max_batch_size,
            max_input_elements=self.max_input_elements,
            max_workspace_bytes=self.max_workspace_bytes,
            max_prediction_time_ms=self.max_prediction_time_ms,
            max_concurrent_requests=self.max_concurrent_requests,
        )

    def __repr__(self) -> str:
        return (
            f"RuntimeLimits(max_batch_size={self.max_batch_size}, "
            f"max_input_elements={self.max_input_elements}, "
            f"max_workspace_bytes={self.max_workspace_bytes}, "
            f"max_prediction_time_ms={self.max_prediction_time_ms}, "
            f"max_concurrent_requests={self.max_concurrent_requests})"
        )
