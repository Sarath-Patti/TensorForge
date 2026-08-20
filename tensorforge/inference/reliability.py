"""Production Reliability, Circuit Breakers, Retry Policies, and Health State for TensorForge."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import threading
import time
from typing import Any, Dict, Optional, Tuple, Type, Union

from tensorforge.utils.validation import (
    CircuitBreakerOpenError,
    RequestCancelledError,
    RequestDeadlineExceededError,
    RetryLimitExceededError,
    TensorForgeInputError,
)


class HealthState(str, Enum):
    """Health classification states for a model version or server instance."""

    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"


class CircuitState(str, Enum):
    """Circuit breaker states for model execution isolation."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class RequestState(str, Enum):
    """Request lifecycle ownership states ensuring single terminal state transitions."""

    CREATED = "CREATED"
    ADMITTED = "ADMITTED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"
    FAILED = "FAILED"


@dataclass
class CircuitBreakerConfig:
    """Configuration options for lightweight per-model circuit breaker."""

    failure_threshold: int = 5
    recovery_timeout_ms: float = 5000.0
    half_open_max_requests: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_ms": self.recovery_timeout_ms,
            "half_open_max_requests": self.half_open_max_requests,
        }


class CircuitBreaker:
    """Thread-safe, lightweight circuit breaker state machine for per-model failure isolation.

    Protects downstream inference runtime execution by rejecting requests immediately
    when error thresholds are exceeded and probing for recovery after a monotonic delay.
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config: CircuitBreakerConfig = config if config is not None else CircuitBreakerConfig()
        self._lock: threading.RLock = threading.RLock()
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._last_opened_timestamp: float = 0.0
        self._half_open_active_requests: int = 0
        self._total_transitions: int = 0
        self._recovery_attempts: int = 0
        self._recovery_successes: int = 0
        self._recovery_failures: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        with self._lock:
            self._check_recovery_timeout()
            return self._state

    @property
    def config(self) -> CircuitBreakerConfig:
        """Circuit breaker configuration."""
        return self._config

    def _check_recovery_timeout(self) -> None:
        """Check if OPEN state recovery timeout has elapsed using monotonic time."""
        if self._state == CircuitState.OPEN:
            elapsed_ms = (time.monotonic() - self._last_opened_timestamp) * 1000.0
            if elapsed_ms >= self._config.recovery_timeout_ms:
                self._state = CircuitState.HALF_OPEN
                self._half_open_active_requests = 0
                self._total_transitions += 1
                self._recovery_attempts += 1

    def allow_request(self) -> bool:
        """Check if a request is permitted to proceed into execution."""
        with self._lock:
            self._check_recovery_timeout()
            if self._state == CircuitState.CLOSED:
                return True
            elif self._state == CircuitState.HALF_OPEN:
                if self._half_open_active_requests < self._config.half_open_max_requests:
                    self._half_open_active_requests += 1
                    return True
                return False
            else:  # CircuitState.OPEN
                return False

    def record_success(self) -> None:
        """Record successful inference execution."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_active_requests = 0
                self._total_transitions += 1
                self._recovery_successes += 1
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record execution failure (excluding client validation errors)."""
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                self._last_opened_timestamp = time.monotonic()
                self._half_open_active_requests = 0
                self._total_transitions += 1
                self._recovery_failures += 1
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self._config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._last_opened_timestamp = time.monotonic()
                    self._total_transitions += 1

    def reset(self) -> None:
        """Manually reset circuit breaker state to CLOSED."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_active_requests = 0

    def to_dict(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of circuit breaker metrics."""
        with self._lock:
            self._check_recovery_timeout()
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_transitions": self._total_transitions,
                "recovery_attempts": self._recovery_attempts,
                "recovery_successes": self._recovery_successes,
                "recovery_failures": self._recovery_failures,
                "config": self._config.to_dict(),
            }


@dataclass
class RetryConfig:
    """Explicit configuration for request-level retry policy and backoff."""

    max_retries: int = 0
    base_delay_ms: float = 50.0
    max_delay_ms: float = 1000.0
    backoff_factor: float = 2.0
    retryable_exceptions: Tuple[Type[BaseException], ...] = field(
        default_factory=lambda: (RuntimeError, Exception)
    )

    def is_retryable(self, exc: BaseException) -> bool:
        """Check if an exception is explicitly permitted to be retried."""
        if isinstance(exc, (TensorForgeInputError, RequestCancelledError, RequestDeadlineExceededError, CircuitBreakerOpenError)):
            return False
        return isinstance(exc, self.retryable_exceptions)

    def to_dict(self) -> Dict[str, Any]:
        """Convert retry configuration to dictionary."""
        return {
            "max_retries": self.max_retries,
            "base_delay_ms": self.base_delay_ms,
            "max_delay_ms": self.max_delay_ms,
            "backoff_factor": self.backoff_factor,
        }


def compute_backoff_delay_sec(attempt: int, config: RetryConfig) -> float:
    """Compute bounded exponential backoff delay in seconds using monotonic math.

    Formula: min(base_delay_ms * (backoff_factor ** attempt) / 1000, max_delay_ms / 1000)

    Args:
        attempt: Zero-indexed retry attempt number (0, 1, 2, ...).
        config: RetryConfig instance.

    Returns:
        Delay in seconds to sleep outside of locks.
    """
    if attempt < 0:
        attempt = 0
    raw_delay_ms = config.base_delay_ms * (config.backoff_factor ** attempt)
    capped_delay_ms = min(raw_delay_ms, config.max_delay_ms)
    return capped_delay_ms / 1000.0


@dataclass
class ReliabilityMetrics:
    """Structured metrics tracking v1.9 production reliability events."""

    timeouts: int = 0
    cancellations: int = 0
    rejections: int = 0
    circuit_open_rejections: int = 0
    circuit_transitions: int = 0
    retries: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    recovery_failures: int = 0
    resource_exhaustion_count: int = 0
    execution_failures: int = 0
    shutdown_cancellations: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)
