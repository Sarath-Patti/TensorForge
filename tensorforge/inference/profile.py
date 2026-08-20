"""Runtime Workload Profiles & Configuration Presets for TensorForge v2.0."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Union

from tensorforge.inference.limits import RuntimeLimits
from tensorforge.inference.reliability import CircuitBreakerConfig, RetryConfig
from tensorforge.inference.scheduler import SchedulerConfig, SchedulingPolicy
from tensorforge.utils.validation import TensorForgeInputError


class RuntimeProfileType(str, Enum):
    """Pre-packaged configuration profiles for different inference workloads."""

    LOW_LATENCY = "LOW_LATENCY"
    HIGH_THROUGHPUT = "HIGH_THROUGHPUT"
    BALANCED = "BALANCED"
    EMBEDDED = "EMBEDDED"


@dataclass
class RuntimeProfile:
    """Cohesive configuration bundle combining scheduler, limits, circuit breaker, and retries."""

    profile_type: RuntimeProfileType
    description: str
    scheduler_config: SchedulerConfig
    runtime_limits: RuntimeLimits
    circuit_breaker_config: CircuitBreakerConfig
    retry_config: RetryConfig

    def to_dict(self) -> Dict[str, Any]:
        """Return structured dictionary representation of the profile."""
        return {
            "profile_type": self.profile_type.value,
            "description": self.description,
            "scheduler_config": self.scheduler_config.to_dict(),
            "runtime_limits": self.runtime_limits.to_dict(),
            "circuit_breaker_config": self.circuit_breaker_config.to_dict(),
            "retry_config": self.retry_config.to_dict(),
        }


# Public alias for v2.0 specification consistency
RuntimeConfig = RuntimeProfile


def get_runtime_profile(profile_type: Union[str, RuntimeProfileType]) -> RuntimeProfile:
    """Retrieve pre-packaged RuntimeProfile preset by type name or enum.

    Args:
        profile_type: Profile type string ('LOW_LATENCY', 'HIGH_THROUGHPUT', 'BALANCED', 'EMBEDDED') or Enum.

    Returns:
        RuntimeProfile preset instance.

    Raises:
        TensorForgeInputError: If profile_type is invalid or unrecognized.
    """
    if isinstance(profile_type, str):
        try:
            pt = RuntimeProfileType[profile_type.upper()]
        except KeyError:
            try:
                pt = RuntimeProfileType(profile_type.upper())
            except ValueError:
                valid_types = [p.value for p in RuntimeProfileType]
                raise TensorForgeInputError(
                    f"Unknown RuntimeProfileType '{profile_type}'. Valid profiles: {valid_types}"
                )
    elif isinstance(profile_type, RuntimeProfileType):
        pt = profile_type
    else:
        raise TensorForgeInputError(f"Invalid profile_type type: {type(profile_type)}")

    if pt == RuntimeProfileType.LOW_LATENCY:
        return RuntimeProfile(
            profile_type=RuntimeProfileType.LOW_LATENCY,
            description="Optimized for ultra-low latency single-sample requests with minimal queueing.",
            scheduler_config=SchedulerConfig(
                max_batch_size=1,
                batch_timeout_ms=0.5,
                max_queue_size=100,
                policy=SchedulingPolicy.FIFO,
            ),
            runtime_limits=RuntimeLimits(max_batch_size=1, max_input_elements=1_000_000),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_ms=2000.0,
                half_open_max_requests=1,
            ),
            retry_config=RetryConfig(
                max_retries=1,
                base_delay_ms=10.0,
                max_delay_ms=100.0,
                backoff_factor=2.0,
            ),
        )

    elif pt == RuntimeProfileType.HIGH_THROUGHPUT:
        return RuntimeProfile(
            profile_type=RuntimeProfileType.HIGH_THROUGHPUT,
            description="Optimized for maximum inference throughput via large dynamic batching.",
            scheduler_config=SchedulerConfig(
                max_batch_size=64,
                batch_timeout_ms=50.0,
                max_queue_size=5000,
                policy=SchedulingPolicy.FIFO,
            ),
            runtime_limits=RuntimeLimits(max_batch_size=64, max_input_elements=100_000_000),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=10,
                recovery_timeout_ms=10000.0,
                half_open_max_requests=2,
            ),
            retry_config=RetryConfig(
                max_retries=3,
                base_delay_ms=50.0,
                max_delay_ms=1000.0,
                backoff_factor=2.0,
            ),
        )

    elif pt == RuntimeProfileType.EMBEDDED:
        return RuntimeProfile(
            profile_type=RuntimeProfileType.EMBEDDED,
            description="Optimized for resource-constrained edge/embedded environments with bounded memory.",
            scheduler_config=SchedulerConfig(
                max_batch_size=4,
                batch_timeout_ms=2.0,
                max_queue_size=50,
                policy=SchedulingPolicy.FIFO,
            ),
            runtime_limits=RuntimeLimits(max_batch_size=4, max_input_elements=500_000),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout_ms=3000.0,
                half_open_max_requests=1,
            ),
            retry_config=RetryConfig(
                max_retries=1,
                base_delay_ms=15.0,
                max_delay_ms=150.0,
                backoff_factor=2.0,
            ),
        )

    else:  # BALANCED
        return RuntimeProfile(
            profile_type=RuntimeProfileType.BALANCED,
            description="General-purpose production profile balancing latency, throughput, and reliability.",
            scheduler_config=SchedulerConfig(
                max_batch_size=16,
                batch_timeout_ms=5.0,
                max_queue_size=1000,
                policy=SchedulingPolicy.FIFO,
            ),
            runtime_limits=RuntimeLimits(max_batch_size=16, max_input_elements=10_000_000),
            circuit_breaker_config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout_ms=5000.0,
                half_open_max_requests=1,
            ),
            retry_config=RetryConfig(
                max_retries=2,
                base_delay_ms=20.0,
                max_delay_ms=500.0,
                backoff_factor=2.0,
            ),
        )
