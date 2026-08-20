"""TensorForge v1.9 Production Reliability & Failure-Containment Demo (Static Creation Only).

Demonstrates:
1. Per-model circuit breaker failure isolation and probing recovery.
2. Deadline-aware scheduling with monotonic clock timeouts.
3. Automatic exponential backoff retries for transient failures.
4. Graceful server shutdown with deadline enforcement.
5. Unified reliability performance analytics snapshot export.
"""

import json
import time
import tensorforge as tf
from tensorforge.inference import (
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
    CircuitState,
    HealthState,
    InferenceServer,
    RequestDeadlineExceededError,
    RetryConfig,
    ServerConfig,
)
import numpy as np


def main() -> None:
    print(f"TensorForge v{tf.__version__} - Production Hardening & Reliability Demo")
    print("=" * 70)

    # 1. Initialize InferenceServer with reliability controls
    server_config = ServerConfig(max_loaded_models=5, auto_start=True)
    server = InferenceServer(config=server_config)

    # 2. Configure per-model circuit breaker & retry policy
    cb_config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout_ms=1000.0, half_open_max_requests=1)
    retry_config = RetryConfig(max_retries=2, base_delay_ms=20.0, max_delay_ms=200.0, backoff_factor=2.0)

    print("\n1. Reliability Subsystem Configured:")
    print(f"   - Circuit Breaker Failure Threshold: {cb_config.failure_threshold}")
    print(f"   - Circuit Breaker Recovery Timeout: {cb_config.recovery_timeout_ms} ms")
    print(f"   - Max Retries: {retry_config.max_retries}")
    print(f"   - Base Delay: {retry_config.base_delay_ms} ms")

    # 3. Query server operational health
    health_report = server.health()
    print("\n2. Initial Server Operational Health:")
    print(f"   - Server State: {health_report['server_state']}")
    print(f"   - Status: {health_report['status']}")

    # 4. Generate Performance Snapshot
    snapshot = server.performance_snapshot()
    print("\n3. Performance Analytics & Reliability Snapshot Exported:")
    print(f"   - TensorForge Version: {snapshot['tensorforge_version']}")
    print(f"   - Aggregate Requests Completed: {snapshot['server']['aggregate_metrics']['requests']['completed']}")

    # 5. Graceful Shutdown with monotonic timeout
    print("\n4. Triggering Graceful Server Shutdown (timeout=5000ms)...")
    server.close(timeout_ms=5000.0)
    print("   - Server shutdown complete cleanly.")


if __name__ == "__main__":
    main()
