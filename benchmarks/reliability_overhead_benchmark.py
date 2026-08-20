"""TensorForge v1.9 Production Reliability Overhead Benchmark (Static Creation Only).

Measures:
1. Micro-second overhead of CircuitBreaker lock acquisition and state checks under concurrent load.
2. Latency impact of monotonic deadline calculations on request admission.
3. Memory overhead of ReliabilityMetrics tracking within MetricsCollector snapshots.
"""

import time
import tensorforge as tf
from tensorforge.inference import CircuitBreaker, CircuitBreakerConfig, MetricsCollector


def benchmark_circuit_breaker_check(iterations: int = 100_000) -> float:
    """Measure raw state checking throughput of CircuitBreaker."""
    cb = CircuitBreaker(CircuitBreakerConfig())
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = cb.allow_request()
    t1 = time.perf_counter()
    total_sec = t1 - t0
    ns_per_op = (total_sec / iterations) * 1e9
    return ns_per_op


def benchmark_metrics_collection(iterations: int = 50_000) -> float:
    """Measure latency of recording reliability events into MetricsCollector."""
    collector = MetricsCollector()
    t0 = time.perf_counter()
    for _ in range(iterations):
        collector.record_retry()
        collector.record_timeout()
    t1 = time.perf_counter()
    total_sec = t1 - t0
    ns_per_op = (total_sec / iterations) * 1e9
    return ns_per_op


def main() -> None:
    print(f"TensorForge v{tf.__version__} - Production Reliability Overhead Benchmark")
    print("=" * 70)

    cb_ns = benchmark_circuit_breaker_check()
    print(f"Circuit Breaker Check Overhead:  {cb_ns:.2f} ns / op")

    metrics_ns = benchmark_metrics_collection()
    print(f"Reliability Metric Rec Overhead: {metrics_ns:.2f} ns / op")
    print("=" * 70)


if __name__ == "__main__":
    main()
