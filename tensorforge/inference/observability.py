"""Production Inference Observability & Performance Analytics Subsystem for TensorForge."""

from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field
import json
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from tensorforge.serialization.format import LIBRARY_VERSION


class LatencyHistogram:
    """Bounded reservoir for low-overhead latency distribution and percentile estimation.

    Maintains a fixed capacity ring reservoir ensuring O(1) memory usage and predictable
    overhead during high-throughput inference monitoring.

    Args:
        capacity: Maximum number of latency samples retained (default: 2048).
    """

    def __init__(self, capacity: int = 2048) -> None:
        if capacity < 10:
            capacity = 10
        self._capacity: int = capacity
        self._samples: List[float] = []
        self._index: int = 0
        self._is_full: bool = False
        self._total_count: int = 0
        self._sum_ms: float = 0.0
        self._min_ms: float = float("inf")
        self._max_ms: float = 0.0
        self._lock: threading.RLock = threading.RLock()

    def record(self, latency_ms: float) -> None:
        """Record a single latency sample in milliseconds.

        Args:
            latency_ms: Latency observation in milliseconds.
        """
        if latency_ms < 0.0:
            latency_ms = 0.0

        with self._lock:
            self._total_count += 1
            self._sum_ms += latency_ms
            if latency_ms < self._min_ms:
                self._min_ms = latency_ms
            if latency_ms > self._max_ms:
                self._max_ms = latency_ms

            if len(self._samples) < self._capacity:
                self._samples.append(latency_ms)
            else:
                self._samples[self._index] = latency_ms
                self._index = (self._index + 1) % self._capacity
                self._is_full = True

    def stats(self) -> LatencyStats:
        """Compute latency distribution statistics and percentiles.

        Returns:
            LatencyStats dataclass with min, max, mean, p50, p90, p95, p99.
        """
        with self._lock:
            count = self._total_count
            if count == 0 or len(self._samples) == 0:
                return LatencyStats(
                    min_ms=0.0,
                    max_ms=0.0,
                    mean_ms=0.0,
                    p50_ms=0.0,
                    p90_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    sample_count=0,
                )

            mean_val = self._sum_ms / count
            sorted_samples = sorted(self._samples)
            n = len(sorted_samples)

            def get_percentile(p: float) -> float:
                if n == 1:
                    return sorted_samples[0]
                idx = (p / 100.0) * (n - 1)
                low = int(math.floor(idx))
                high = int(math.ceil(idx))
                if low == high:
                    return sorted_samples[low]
                frac = idx - low
                return sorted_samples[low] * (1.0 - frac) + sorted_samples[high] * frac

            return LatencyStats(
                min_ms=float(self._min_ms if self._min_ms != float("inf") else 0.0),
                max_ms=float(self._max_ms),
                mean_ms=float(mean_val),
                p50_ms=float(get_percentile(50.0)),
                p90_ms=float(get_percentile(90.0)),
                p95_ms=float(get_percentile(95.0)),
                p99_ms=float(get_percentile(99.0)),
                sample_count=count,
            )

    def reset(self) -> None:
        """Reset all recorded samples and counters in the histogram."""
        with self._lock:
            self._samples.clear()
            self._index = 0
            self._is_full = False
            self._total_count = 0
            self._sum_ms = 0.0
            self._min_ms = float("inf")
            self._max_ms = 0.0


@dataclass(frozen=True)
class LatencyStats:
    """Statistical summary of a latency distribution in milliseconds."""

    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    sample_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LatencyMetrics:
    """Structured latency breakdowns across queue, execution, and end-to-end stages."""

    queue: LatencyStats
    execution: LatencyStats
    end_to_end: LatencyStats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "queue": self.queue.to_dict(),
            "execution": self.execution.to_dict(),
            "end_to_end": self.end_to_end.to_dict(),
        }


@dataclass(frozen=True)
class RequestMetrics:
    """Request-level operational tracking and outcome counters."""

    submitted: int = 0
    completed: int = 0
    failed: int = 0
    rejected: int = 0
    cancelled: int = 0
    active: int = 0
    queue_depth: int = 0
    peak_queue_depth: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BatchMetrics:
    """Dynamic batching aggregation and capacity utilization metrics."""

    batches_formed: int = 0
    samples_processed: int = 0
    total_batch_size: int = 0
    average_batch_size: float = 0.0
    min_batch_size: int = 0
    max_batch_size: int = 0
    peak_batch_size: int = 0
    batch_utilization: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThroughputStats:
    """Monotonic throughput execution rates."""

    requests_per_sec: float = 0.0
    samples_per_sec: float = 0.0
    batches_per_sec: float = 0.0
    window_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendMetrics:
    """Backend execution breakdown and automatic fallback telemetry."""

    numpy_count: int = 0
    native_count: int = 0
    native_fused_count: int = 0
    numpy_fused_count: int = 0
    compiled_count: int = 0
    eager_count: int = 0
    native_requested: int = 0
    native_executed: int = 0
    native_fallback: int = 0
    fallback_reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numpy_count": self.numpy_count,
            "native_count": self.native_count,
            "native_fused_count": self.native_fused_count,
            "numpy_fused_count": self.numpy_fused_count,
            "compiled_count": self.compiled_count,
            "eager_count": self.eager_count,
            "native_requested": self.native_requested,
            "native_executed": self.native_executed,
            "native_fallback": self.native_fallback,
            "fallback_reasons": dict(self.fallback_reasons),
        }


@dataclass(frozen=True)
class CompilerMetrics:
    """Inference compiler cache hits, misses, and execution planning analytics."""

    compile_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0
    compiled_executions: int = 0
    eager_executions: int = 0
    plan_reuse_count: int = 0
    total_compilation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryMetrics:
    """Memory planner, workspace arena, and parameter storage metrics."""

    workspace_bytes: int = 0
    peak_workspace_bytes: int = 0
    planned_workspace_bytes: int = 0
    active_workspace_bytes: int = 0
    parameter_bytes: int = 0
    model_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SchedulerMetrics:
    """Scheduler queue depth and configuration telemetry."""

    queue_depth: int = 0
    max_queue_size: int = 0
    max_batch_size: int = 0
    batch_timeout_ms: float = 0.0
    policy: str = "FIFO"
    lifecycle_state: str = "RUNNING"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Immutable, unified diagnostic and performance analytics snapshot."""

    requests: RequestMetrics
    batches: BatchMetrics
    latency: LatencyMetrics
    throughput: ThroughputStats
    backends: BackendMetrics
    compiler: CompilerMetrics
    memory: MemoryMetrics
    scheduler: Optional[SchedulerMetrics] = None
    timestamp: float = field(default_factory=time.time)
    tensorforge_version: str = LIBRARY_VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to a structured dictionary."""
        d = {
            "requests": self.requests.to_dict(),
            "batches": self.batches.to_dict(),
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "backends": self.backends.to_dict(),
            "compiler": self.compiler.to_dict(),
            "memory": self.memory.to_dict(),
            "timestamp": self.timestamp,
            "tensorforge_version": self.tensorforge_version,
        }
        if self.scheduler is not None:
            d["scheduler"] = self.scheduler.to_dict()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize snapshot to formatted JSON."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str, indent: int = 2) -> None:
        """Export snapshot to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json(indent=indent))

    def summary(self) -> str:
        """Format a human-readable terminal performance report."""
        lines = [
            "=" * 70,
            f"TensorForge Inference Performance Analytics (v{self.tensorforge_version})",
            "=" * 70,
            f"Requests:       Submitted={self.requests.submitted} | Completed={self.requests.completed} | Failed={self.requests.failed} | Rejected={self.requests.rejected}",
            f"Batches:        Formed={self.batches.batches_formed} | Avg Size={self.batches.average_batch_size:.2f} | Utilization={self.batches.batch_utilization * 100:.1f}%",
            f"Throughput:     {self.throughput.samples_per_sec:.1f} samples/sec | {self.throughput.requests_per_sec:.1f} req/sec | {self.throughput.batches_per_sec:.1f} batch/sec",
            "-" * 70,
            "Latency Distribution (ms):",
            f"  Queue Wait:   Mean={self.latency.queue.mean_ms:.3f} | P50={self.latency.queue.p50_ms:.3f} | P95={self.latency.queue.p95_ms:.3f} | P99={self.latency.queue.p99_ms:.3f}",
            f"  Execution:    Mean={self.latency.execution.mean_ms:.3f} | P50={self.latency.execution.p50_ms:.3f} | P95={self.latency.execution.p95_ms:.3f} | P99={self.latency.execution.p99_ms:.3f}",
            f"  End-to-End:   Mean={self.latency.end_to_end.mean_ms:.3f} | P50={self.latency.end_to_end.p50_ms:.3f} | P95={self.latency.end_to_end.p95_ms:.3f} | P99={self.latency.end_to_end.p99_ms:.3f}",
            "-" * 70,
            f"Backends:       NumPy={self.backends.numpy_count} | Native={self.backends.native_count} | Fused={self.backends.native_fused_count + self.backends.numpy_fused_count} | Fallback={self.backends.native_fallback}",
            f"Compiler:       Hits={self.compiler.cache_hits} | Misses={self.compiler.cache_misses} | Hit Rate={self.compiler.cache_hit_rate * 100:.1f}%",
            f"Memory:         Workspace={self.memory.workspace_bytes} B | Peak={self.memory.peak_workspace_bytes} B | Params={self.memory.parameter_bytes} B",
            "=" * 70,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class MetricsCollector:
    """Thread-safe, central metrics accumulator for the inference stack.

    Collects request, dynamic batching, backend execution, compiler, and memory metrics
    with bounded memory histograms and minimal critical section locking.

    Args:
        history_size: Capacity of bounded latency histograms (default: 2048).
    """

    def __init__(self, history_size: int = 2048) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._history_size: int = history_size

        # Monotonic time base
        self._start_time_ns: int = time.perf_counter_ns()

        # Bounded latency histograms
        self._queue_hist: LatencyHistogram = LatencyHistogram(capacity=history_size)
        self._exec_hist: LatencyHistogram = LatencyHistogram(capacity=history_size)
        self._e2e_hist: LatencyHistogram = LatencyHistogram(capacity=history_size)

        # Request counters
        self._submitted: int = 0
        self._completed: int = 0
        self._failed: int = 0
        self._rejected: int = 0
        self._cancelled: int = 0
        self._active: int = 0
        self._queue_depth: int = 0
        self._peak_queue_depth: int = 0

        # Batch counters
        self._batches_formed: int = 0
        self._samples_processed: int = 0
        self._total_batch_size: int = 0
        self._min_batch_size: int = 0
        self._max_batch_size: int = 0
        self._peak_batch_size: int = 0
        self._configured_max_batch: int = 32

        # Backend counters
        self._numpy_count: int = 0
        self._native_count: int = 0
        self._native_fused_count: int = 0
        self._numpy_fused_count: int = 0
        self._compiled_count: int = 0
        self._eager_count: int = 0
        self._native_requested: int = 0
        self._native_executed: int = 0
        self._native_fallback: int = 0
        self._fallback_reasons: Dict[str, int] = collections.defaultdict(int)

        # Compiler counters
        self._compile_requests: int = 0
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._total_compilation_ns: int = 0

        # Memory counters
        self._workspace_bytes: int = 0
        self._peak_workspace_bytes: int = 0
        self._planned_workspace_bytes: int = 0
        self._active_workspace_bytes: int = 0
        self._parameter_bytes: int = 0
        self._model_size_bytes: int = 0

        # Scheduler metadata
        self._scheduler_metrics: Optional[SchedulerMetrics] = None

    def record_request_submitted(self, queue_depth: int = 0) -> None:
        """Record an incoming request submission."""
        with self._lock:
            self._submitted += 1
            self._active += 1
            self._queue_depth = queue_depth
            if queue_depth > self._peak_queue_depth:
                self._peak_queue_depth = queue_depth

    def record_request_completed(
        self,
        queue_wait_ms: float = 0.0,
        exec_ms: float = 0.0,
        e2e_ms: float = 0.0,
        samples: int = 1,
    ) -> None:
        """Record a successful request completion and its latency components."""
        self._queue_hist.record(queue_wait_ms)
        self._exec_hist.record(exec_ms)
        self._e2e_hist.record(e2e_ms)

        with self._lock:
            self._completed += 1
            if self._active > 0:
                self._active -= 1

    def record_request_failed(self, exec_ms: float = 0.0) -> None:
        """Record an inference execution failure."""
        if exec_ms > 0.0:
            self._exec_hist.record(exec_ms)

        with self._lock:
            self._failed += 1
            if self._active > 0:
                self._active -= 1

    def record_request_rejected(self, reason: str = "limit") -> None:
        """Record a rejected request (e.g. queue full, resource limit)."""
        with self._lock:
            self._rejected += 1
            if self._active > 0:
                self._active -= 1

    def record_request_cancelled(self) -> None:
        """Record a cancelled pending request."""
        with self._lock:
            self._cancelled += 1
            if self._active > 0:
                self._active -= 1

    def record_batch(
        self,
        batch_size: int,
        configured_max_batch: int = 32,
    ) -> None:
        """Record dynamic batch formation and sizing metrics."""
        with self._lock:
            self._batches_formed += 1
            self._samples_processed += batch_size
            self._total_batch_size += batch_size
            self._configured_max_batch = configured_max_batch

            if self._min_batch_size == 0 or batch_size < self._min_batch_size:
                self._min_batch_size = batch_size
            if batch_size > self._max_batch_size:
                self._max_batch_size = batch_size
            if batch_size > self._peak_batch_size:
                self._peak_batch_size = batch_size

    def record_backend(
        self,
        backend: str,
        is_fused: bool = False,
        is_compiled: bool = False,
        was_fallback: bool = False,
        fallback_reason: Optional[str] = None,
    ) -> None:
        """Record backend execution and fallback occurrence."""
        with self._lock:
            backend_lower = backend.lower()
            if "native" in backend_lower:
                if is_fused:
                    self._native_fused_count += 1
                else:
                    self._native_count += 1
                self._native_executed += 1
            else:
                if is_fused:
                    self._numpy_fused_count += 1
                else:
                    self._numpy_count += 1

            if is_compiled:
                self._compiled_count += 1
            else:
                self._eager_count += 1

            if was_fallback:
                self._native_fallback += 1
                if fallback_reason:
                    self._fallback_reasons[fallback_reason] += 1

    def record_compiler(self, cache_hit: bool, compilation_ns: int = 0) -> None:
        """Record compiler cache hit or miss event."""
        with self._lock:
            self._compile_requests += 1
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
            if compilation_ns > 0:
                self._total_compilation_ns += compilation_ns

    def record_memory(
        self,
        workspace_bytes: int = 0,
        peak_bytes: int = 0,
        planned_bytes: int = 0,
        active_bytes: int = 0,
        parameter_bytes: int = 0,
        model_size_bytes: int = 0,
        param_bytes: int = 0,
    ) -> None:
        """Update workspace and parameter memory telemetry."""
        with self._lock:
            self._workspace_bytes = workspace_bytes
            self._peak_workspace_bytes = max(self._peak_workspace_bytes, peak_bytes, workspace_bytes)
            if planned_bytes > 0:
                self._planned_workspace_bytes = planned_bytes
            if active_bytes > 0:
                self._active_workspace_bytes = active_bytes
            p_bytes = parameter_bytes or param_bytes
            if p_bytes > 0:
                self._parameter_bytes = p_bytes
            if model_size_bytes > 0:
                self._model_size_bytes = model_size_bytes

    def set_scheduler_metrics(self, scheduler_metrics: SchedulerMetrics) -> None:
        """Attach scheduler queue state and configuration."""
        with self._lock:
            self._scheduler_metrics = scheduler_metrics

    def snapshot(self) -> PerformanceSnapshot:
        """Generate an immutable PerformanceSnapshot of all collected metrics.

        Returns:
            PerformanceSnapshot object.
        """
        now_ns = time.perf_counter_ns()
        with self._lock:
            elapsed_sec = max(1e-9, (now_ns - self._start_time_ns) / 1e9)

            # 1. Request metrics
            req_metrics = RequestMetrics(
                submitted=self._submitted,
                completed=self._completed,
                failed=self._failed,
                rejected=self._rejected,
                cancelled=self._cancelled,
                active=self._active,
                queue_depth=self._queue_depth,
                peak_queue_depth=self._peak_queue_depth,
            )

            # 2. Batch metrics
            batches = self._batches_formed
            samples = self._samples_processed
            avg_batch_sz = (samples / batches) if batches > 0 else (float(samples) if samples > 0 else 0.0)
            max_cfg = max(1, self._configured_max_batch)
            utilization = min(1.0, max(0.0, avg_batch_sz / max_cfg)) if batches > 0 else 0.0

            batch_metrics = BatchMetrics(
                batches_formed=batches,
                samples_processed=samples,
                total_batch_size=self._total_batch_size,
                average_batch_size=float(avg_batch_sz),
                min_batch_size=self._min_batch_size,
                max_batch_size=self._max_batch_size,
                peak_batch_size=self._peak_batch_size,
                batch_utilization=float(utilization),
            )

            # 3. Latency metrics
            latency_metrics = LatencyMetrics(
                queue=self._queue_hist.stats(),
                execution=self._exec_hist.stats(),
                end_to_end=self._e2e_hist.stats(),
            )

            # 4. Throughput metrics
            throughput_stats = ThroughputStats(
                requests_per_sec=float(self._completed / elapsed_sec),
                samples_per_sec=float((samples if samples > 0 else self._completed) / elapsed_sec),
                batches_per_sec=float(batches / elapsed_sec),
                window_seconds=float(elapsed_sec),
            )

            # 5. Backend metrics
            backend_metrics = BackendMetrics(
                numpy_count=self._numpy_count,
                native_count=self._native_count,
                native_fused_count=self._native_fused_count,
                numpy_fused_count=self._numpy_fused_count,
                compiled_count=self._compiled_count,
                eager_count=self._eager_count,
                native_requested=self._native_requested,
                native_executed=self._native_executed,
                native_fallback=self._native_fallback,
                fallback_reasons=dict(self._fallback_reasons),
            )

            # 6. Compiler metrics
            total_cache_lookups = self._cache_hits + self._cache_misses
            hit_rate = (self._cache_hits / total_cache_lookups) if total_cache_lookups > 0 else 0.0
            comp_time_ms = (self._total_compilation_ns / 1e6)

            compiler_metrics = CompilerMetrics(
                compile_requests=self._compile_requests,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_hit_rate=float(hit_rate),
                compiled_executions=self._compiled_count,
                eager_executions=self._eager_count,
                plan_reuse_count=self._cache_hits,
                total_compilation_time_ms=float(comp_time_ms),
            )

            # 7. Memory metrics
            memory_metrics = MemoryMetrics(
                workspace_bytes=self._workspace_bytes,
                peak_workspace_bytes=self._peak_workspace_bytes,
                planned_workspace_bytes=self._planned_workspace_bytes,
                active_workspace_bytes=self._active_workspace_bytes,
                parameter_bytes=self._parameter_bytes,
                model_size_bytes=self._model_size_bytes,
            )

            return PerformanceSnapshot(
                requests=req_metrics,
                batches=batch_metrics,
                latency=latency_metrics,
                throughput=throughput_stats,
                backends=backend_metrics,
                compiler=compiler_metrics,
                memory=memory_metrics,
                scheduler=self._scheduler_metrics,
                timestamp=time.time(),
                tensorforge_version=LIBRARY_VERSION,
            )

    def reset(self) -> None:
        """Reset all metric counters, latency reservoirs, and timers."""
        with self._lock:
            self._start_time_ns = time.perf_counter_ns()
            self._queue_hist.reset()
            self._exec_hist.reset()
            self._e2e_hist.reset()

            self._submitted = 0
            self._completed = 0
            self._failed = 0
            self._rejected = 0
            self._cancelled = 0
            self._active = 0
            self._queue_depth = 0
            self._peak_queue_depth = 0

            self._batches_formed = 0
            self._samples_processed = 0
            self._total_batch_size = 0
            self._min_batch_size = 0
            self._max_batch_size = 0
            self._peak_batch_size = 0

            self._numpy_count = 0
            self._native_count = 0
            self._native_fused_count = 0
            self._numpy_fused_count = 0
            self._compiled_count = 0
            self._eager_count = 0
            self._native_requested = 0
            self._native_executed = 0
            self._native_fallback = 0
            self._fallback_reasons.clear()

            self._compile_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._total_compilation_ns = 0

            self._workspace_bytes = 0
            self._peak_workspace_bytes = 0
            self._planned_workspace_bytes = 0
            self._active_workspace_bytes = 0
            self._parameter_bytes = 0
            self._model_size_bytes = 0
            self._scheduler_metrics = None
