"""Production-oriented Runtime Profiler and Observability Subsystem for TensorForge.

Provides high-resolution monotonic timing, per-operator performance breakdown,
backend execution distribution, latency statistics (min, max, mean, p50, p95, p99),
compiler cache analytics, and workspace memory telemetry with near-zero overhead when disabled.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Dict, List, Optional, Tuple


class ProfileEvent:
    """Immutable descriptor for a single profiled operation or execution step."""

    __slots__ = (
        "name",
        "op_type",
        "backend",
        "mode",
        "start_time_ns",
        "end_time_ns",
        "input_shape",
        "output_shape",
        "dtype",
        "batch_size",
        "estimated_flops",
        "workspace_bytes",
        "num_threads",
        "is_fused",
        "is_compiled",
        "context_id",
        "extra",
    )

    def __init__(
        self,
        name: str,
        op_type: str,
        backend: str,
        mode: str = "compiled",
        start_time_ns: int = 0,
        end_time_ns: int = 0,
        input_shape: Tuple[int, ...] = (),
        output_shape: Tuple[int, ...] = (),
        dtype: str = "float32",
        batch_size: int = 1,
        estimated_flops: int = 0,
        workspace_bytes: int = 0,
        num_threads: int = 1,
        is_fused: bool = False,
        is_compiled: bool = False,
        context_id: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name: str = name
        self.op_type: str = op_type
        self.backend: str = backend
        self.mode: str = mode
        self.start_time_ns: int = start_time_ns
        self.end_time_ns: int = end_time_ns
        self.input_shape: Tuple[int, ...] = tuple(input_shape)
        self.output_shape: Tuple[int, ...] = tuple(output_shape)
        self.dtype: str = dtype
        self.batch_size: int = batch_size
        self.estimated_flops: int = estimated_flops
        self.workspace_bytes: int = workspace_bytes
        self.num_threads: int = num_threads
        self.is_fused: bool = is_fused
        self.is_compiled: bool = is_compiled
        self.context_id: int = context_id
        self.extra: Dict[str, Any] = extra or {}

    @property
    def duration_ns(self) -> int:
        """Elapsed duration in nanoseconds."""
        return max(0, self.end_time_ns - self.start_time_ns)

    @property
    def duration_us(self) -> float:
        """Elapsed duration in microseconds."""
        return self.duration_ns / 1_000.0

    @property
    def duration_ms(self) -> float:
        """Elapsed duration in milliseconds."""
        return self.duration_ns / 1_000_000.0

    @property
    def duration_sec(self) -> float:
        """Elapsed duration in seconds."""
        return self.duration_ns / 1_000_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert event properties into a dictionary."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "backend": self.backend,
            "mode": self.mode,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "duration_us": self.duration_us,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "estimated_flops": self.estimated_flops,
            "workspace_bytes": self.workspace_bytes,
            "num_threads": self.num_threads,
            "is_fused": self.is_fused,
            "is_compiled": self.is_compiled,
            "context_id": self.context_id,
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        return (
            f"ProfileEvent(name='{self.name}', op='{self.op_type}', backend='{self.backend}', "
            f"duration={self.duration_ms:.3f}ms, shape={self.input_shape}->{self.output_shape})"
        )


class PerformanceReport:
    """Structured, printable performance diagnostic report for TensorForge inference."""

    def __init__(
        self,
        prediction_count: int,
        total_time_ms: float,
        latency_stats: Dict[str, float],
        backend_stats: Dict[str, Any],
        operation_stats: Dict[str, Any],
        memory_stats: Dict[str, Any],
        compiler_stats: Dict[str, Any],
        events: List[ProfileEvent],
    ) -> None:
        self.prediction_count: int = prediction_count
        self.total_time_ms: float = total_time_ms
        self.latency_stats: Dict[str, float] = latency_stats
        self.backend_stats: Dict[str, Any] = backend_stats
        self.operation_stats: Dict[str, Any] = operation_stats
        self.memory_stats: Dict[str, Any] = memory_stats
        self.compiler_stats: Dict[str, Any] = compiler_stats
        self.events: List[ProfileEvent] = events

    def summary(self) -> str:
        """Produce a formatted high-level summary string."""
        lines = [
            "=" * 60,
            "TensorForge Performance Report",
            "=" * 60,
            f"Predictions:       {self.prediction_count}",
            f"Total Time:        {self.total_time_ms:.3f} ms",
            f"Mean Latency:      {self.latency_stats.get('mean_ms', 0.0):.4f} ms",
            f"P50 Latency:       {self.latency_stats.get('p50_ms', 0.0):.4f} ms",
            f"P95 Latency:       {self.latency_stats.get('p95_ms', 0.0):.4f} ms",
            f"P99 Latency:       {self.latency_stats.get('p99_ms', 0.0):.4f} ms",
            f"Throughput:        {self.latency_stats.get('throughput_samples_per_sec', 0.0):.1f} samples/sec",
            "",
            self.operation_breakdown(),
            "",
            self.backend_breakdown(),
            "",
            self.memory_summary(),
            "",
            self.compiler_summary(),
            "=" * 60,
        ]
        return "\n".join(lines)

    def operation_breakdown(self) -> str:
        """Produce an operation breakdown table."""
        lines = ["Operations Breakdown", "-" * 60]
        total_op_time = sum(info.get("time_ms", 0.0) for info in self.operation_stats.values())
        if not self.operation_stats or total_op_time == 0:
            lines.append("  (No operation-level events recorded)")
            return "\n".join(lines)

        lines.append(f"{'Operator':<24} {'Count':<8} {'Time (ms)':<12} {'Percent':<10}")
        lines.append("-" * 60)
        sorted_ops = sorted(self.operation_stats.items(), key=lambda x: x[1].get("time_ms", 0.0), reverse=True)
        for op, data in sorted_ops:
            op_time = data.get("time_ms", 0.0)
            op_count = data.get("count", 0)
            pct = (op_time / total_op_time * 100.0) if total_op_time > 0 else 0.0
            lines.append(f"{op:<24} {op_count:<8} {op_time:<12.4f} {pct:>6.1f}%")
        return "\n".join(lines)

    def backend_breakdown(self) -> str:
        """Produce a backend execution distribution table."""
        lines = ["Backend Execution Breakdown", "-" * 60]
        native_ops = self.backend_stats.get("native", {}).get("operations", 0)
        numpy_ops = self.backend_stats.get("numpy", {}).get("operations", 0)
        fallbacks = self.backend_stats.get("fallbacks", 0)
        native_time = self.backend_stats.get("native", {}).get("time_ms", 0.0)
        numpy_time = self.backend_stats.get("numpy", {}).get("time_ms", 0.0)
        total_time = native_time + numpy_time

        native_pct = (native_time / total_time * 100.0) if total_time > 0 else (100.0 if native_ops > 0 else 0.0)
        numpy_pct = (numpy_time / total_time * 100.0) if total_time > 0 else (100.0 if numpy_ops > 0 else 0.0)

        lines.append(f"{'Backend':<16} {'Operations':<12} {'Time (ms)':<12} {'Percent':<10}")
        lines.append("-" * 60)
        lines.append(f"{'Native':<16} {native_ops:<12} {native_time:<12.4f} {native_pct:>6.1f}%")
        lines.append(f"{'NumPy':<16} {numpy_ops:<12} {numpy_time:<12.4f} {numpy_pct:>6.1f}%")
        if fallbacks > 0:
            lines.append(f"Fallbacks: {fallbacks}")
        return "\n".join(lines)

    def latency_summary(self) -> str:
        """Produce latency summary string."""
        lines = [
            "Latency Distribution",
            "-" * 60,
            f"  Min Latency:     {self.latency_stats.get('min_ms', 0.0):.4f} ms",
            f"  Mean Latency:    {self.latency_stats.get('mean_ms', 0.0):.4f} ms",
            f"  P50 Latency:     {self.latency_stats.get('p50_ms', 0.0):.4f} ms",
            f"  P95 Latency:     {self.latency_stats.get('p95_ms', 0.0):.4f} ms",
            f"  P99 Latency:     {self.latency_stats.get('p99_ms', 0.0):.4f} ms",
            f"  Max Latency:     {self.latency_stats.get('max_ms', 0.0):.4f} ms",
            f"  Throughput:      {self.latency_stats.get('throughput_samples_per_sec', 0.0):.1f} samples/sec",
        ]
        return "\n".join(lines)

    def memory_summary(self) -> str:
        """Produce memory telemetry summary string."""
        ws_bytes = self.memory_stats.get("workspace_bytes", 0)
        ws_kb = ws_bytes / 1024.0
        regions = self.memory_stats.get("num_regions", 0)
        reused = self.memory_stats.get("reused_buffers", 0)
        active_ctx = self.memory_stats.get("active_contexts", 0)
        pooled_ctx = self.memory_stats.get("pooled_contexts", 0)

        lines = [
            "Memory Telemetry",
            "-" * 60,
            f"  Workspace Memory: {ws_bytes} bytes ({ws_kb:.2f} KB)",
            f"  Memory Regions:   {regions}",
            f"  Reused Buffers:   {reused}",
            f"  Active Contexts:  {active_ctx}",
            f"  Pooled Contexts:  {pooled_ctx}",
        ]
        return "\n".join(lines)

    def compiler_summary(self) -> str:
        """Produce compiler and plan cache summary string."""
        hits = self.compiler_stats.get("cache_hits", 0)
        misses = self.compiler_stats.get("cache_misses", 0)
        total_queries = hits + misses
        hit_rate = (hits / total_queries * 100.0) if total_queries > 0 else 0.0
        compile_time = self.compiler_stats.get("total_compilation_time_ms", 0.0)
        cached_plans = self.compiler_stats.get("cached_plan_count", 0)

        lines = [
            "Compiler & Plan Cache",
            "-" * 60,
            f"  Compilations:     {misses}",
            f"  Cache Hits:       {hits}",
            f"  Cache Misses:     {misses}",
            f"  Cache Hit Rate:   {hit_rate:.1f}%",
            f"  Cached Plans:     {cached_plans}",
            f"  Compilation Time: {compile_time:.3f} ms",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export full performance diagnostic report as a nested dictionary."""
        return {
            "prediction_count": self.prediction_count,
            "total_time_ms": self.total_time_ms,
            "latency": self.latency_stats,
            "backend": self.backend_stats,
            "operations": self.operation_stats,
            "memory": self.memory_stats,
            "compiler": self.compiler_stats,
            "event_count": len(self.events),
        }

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"PerformanceReport(predictions={self.prediction_count}, "
            f"mean_latency={self.latency_stats.get('mean_ms', 0.0):.3f}ms, "
            f"p95={self.latency_stats.get('p95_ms', 0.0):.3f}ms)"
        )


class ProfileSession:
    """Scoped context manager for temporary runtime profiling."""

    def __init__(self, profiler: RuntimeProfiler, detailed: bool = True) -> None:
        self._profiler: RuntimeProfiler = profiler
        self._detailed: bool = detailed
        self._prev_enabled: bool = False
        self._prev_detailed: bool = False
        self._start_time_ns: int = 0
        self._end_time_ns: int = 0
        self._events: List[ProfileEvent] = []
        self._initial_event_count: int = 0

    def __enter__(self) -> ProfileSession:
        self._prev_enabled = self._profiler.is_enabled
        self._prev_detailed = self._profiler.is_detailed
        self._initial_event_count = len(self._profiler.get_events())
        self._profiler.enable(detailed=self._detailed)
        self._start_time_ns = time.perf_counter_ns()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        self._end_time_ns = time.perf_counter_ns()
        all_events = self._profiler.get_events()
        self._events = all_events[self._initial_event_count :]
        if not self._prev_enabled:
            self._profiler.disable()
        else:
            self._profiler.enable(detailed=self._prev_detailed)

    @property
    def duration_ns(self) -> int:
        """Total duration of the session scope in nanoseconds."""
        return max(0, self._end_time_ns - self._start_time_ns)

    @property
    def duration_ms(self) -> float:
        """Total duration of the session scope in milliseconds."""
        return self.duration_ns / 1_000_000.0

    @property
    def duration_us(self) -> float:
        """Total duration of the session scope in microseconds."""
        return self.duration_ns / 1_000.0

    @property
    def events(self) -> List[ProfileEvent]:
        """List of events collected during this session."""
        return self._events

    def summary(self) -> str:
        """Formatted summary of the session."""
        lines = [
            f"ProfileSession (Duration: {self.duration_ms:.3f} ms, Events: {len(self._events)})",
            "-" * 60,
        ]
        for ev in self._events:
            lines.append(f"  [{ev.backend}] {ev.name:<24} {ev.duration_ms:>8.4f} ms ({ev.input_shape}->{ev.output_shape})")
        return "\n".join(lines)

    def report(self) -> PerformanceReport:
        """Generate a PerformanceReport from this session."""
        return self._profiler.generate_report(session_events=self._events)


class RuntimeProfiler:
    """Thread-safe performance profiler and telemetry accumulator for TensorForge InferenceRuntime."""

    def __init__(self, history_size: int = 1000) -> None:
        self._enabled: bool = False
        self._detailed: bool = False
        self._history_size: int = max(10, history_size)
        self._lock: threading.Lock = threading.Lock()

        # Latency & throughput tracking
        self._prediction_count: int = 0
        self._total_samples: int = 0
        self._total_prediction_time_ns: int = 0
        self._latencies_ns: collections.deque[int] = collections.deque(maxlen=self._history_size)

        # Backend operation statistics
        self._backend_ops: Dict[str, int] = {
            "native": 0,
            "numpy": 0,
            "native_fused": 0,
            "numpy_fused": 0,
            "fallbacks": 0,
        }
        self._backend_times_ns: Dict[str, int] = {
            "native": 0,
            "numpy": 0,
        }

        # Operator statistics
        self._op_counts: Dict[str, int] = collections.defaultdict(int)
        self._op_times_ns: Dict[str, int] = collections.defaultdict(int)

        # Compiler / cache statistics
        self._compiler_stats: Dict[str, Any] = {
            "compilation_count": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_compilation_time_ns": 0,
            "cached_plan_count": 0,
        }

        # Event storage
        self._events: List[ProfileEvent] = []

    def enable(self, detailed: bool = False) -> None:
        """Enable inference runtime profiling."""
        with self._lock:
            self._enabled = True
            self._detailed = bool(detailed)

    def disable(self) -> None:
        """Disable inference runtime profiling."""
        with self._lock:
            self._enabled = False
            self._detailed = False

    @property
    def is_enabled(self) -> bool:
        """Check if profiling is currently active."""
        return self._enabled

    @property
    def is_detailed(self) -> bool:
        """Check if detailed per-operator profiling is currently active."""
        return self._enabled and self._detailed

    @property
    def mode(self) -> str:
        """Return the current profiling mode string ('disabled', 'summary', or 'detailed')."""
        if not self._enabled:
            return "disabled"
        return "detailed" if self._detailed else "summary"

    def set_history_size(self, size: int) -> None:
        """Set the maximum capacity of the bounded latency history buffer."""
        with self._lock:
            self._history_size = max(10, size)
            new_deque: collections.deque[int] = collections.deque(self._latencies_ns, maxlen=self._history_size)
            self._latencies_ns = new_deque

    def record_prediction(self, duration_ns: int, batch_size: int = 1) -> None:
        """Record a completed prediction invocation."""
        if not self._enabled:
            return
        with self._lock:
            self._prediction_count += 1
            self._total_samples += max(1, batch_size)
            self._total_prediction_time_ns += duration_ns
            self._latencies_ns.append(duration_ns)

    def record_event(self, event: ProfileEvent) -> None:
        """Record a fine-grained operator profiling event."""
        if not self._enabled:
            return
        with self._lock:
            self._events.append(event)
            op_key = event.op_type
            if event.is_fused and "activation" in event.extra:
                op_key = f"{event.op_type}({event.extra['activation'].capitalize()})"
            self._op_counts[op_key] += 1
            self._op_times_ns[op_key] += event.duration_ns

    def record_backend_op(
        self,
        backend_dispatch: str,
        duration_ns: int,
        is_fused: bool = False,
        is_fallback: bool = False,
    ) -> None:
        """Record backend execution statistics."""
        if not self._enabled:
            return
        with self._lock:
            is_native = backend_dispatch.startswith("native")
            backend_group = "native" if is_native else "numpy"

            if is_native:
                self._backend_ops["native"] += 1
                if is_fused or "fused" in backend_dispatch:
                    self._backend_ops["native_fused"] += 1
                self._backend_times_ns["native"] += duration_ns
            else:
                self._backend_ops["numpy"] += 1
                if is_fused or "fused" in backend_dispatch:
                    self._backend_ops["numpy_fused"] += 1
                self._backend_times_ns["numpy"] += duration_ns

            if is_fallback:
                self._backend_ops["fallbacks"] += 1

    def record_compiler_event(
        self,
        cache_hit: bool,
        compilation_time_ns: int = 0,
        cached_plans: int = 0,
    ) -> None:
        """Record compiler cache hit or miss event."""
        with self._lock:
            if cache_hit:
                self._compiler_stats["cache_hits"] += 1
            else:
                self._compiler_stats["cache_misses"] += 1
                self._compiler_stats["compilation_count"] += 1
                self._compiler_stats["total_compilation_time_ns"] += compilation_time_ns
            if cached_plans > 0:
                self._compiler_stats["cached_plan_count"] = cached_plans

    def latency_stats(self) -> Dict[str, float]:
        """Compute latency statistics (min, max, mean, p50, p95, p99) and throughput."""
        with self._lock:
            if not self._latencies_ns:
                return {
                    "prediction_count": self._prediction_count,
                    "total_samples": self._total_samples,
                    "total_time_ms": self._total_prediction_time_ns / 1_000_000.0,
                    "min_ms": 0.0,
                    "max_ms": 0.0,
                    "mean_ms": 0.0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "throughput_samples_per_sec": 0.0,
                }

            sorted_lat = sorted(self._latencies_ns)
            n = len(sorted_lat)
            min_ns = sorted_lat[0]
            max_ns = sorted_lat[-1]
            mean_ns = sum(sorted_lat) / n
            p50_ns = sorted_lat[int(0.50 * (n - 1))]
            p95_ns = sorted_lat[int(0.95 * (n - 1))]
            p99_ns = sorted_lat[int(0.99 * (n - 1))]

            total_sec = self._total_prediction_time_ns / 1_000_000_000.0
            throughput = (self._total_samples / total_sec) if total_sec > 0 else 0.0

            return {
                "prediction_count": self._prediction_count,
                "total_samples": self._total_samples,
                "total_time_ms": self._total_prediction_time_ns / 1_000_000.0,
                "min_ms": min_ns / 1_000_000.0,
                "max_ms": max_ns / 1_000_000.0,
                "mean_ms": mean_ns / 1_000_000.0,
                "p50_ms": p50_ns / 1_000_000.0,
                "p95_ms": p95_ns / 1_000_000.0,
                "p99_ms": p99_ns / 1_000_000.0,
                "throughput_samples_per_sec": throughput,
            }

    def backend_stats(self) -> Dict[str, Any]:
        """Return backend execution telemetry."""
        with self._lock:
            return {
                "native": {
                    "operations": self._backend_ops["native"],
                    "fused_operations": self._backend_ops["native_fused"],
                    "time_ms": self._backend_times_ns["native"] / 1_000_000.0,
                },
                "numpy": {
                    "operations": self._backend_ops["numpy"],
                    "fused_operations": self._backend_ops["numpy_fused"],
                    "time_ms": self._backend_times_ns["numpy"] / 1_000_000.0,
                },
                "fallbacks": self._backend_ops["fallbacks"],
            }

    def operation_stats(self) -> Dict[str, Any]:
        """Return aggregated operator-level breakdown."""
        with self._lock:
            res: Dict[str, Any] = {}
            for op, count in self._op_counts.items():
                time_ms = self._op_times_ns[op] / 1_000_000.0
                res[op] = {
                    "count": count,
                    "time_ms": time_ms,
                    "avg_time_ms": (time_ms / count) if count > 0 else 0.0,
                }
            return res

    def compiler_stats(self) -> Dict[str, Any]:
        """Return compiler cache and compilation statistics."""
        with self._lock:
            hits = self._compiler_stats["cache_hits"]
            misses = self._compiler_stats["cache_misses"]
            total = hits + misses
            hit_rate = (hits / total) if total > 0 else 0.0
            return {
                "compilation_count": self._compiler_stats["compilation_count"],
                "cache_hits": hits,
                "cache_misses": misses,
                "cache_hit_rate": hit_rate,
                "cached_plan_count": self._compiler_stats["cached_plan_count"],
                "total_compilation_time_ms": self._compiler_stats["total_compilation_time_ns"] / 1_000_000.0,
            }

    def get_events(self) -> List[ProfileEvent]:
        """Return a snapshot list of recorded profile events."""
        with self._lock:
            return list(self._events)

    def clear(self) -> None:
        """Reset all profiling statistics and events."""
        with self._lock:
            self._prediction_count = 0
            self._total_samples = 0
            self._total_prediction_time_ns = 0
            self._latencies_ns.clear()
            self._backend_ops = {
                "native": 0,
                "numpy": 0,
                "native_fused": 0,
                "numpy_fused": 0,
                "fallbacks": 0,
            }
            self._backend_times_ns = {"native": 0, "numpy": 0}
            self._op_counts.clear()
            self._op_times_ns.clear()
            self._compiler_stats = {
                "compilation_count": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_compilation_time_ns": 0,
                "cached_plan_count": 0,
            }
            self._events.clear()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()

    def generate_report(
        self,
        runtime_memory_stats: Optional[Dict[str, Any]] = None,
        session_events: Optional[List[ProfileEvent]] = None,
    ) -> PerformanceReport:
        """Generate a comprehensive PerformanceReport."""
        lat = self.latency_stats()
        bk = self.backend_stats()
        ops = self.operation_stats()
        comp = self.compiler_stats()
        mem = runtime_memory_stats or {}
        events = session_events if session_events is not None else self.get_events()

        return PerformanceReport(
            prediction_count=int(lat.get("prediction_count", 0)),
            total_time_ms=lat.get("total_time_ms", 0.0),
            latency_stats=lat,
            backend_stats=bk,
            operation_stats=ops,
            memory_stats=mem,
            compiler_stats=comp,
            events=events,
        )
