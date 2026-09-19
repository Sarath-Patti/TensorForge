"""TensorForge v2.2.0: Throughput & Concurrency Scaling Benchmark Suite.

Measures:
1. Batch Size Scaling: Throughput and latency across batch sizes 1, 2, 4, 8, 16, 32, 64, 128.
2. Concurrency Scaling: Thread parallelism scaling efficiency across 1, 2, 4, 8 workers.
3. Dynamic Batching Impact: Comparing unbatched vs dynamic batching under concurrent load.

Formula:
  scaling_efficiency = actual_throughput / (single_worker_throughput * worker_count)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime, InferenceServer, SchedulerConfig
from tensorforge.serialization import save_model


class BatchScalingResult:
    """Descriptor for a single batch size scaling benchmark measurement."""

    def __init__(
        self,
        batch_size: int,
        throughput_samples_per_sec: float,
        requests_per_sec: float,
        avg_latency_ms: float,
        p50_latency_ms: float,
        p95_latency_ms: float,
        workspace_bytes: int,
    ) -> None:
        self.batch_size: int = batch_size
        self.throughput_samples_per_sec: float = throughput_samples_per_sec
        self.requests_per_sec: float = requests_per_sec
        self.avg_latency_ms: float = avg_latency_ms
        self.p50_latency_ms: float = p50_latency_ms
        self.p95_latency_ms: float = p95_latency_ms
        self.workspace_bytes: int = workspace_bytes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 4),
            "workspace_bytes": self.workspace_bytes,
        }


class ConcurrencyScalingResult:
    """Descriptor for a worker thread concurrency scaling measurement."""

    def __init__(
        self,
        worker_count: int,
        throughput_samples_per_sec: float,
        avg_latency_ms: float,
        scaling_efficiency: float,
    ) -> None:
        self.worker_count: int = worker_count
        self.throughput_samples_per_sec: float = throughput_samples_per_sec
        self.avg_latency_ms: float = avg_latency_ms
        self.scaling_efficiency: float = scaling_efficiency

    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_count": self.worker_count,
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "scaling_efficiency": round(self.scaling_efficiency, 4),
            "scaling_efficiency_pct": round(self.scaling_efficiency * 100.0, 1),
        }


class DynamicBatchingScalingResult:
    """Descriptor for dynamic batching comparison measurement."""

    def __init__(
        self,
        enabled: bool,
        throughput_samples_per_sec: float,
        avg_latency_ms: float,
        completed_requests: int,
        batches_formed: int,
    ) -> None:
        self.enabled: bool = enabled
        self.throughput_samples_per_sec: float = throughput_samples_per_sec
        self.avg_latency_ms: float = avg_latency_ms
        self.completed_requests: int = completed_requests
        self.batches_formed: int = batches_formed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dynamic_batching_enabled": self.enabled,
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "completed_requests": self.completed_requests,
            "batches_formed": self.batches_formed,
        }


class ThroughputScalingSuite:
    """Benchmark suite measuring batch size scaling, thread concurrency scaling, and dynamic batching."""

    def __init__(
        self,
        batch_sizes: Optional[List[int]] = None,
        worker_counts: Optional[List[int]] = None,
        iterations: int = 100,
        in_features: int = 64,
        out_features: int = 10,
    ) -> None:
        self.batch_sizes: List[int] = batch_sizes or [1, 2, 4, 8, 16, 32, 64, 128]
        self.worker_counts: List[int] = worker_counts or [1, 2, 4, 8]
        self.iterations: int = iterations
        self.in_features: int = in_features
        self.out_features: int = out_features

        self.batch_results: List[BatchScalingResult] = []
        self.concurrency_results: List[ConcurrencyScalingResult] = []
        self.dynamic_batching_results: List[DynamicBatchingScalingResult] = []

    def _create_model(self, tmpdir: str) -> str:
        model = nn.Sequential(
            nn.Linear(self.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.out_features),
            nn.Softmax(dim=-1),
        )
        model_path = os.path.join(tmpdir, "model.tfmodel")
        save_model(model, model_path)
        return model_path

    def run_batch_size_scaling(self) -> List[BatchScalingResult]:
        """Benchmark throughput scaling across increasing batch sizes."""
        self.batch_results.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = self._create_model(tmpdir)

            for b in self.batch_sizes:
                runtime = InferenceRuntime.load(model_path).compile(input_shape=(b, self.in_features))
                x_input = tf.randn((b, self.in_features))

                # Warmup
                for _ in range(10):
                    _ = runtime.predict(x_input)

                # Measure
                lats_ms: List[float] = []
                t0 = time.perf_counter_ns()
                for _ in range(self.iterations):
                    it0 = time.perf_counter_ns()
                    _ = runtime.predict(x_input)
                    it1 = time.perf_counter_ns()
                    lats_ms.append((it1 - it0) / 1_000_000.0)

                t1 = time.perf_counter_ns()
                total_sec = max(1e-9, (t1 - t0) / 1_000_000_000.0)
                total_samples = self.iterations * b

                throughput = total_samples / total_sec
                requests_sec = self.iterations / total_sec

                sorted_lats = sorted(lats_ms)
                n = len(sorted_lats)
                avg_lat = sum(sorted_lats) / n
                p50_lat = sorted_lats[int(0.50 * (n - 1))]
                p95_lat = sorted_lats[int(0.95 * (n - 1))]

                res = BatchScalingResult(
                    batch_size=b,
                    throughput_samples_per_sec=throughput,
                    requests_per_sec=requests_sec,
                    avg_latency_ms=avg_lat,
                    p50_latency_ms=p50_lat,
                    p95_latency_ms=p95_lat,
                    workspace_bytes=runtime.workspace_size,
                )
                self.batch_results.append(res)
                runtime.close()

        return self.batch_results

    def run_concurrency_scaling(self, eval_batch_size: int = 16) -> List[ConcurrencyScalingResult]:
        """Benchmark worker thread parallelism scaling and compute scaling efficiency."""
        self.concurrency_results.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = self._create_model(tmpdir)
            base_throughput = 0.0

            for w in self.worker_counts:
                runtime = InferenceRuntime.load(model_path, num_threads=w).compile(
                    input_shape=(eval_batch_size, self.in_features),
                    num_threads=w,
                )

                x_input = tf.randn((eval_batch_size, self.in_features))

                # Warmup
                for _ in range(10):
                    _ = runtime.predict(x_input)

                t0 = time.perf_counter_ns()
                for _ in range(self.iterations):
                    _ = runtime.predict(x_input)
                t1 = time.perf_counter_ns()

                total_sec = max(1e-9, (t1 - t0) / 1_000_000_000.0)
                total_samples = self.iterations * eval_batch_size
                throughput = total_samples / total_sec
                avg_lat = (total_sec * 1000.0) / self.iterations

                if w == 1 or base_throughput == 0.0:
                    base_throughput = throughput
                    scaling_eff = 1.0
                else:
                    ideal_linear_throughput = base_throughput * w
                    scaling_eff = throughput / ideal_linear_throughput if ideal_linear_throughput > 0 else 1.0

                res = ConcurrencyScalingResult(
                    worker_count=w,
                    throughput_samples_per_sec=throughput,
                    avg_latency_ms=avg_lat,
                    scaling_efficiency=scaling_eff,
                )
                self.concurrency_results.append(res)
                runtime.close()

        return self.concurrency_results

    def run_dynamic_batching_scaling(self, num_producers: int = 4, requests_per_producer: int = 25) -> List[DynamicBatchingScalingResult]:
        """Benchmark dynamic batching impact comparing unbatched vs dynamic batching under concurrent load."""
        self.dynamic_batching_results.clear()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = self._create_model(tmpdir)

            for enabled in [False, True]:
                with InferenceServer() as server:
                    scheduler_cfg = SchedulerConfig(max_batch_size=16, batch_timeout_ms=5.0) if enabled else SchedulerConfig(max_batch_size=1)
                    server.load_model("classifier", model_path, scheduler_config=scheduler_cfg)

                    sample_input = np.ones((1, self.in_features), dtype=np.float32)

                    # Multi-threaded producer submission
                    def producer():
                        for _ in range(requests_per_producer):
                            _ = server.predict("classifier", sample_input)

                    t0 = time.perf_counter_ns()
                    threads = [threading.Thread(target=producer) for _ in range(num_producers)]
                    for t in threads:
                        t.start()
                    for t in threads:
                        t.join()
                    t1 = time.perf_counter_ns()

                    total_sec = max(1e-9, (t1 - t0) / 1_000_000_000.0)
                    total_requests = num_producers * requests_per_producer
                    throughput = total_requests / total_sec
                    avg_lat = (total_sec * 1000.0) / total_requests

                    stats = server.stats()
                    batches_formed = stats.get("batches_formed", 0)

                    res = DynamicBatchingScalingResult(
                        enabled=enabled,
                        throughput_samples_per_sec=throughput,
                        avg_latency_ms=avg_lat,
                        completed_requests=total_requests,
                        batches_formed=batches_formed,
                    )
                    self.dynamic_batching_results.append(res)

        return self.dynamic_batching_results

    def summary_table(self) -> str:
        """Produce formatted ASCII tables for batch size, concurrency, and dynamic batching scaling."""
        lines = []

        if self.batch_results:
            lines.extend([
                "=" * 85,
                "Batch Size Scaling Performance",
                "=" * 85,
                f"{'Batch Size':<12} | {'Throughput (samp/s)':<22} | {'Avg Lat (ms)':<14} | {'P95 Lat (ms)':<14} | {'Workspace (B)':<12}",
                "-" * 85,
            ])
            for b in self.batch_results:
                lines.append(
                    f"{b.batch_size:<12} | {b.throughput_samples_per_sec:>20.1f}   | "
                    f"{b.avg_latency_ms:>12.4f}   | {b.p95_latency_ms:>12.4f}   | {b.workspace_bytes:>12}"
                )

        if self.concurrency_results:
            lines.extend([
                "\n" + "=" * 85,
                "Worker Thread Concurrency Scaling Efficiency",
                "=" * 85,
                f"{'Worker Threads':<16} | {'Throughput (samp/s)':<22} | {'Avg Lat (ms)':<14} | {'Scaling Efficiency':<18}",
                "-" * 85,
            ])
            for c in self.concurrency_results:
                eff_pct = c.scaling_efficiency * 100.0
                lines.append(
                    f"{c.worker_count:<16} | {c.throughput_samples_per_sec:>20.1f}   | "
                    f"{c.avg_latency_ms:>12.4f}   | {eff_pct:>16.1f}%"
                )

        if self.dynamic_batching_results:
            lines.extend([
                "\n" + "=" * 85,
                "Dynamic Batching Concurrent Scaling Comparison",
                "=" * 85,
                f"{'Dynamic Batching':<20} | {'Throughput (req/s)':<22} | {'Avg Lat (ms)':<14} | {'Batches Formed':<14}",
                "-" * 85,
            ])
            for d in self.dynamic_batching_results:
                label = "Enabled (Max=16)" if d.enabled else "Disabled (Max=1)"
                lines.append(
                    f"{label:<20} | {d.throughput_samples_per_sec:>20.1f}   | "
                    f"{d.avg_latency_ms:>12.4f}   | {d.batches_formed:>14}"
                )

        lines.append("=" * 85)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Export full scaling suite results into structured dictionary format."""
        return {
            "batch_scaling": [b.to_dict() for b in self.batch_results],
            "concurrency_scaling": [c.to_dict() for c in self.concurrency_results],
            "dynamic_batching_scaling": [d.to_dict() for d in self.dynamic_batching_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export scaling suite results as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, filepath: str, indent: int = 2) -> None:
        """Save scaling suite results to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)


def run_throughput_scaling_suite(iterations: int = 100) -> ThroughputScalingSuite:
    """Convenience runner function for batch size, worker concurrency, and dynamic batching scaling."""
    suite = ThroughputScalingSuite(iterations=iterations)
    suite.run_batch_size_scaling()
    suite.run_concurrency_scaling()
    suite.run_dynamic_batching_scaling()
    return suite


if __name__ == "__main__":
    suite = run_throughput_scaling_suite()
    print(suite.summary_table())
