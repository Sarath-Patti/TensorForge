"""TensorForge v2.2.0: Performance Optimization Experiments Suite.

Establishes a rigorous BASELINE -> OPTIMIZATION -> MEASUREMENT -> COMPARISON benchmark framework.

Stages:
  - Stage A: Baseline (Eager execution, no operator fusion, no compilation)
  - Stage B: Operator Fusion (Graph-level fused execution)
  - Stage C: Static Memory Planning & Compiled Execution (Compiled ExecutionPlan IR)
  - Stage D: INT8 Quantization (INT8 quantized model & runtime)

Models:
  - Small MLP  (16 -> 32 -> 8)
  - Medium MLP (64 -> 128 -> 64 -> 10)
  - Larger MLP (256 -> 512 -> 256 -> 128 -> 32)
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


class OptimizationExperimentResult:
    """Immutable result structure for a single optimization stage experiment."""

    def __init__(
        self,
        model_name: str,
        stage_name: str,
        batch_size: int,
        iterations: int,
        throughput_samples_per_sec: float,
        requests_per_sec: float,
        avg_latency_ms: float,
        min_latency_ms: float,
        max_latency_ms: float,
        p50_latency_ms: float,
        p95_latency_ms: float,
        workspace_bytes: int,
        peak_memory_bytes: int,
        speedup_vs_baseline: float = 1.0,
        latency_change_pct: float = 0.0,
        memory_reduction_pct: float = 0.0,
        backend: str = "numpy",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name: str = model_name
        self.stage_name: str = stage_name
        self.batch_size: int = batch_size
        self.iterations: int = iterations
        self.throughput_samples_per_sec: float = throughput_samples_per_sec
        self.requests_per_sec: float = requests_per_sec
        self.avg_latency_ms: float = avg_latency_ms
        self.min_latency_ms: float = min_latency_ms
        self.max_latency_ms: float = max_latency_ms
        self.p50_latency_ms: float = p50_latency_ms
        self.p95_latency_ms: float = p95_latency_ms
        self.workspace_bytes: int = workspace_bytes
        self.peak_memory_bytes: int = peak_memory_bytes
        self.speedup_vs_baseline: float = speedup_vs_baseline
        self.latency_change_pct: float = latency_change_pct
        self.memory_reduction_pct: float = memory_reduction_pct
        self.backend: str = backend
        self.extra: Dict[str, Any] = extra or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment result to dictionary representation."""
        return {
            "model_name": self.model_name,
            "stage_name": self.stage_name,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "throughput_samples_per_sec": round(self.throughput_samples_per_sec, 2),
            "requests_per_sec": round(self.requests_per_sec, 2),
            "avg_latency_ms": round(self.avg_latency_ms, 4),
            "min_latency_ms": round(self.min_latency_ms, 4),
            "max_latency_ms": round(self.max_latency_ms, 4),
            "p50_latency_ms": round(self.p50_latency_ms, 4),
            "p95_latency_ms": round(self.p95_latency_ms, 4),
            "workspace_bytes": self.workspace_bytes,
            "peak_memory_bytes": self.peak_memory_bytes,
            "speedup_vs_baseline": round(self.speedup_vs_baseline, 2),
            "latency_change_pct": round(self.latency_change_pct, 2),
            "memory_reduction_pct": round(self.memory_reduction_pct, 2),
            "backend": self.backend,
            "extra": self.extra,
        }

    def __repr__(self) -> str:
        return (
            f"OptimizationExperimentResult(stage='{self.stage_name}', "
            f"throughput={self.throughput_samples_per_sec:.1f} samp/s, "
            f"speedup={self.speedup_vs_baseline:.2f}x, avg_lat={self.avg_latency_ms:.3f}ms)"
        )


class OptimizationExperimentSuite:
    """Benchmark framework executing controlled optimization experiments across model architectures."""

    def __init__(
        self,
        iterations: int = 100,
        warmup_iterations: int = 10,
        batch_size: int = 16,
    ) -> None:
        self.iterations: int = iterations
        self.warmup_iterations: int = warmup_iterations
        self.batch_size: int = batch_size
        self.results: List[OptimizationExperimentResult] = []

    @staticmethod
    def get_benchmark_models() -> Dict[str, Tuple[nn.Module, Tuple[int, ...]]]:
        """Return standardized benchmark model architectures and input shapes."""
        small_mlp = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.Softmax(dim=-1),
        )

        medium_mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1),
        )

        larger_mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Softmax(dim=-1),
        )

        return {
            "Small MLP (16->32->8)": (small_mlp, (16,)),
            "Medium MLP (64->128->64->10)": (medium_mlp, (64,)),
            "Larger MLP (256->512->256->128->32)": (larger_mlp, (256,)),
        }

    def run_suite(self) -> List[OptimizationExperimentResult]:
        """Run full controlled optimization suite across all benchmark models and stages."""
        self.results.clear()
        models_dict = self.get_benchmark_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            for model_name, (model, in_feat) in models_dict.items():
                model_path = os.path.join(tmpdir, "model.tfmodel")
                save_model(model, model_path)
                in_shape = (self.batch_size, in_feat[0])
                input_tensor = tf.randn(in_shape)

                # -----------------------------------------------------------------
                # Stage A: Baseline (Eager, No Fusion, No Compile)
                # -----------------------------------------------------------------
                runtime_a = InferenceRuntime.load(model_path)
                res_a = self._measure_stage(
                    model_name=model_name,
                    stage_name="Stage A: Baseline (Eager)",
                    runtime=runtime_a,
                    input_tensor=input_tensor,
                )
                self.results.append(res_a)
                runtime_a.close()

                # -----------------------------------------------------------------
                # Stage B: Operator Fusion
                # -----------------------------------------------------------------
                runtime_b = InferenceRuntime.load(model_path)
                runtime_b.optimize()
                res_b = self._measure_stage(
                    model_name=model_name,
                    stage_name="Stage B: Operator Fusion",
                    runtime=runtime_b,
                    input_tensor=input_tensor,
                    baseline_res=res_a,
                )
                self.results.append(res_b)
                runtime_b.close()

                # -----------------------------------------------------------------
                # Stage C: Static Memory Planning / Compiled Execution
                # -----------------------------------------------------------------
                runtime_c = InferenceRuntime.load(model_path)
                runtime_c.compile(input_shape=in_shape)
                res_c = self._measure_stage(
                    model_name=model_name,
                    stage_name="Stage C: Compiled + Memory Plan",
                    runtime=runtime_c,
                    input_tensor=input_tensor,
                    baseline_res=res_a,
                )
                self.results.append(res_c)
                runtime_c.close()

                # -----------------------------------------------------------------
                # Stage D: INT8 Quantization
                # -----------------------------------------------------------------
                try:
                    int8_model_path = os.path.join(tmpdir, "quantized_model.tfmodel")
                    q_state_dict = {}
                    for name, param in model.named_parameters():
                        q_state_dict[name] = quantize(param, scheme="symmetric")

                    write_tfmodel_container(
                        int8_model_path,
                        q_state_dict,
                        metadata={"is_quantized": True, "scheme": "symmetric"},
                        architecture=extract_module_architecture(model),
                    )

                    runtime_d = InferenceRuntime.load(int8_model_path)
                    res_d = self._measure_stage(
                        model_name=model_name,
                        stage_name="Stage D: INT8 Quantization",
                        runtime=runtime_d,
                        input_tensor=input_tensor,
                        baseline_res=res_a,
                    )
                    res_d.extra["status"] = "supported"
                    self.results.append(res_d)
                    runtime_d.close()
                except Exception as exc:
                    res_d = OptimizationExperimentResult(
                        model_name=model_name,
                        stage_name="Stage D: INT8 Quantization",
                        batch_size=self.batch_size,
                        iterations=self.iterations,
                        throughput_samples_per_sec=0.0,
                        requests_per_sec=0.0,
                        avg_latency_ms=0.0,
                        min_latency_ms=0.0,
                        max_latency_ms=0.0,
                        p50_latency_ms=0.0,
                        p95_latency_ms=0.0,
                        workspace_bytes=0,
                        peak_memory_bytes=0,
                        speedup_vs_baseline=0.0,
                        latency_change_pct=0.0,
                        memory_reduction_pct=0.0,
                        backend="numpy",
                        extra={"status": "unsupported", "reason": str(exc)},
                    )
                    self.results.append(res_d)

        return self.results

    def _measure_stage(
        self,
        model_name: str,
        stage_name: str,
        runtime: InferenceRuntime,
        input_tensor: tf.Tensor,
        baseline_res: Optional[OptimizationExperimentResult] = None,
    ) -> OptimizationExperimentResult:
        """Measure latency, throughput, and memory metrics for a single optimization stage."""
        # Warmup
        for _ in range(self.warmup_iterations):
            _ = runtime.predict(input_tensor)

        # Measurement iterations
        latencies_ns: List[int] = []
        t0 = time.perf_counter_ns()

        for _ in range(self.iterations):
            iter_t0 = time.perf_counter_ns()
            _ = runtime.predict(input_tensor)
            iter_t1 = time.perf_counter_ns()
            latencies_ns.append(iter_t1 - iter_t0)

        t1 = time.perf_counter_ns()
        total_sec = max(1e-9, (t1 - t0) / 1_000_000_000.0)
        total_samples = self.iterations * self.batch_size

        throughput = total_samples / total_sec
        requests_per_sec = self.iterations / total_sec

        sorted_lat_ms = sorted([lat / 1_000_000.0 for lat in latencies_ns])
        n = len(sorted_lat_ms)
        avg_lat = sum(sorted_lat_ms) / n
        min_lat = sorted_lat_ms[0]
        max_lat = sorted_lat_ms[-1]
        p50_lat = sorted_lat_ms[int(0.50 * (n - 1))]
        p95_lat = sorted_lat_ms[int(0.95 * (n - 1))]

        workspace_bytes = runtime.workspace_size
        summary = runtime.summary()
        peak_memory_bytes = summary.get("workspace_bytes", workspace_bytes)

        if baseline_res is not None and baseline_res.throughput_samples_per_sec > 0:
            speedup = throughput / baseline_res.throughput_samples_per_sec
            lat_change = ((avg_lat - baseline_res.avg_latency_ms) / baseline_res.avg_latency_ms) * 100.0
            if baseline_res.workspace_bytes > 0:
                mem_red = ((baseline_res.workspace_bytes - workspace_bytes) / baseline_res.workspace_bytes) * 100.0
            else:
                mem_red = 0.0
        else:
            speedup = 1.0
            lat_change = 0.0
            mem_red = 0.0

        return OptimizationExperimentResult(
            model_name=model_name,
            stage_name=stage_name,
            batch_size=self.batch_size,
            iterations=self.iterations,
            throughput_samples_per_sec=throughput,
            requests_per_sec=requests_per_sec,
            avg_latency_ms=avg_lat,
            min_latency_ms=min_lat,
            max_latency_ms=max_lat,
            p50_latency_ms=p50_lat,
            p95_latency_ms=p95_lat,
            workspace_bytes=workspace_bytes,
            peak_memory_bytes=peak_memory_bytes,
            speedup_vs_baseline=speedup,
            latency_change_pct=lat_change,
            memory_reduction_pct=mem_red,
            backend=runtime.backend,
        )

    def summary_table(self) -> str:
        """Produce a formatted ASCII comparison table across all measured models and stages."""
        if not self.results:
            return "No benchmark results recorded."

        lines = [
            "=" * 115,
            f"TensorForge Optimization Experiments Suite (Batch Size={self.batch_size}, Iterations={self.iterations})",
            "=" * 115,
            f"{'Model':<28} | {'Optimization Stage':<32} | {'Throughput (s/s)':<16} | {'Speedup':<8} | {'Avg (ms)':<10} | {'Workspace (B)':<12}",
            "-" * 115,
        ]

        current_model = None
        for res in self.results:
            if current_model != res.model_name:
                if current_model is not None:
                    lines.append("-" * 115)
                current_model = res.model_name

            if res.extra.get("status") == "unsupported":
                lines.append(
                    f"{res.model_name:<28} | {res.stage_name:<32} | "
                    f"{'UNSUPPORTED':>14}   | {'N/A':>7} | "
                    f"{'N/A':>8}   | {0:>11}"
                )
            else:
                lines.append(
                    f"{res.model_name:<28} | {res.stage_name:<32} | "
                    f"{res.throughput_samples_per_sec:>14.1f}   | {res.speedup_vs_baseline:>6.2f}x | "
                    f"{res.avg_latency_ms:>8.4f}   | {res.workspace_bytes:>11}"
                )

        lines.append("=" * 115)
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert full suite results into structured dictionary format."""
        return {
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "warmup_iterations": self.warmup_iterations,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        """Export full suite results as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def export_json(self, filepath: str, indent: int = 2) -> None:
        """Export suite results to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=indent)


def run_optimization_experiments(
    iterations: int = 100,
    batch_size: int = 16,
) -> OptimizationExperimentSuite:
    """Convenience runner function executing the optimization experiment suite."""
    suite = OptimizationExperimentSuite(iterations=iterations, batch_size=batch_size)
    suite.run_suite()
    return suite


if __name__ == "__main__":
    suite = run_optimization_experiments()
    print(suite.summary_table())
