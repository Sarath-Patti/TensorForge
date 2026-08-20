"""TensorForge v1.8 – Production Inference Serving Layer Demonstration.

Demonstrates:
1. Creating an InferenceServer instance.
2. Exporting and loading multiple neural network models into the server registry.
3. Managing model versions and setting default active versions.
4. Performing synchronous predictions (`predict`) routed via model name & version.
5. Performing asynchronous request submissions (`submit`) using futures.
6. Displaying server health (`server.health()`) and loaded models metadata (`server.models()`).
7. Generating an aggregated PerformanceSnapshot across all registered models.
8. Demonstrating safe atomic version switching (`set_active_version`).
9. Unloading an old model version cleanly (`unload_model`).
10. Shutting down the server gracefully.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    InferenceServer,
    SchedulerConfig,
    ServerConfig,
)
from tensorforge.serialization import save_model


def main() -> None:
    print("=" * 105)
    print("TensorForge v1.8 – Production Inference Serving Layer Demonstration")
    print("=" * 105)

    with tempfile.TemporaryDirectory() as tmpdir:
        classifier_v1_path = os.path.join(tmpdir, "classifier_v1.tfmodel")
        classifier_v2_path = os.path.join(tmpdir, "classifier_v2.tfmodel")
        detector_path = os.path.join(tmpdir, "detector_v1.tfmodel")
        metrics_export_path = os.path.join(tmpdir, "server_performance_snapshot.json")

        # --- [Step 1] Construct and Export Neural Network Models ---
        print("\n--- [Step 1] Constructing and Exporting Models ---")
        save_model(nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4)), classifier_v1_path)
        save_model(nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 4)), classifier_v2_path)
        save_model(nn.Sequential(nn.Linear(32, 128), nn.Tanh(), nn.Linear(128, 8)), detector_path)
        print("  ✓ Exported model archives for classifier (v1, v2) and detector (v1).")

        # --- [Step 2] Initializing InferenceServer ---
        print("\n--- [Step 2] Initializing Production InferenceServer ---")
        server_config = ServerConfig(max_loaded_models=10, max_total_pending_requests=100)
        with InferenceServer(config=server_config) as server:

            # Load classifier v1 (active)
            server.load_model(
                name="classifier",
                path=classifier_v1_path,
                version="1",
                active=True,
                scheduler_config=SchedulerConfig(max_batch_size=8, batch_timeout_ms=3.0),
            )
            print("  ✓ Loaded model 'classifier:1' (Active)")

            # Load classifier v2 (inactive draft)
            server.load_model(
                name="classifier",
                path=classifier_v2_path,
                version="2",
                active=False,
                scheduler_config=SchedulerConfig(max_batch_size=16, batch_timeout_ms=2.0),
            )
            print("  ✓ Loaded model 'classifier:2' (Staged)")

            # Load detector v1 (active)
            server.load_model(
                name="detector",
                path=detector_path,
                version="1",
                active=True,
            )
            print("  ✓ Loaded model 'detector:1' (Active)")

            # --- [Step 3] Inspecting Registry & Models Metadata ---
            print("\n--- [Step 3] Model Registry Inspection ---")
            models_meta = server.models()
            for m in models_meta:
                print(f"  Model: {m['name']:<12} Version: {m['version']:<4} Active: {str(m['is_active']):<6} State: {m['state']}")

            # --- [Step 4] Synchronous Request Routing ---
            print("\n--- [Step 4] Synchronous Request Routing (predict) ---")
            x_cls = tf.randn((2, 16))
            x_det = tf.randn((3, 32))

            # Default active version (v1) for classifier
            out_cls_v1 = server.predict(model="classifier", inputs=x_cls)
            print(f"  ✓ 'classifier' default predict (v1) -> shape: {out_cls_v1.shape}")

            # Explicit version 2 for classifier
            out_cls_v2 = server.predict(model="classifier", inputs=x_cls, version="2")
            print(f"  ✓ 'classifier:2' explicit predict -> shape: {out_cls_v2.shape}")

            # Detector predict
            out_det = server.predict(model="detector", inputs=x_det)
            print(f"  ✓ 'detector:1' predict -> shape: {out_det.shape}")

            # --- [Step 5] Asynchronous Request Submission ---
            print("\n--- [Step 5] Asynchronous Request Submission (submit) ---")
            fut1 = server.submit(model="classifier", inputs=x_cls)
            fut2 = server.submit(model="detector", inputs=x_det)

            res1 = fut1.result(timeout=2.0)
            res2 = fut2.result(timeout=2.0)
            print(f"  ✓ Async Future 1 result -> shape: {res1.shape}")
            print(f"  ✓ Async Future 2 result -> shape: {res2.shape}")

            # --- [Step 6] Atomic Active Version Switching ---
            print("\n--- [Step 6] Atomic Active Version Switch ---")
            print(f"  Active version before switch: {server.get_active_version('classifier')}")
            server.set_active_version("classifier", "2")
            print(f"  Active version after switch:  {server.get_active_version('classifier')}")

            out_switched = server.predict(model="classifier", inputs=x_cls)
            print(f"  ✓ 'classifier' default predict now routes to v2 -> shape: {out_switched.shape}")

            # --- [Step 7] Server Health & Statistics ---
            print("\n--- [Step 7] Operational Health & Diagnostics ---")
            health_report = server.health()
            print(f"  Server Status:        {health_report['status']}")
            print(f"  Loaded Models Count:  {health_report['loaded_models_count']}")
            print(f"  Ready Models Count:   {health_report['ready_models_count']}")

            stats_report = server.stats()
            print(f"  Total Requests:       {stats_report['submitted_requests']}")
            print(f"  Completed Requests:   {stats_report['completed_requests']}")

            # --- [Step 8] Performance Analytics Snapshot ---
            print("\n--- [Step 8] Exporting Performance Snapshot to JSON ---")
            server.export_metrics(metrics_export_path, indent=2)
            print(f"  Successfully exported performance metrics to: '{metrics_export_path}'")

            # --- [Step 9] Unloading Model ---
            print("\n--- [Step 9] Unloading Old Version (classifier:1) ---")
            server.unload_model(name="classifier", version="1")
            print("  ✓ Successfully unloaded 'classifier:1'")

    print("\n" + "=" * 105)
    print("TensorForge v1.8 Production Inference Serving Layer Demo Finished Successfully!")
    print("=" * 105)


if __name__ == "__main__":
    main()
