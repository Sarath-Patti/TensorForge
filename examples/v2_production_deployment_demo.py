"""TensorForge v2.0 Production Deployment & Inference Client Demo.

Demonstrates:
1. Declarative deployment manifest configuration & JSON export/import
2. Server bootstrapping via InferenceServer.from_manifest()
3. Pre-packaged workload runtime profiles (LOW_LATENCY, HIGH_THROUGHPUT, BALANCED)
4. Application integration via InferenceClient and InferenceRequestContract
5. Synchronous, batch, and asynchronous inference through high-level client API
"""

import os
import tempfile
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.inference import (
    DeploymentManifest,
    InferenceClient,
    InferenceRequestContract,
    InferenceServer,
    ModelDeploymentSpec,
    ServerConfig,
)
from tensorforge.serialization import save_model


def main() -> None:
    print("=" * 70)
    print("TensorForge v2.0 Production Deployment & Inference Client Demo")
    print(f"TensorForge Version: {tf.__version__}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Create and save model artifacts
        m1_path = os.path.join(tmpdir, "classifier_v1.tfmodel")
        m2_path = os.path.join(tmpdir, "embedder_v1.tfmodel")

        save_model(nn.Linear(8, 4), m1_path)
        save_model(nn.Sequential(nn.Linear(16, 8), nn.ReLU()), m2_path)

        print("\n1. Defining Declarative Deployment Manifest...")
        manifest = DeploymentManifest(
            name="production_serving_cluster",
            server_config=ServerConfig(max_loaded_models=10),
            models=[
                ModelDeploymentSpec(
                    name="classifier",
                    version="1",
                    path=m1_path,
                    profile_type="LOW_LATENCY",
                ),
                ModelDeploymentSpec(
                    name="embedder",
                    version="1",
                    path=m2_path,
                    profile_type="HIGH_THROUGHPUT",
                ),
            ],
        )

        manifest_path = os.path.join(tmpdir, "deployment_manifest.json")
        manifest.save_json(manifest_path)
        print(f"   ✓ Saved manifest to: {manifest_path}")

        print("\n2. Bootstrapping InferenceServer from Manifest...")
        with InferenceServer.from_manifest(manifest_path) as server:
            print("   ✓ InferenceServer successfully bootstrapped!")

            print("\n3. Connecting Application via InferenceClient...")
            with InferenceClient(server) as client:
                print(f"   ✓ InferenceClient active: {client}")

                # Model Discovery
                models_info = client.models()
                print(f"   ✓ Registered Models Count: {len(models_info)}")
                for m in models_info:
                    print(f"     - Model: {m['name']} (v{m['active_version']})")

                # Synchronous Prediction via Contract
                print("\n4. Executing Synchronous Inference Request...")
                input_data = tf.randn((1, 8))
                contract = InferenceRequestContract(
                    model="classifier",
                    inputs=input_data,
                    version="1",
                    timeout_ms=100.0,
                )
                output = client.execute_contract(contract)
                print(f"   ✓ Contract Prediction Output Shape: {output.shape}")

                # Batch Prediction
                print("\n5. Executing Batch Prediction Request...")
                batch_inputs = [tf.randn((1, 16)) for _ in range(5)]
                batch_outputs = client.predict_batch("embedder", batch_inputs)
                print(f"   ✓ Batch Prediction processed {len(batch_outputs)} requests successfully!")

                # Asynchronous Request Submission
                print("\n6. Asynchronous Request Submission...")
                future = client.submit("classifier", tf.randn((1, 8)))
                async_output = future.result()
                print(f"   ✓ Async Future Result Shape: {async_output.shape}")

                # Health and Performance Diagnostics
                print("\n7. Inspecting Client Diagnostics & Performance Snapshot...")
                health = client.health()
                snapshot = client.performance_snapshot()
                print(f"   ✓ Server Health Status: {health['status']}")
                print(f"   ✓ Total Requests Completed: {snapshot['server']['stats']['completed_requests']}")

    print("\n" + "=" * 70)
    print("TensorForge v2.0 Production Deployment Demo Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
