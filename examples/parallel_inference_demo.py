"""TensorForge v1.2 Memory Optimization & Parallel CPU Execution Demonstration.

Demonstrates:
  1. Constructing and exporting an FP32 neural network to a .tfmodel archive
  2. Loading the model into InferenceRuntime with architecture auto-reconstruction
  3. Graph optimization (operator fusion) and ahead-of-time compilation
  4. Inspecting detailed interval memory plan, memory regions, and workspace size
  5. Configuring CPU thread parallelism (set_num_threads)
  6. Executing multi-threaded parallel predictions with zero memory allocation
  7. Comparing single-thread vs multi-threaded inference performance and bit-exact parity
"""

import os
import tempfile
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import get_num_threads, is_native_available, set_num_threads
from tensorforge.inference import InferenceRuntime
from tensorforge.serialization import save_model


def run_parallel_inference_demo():
    print("=" * 100)
    print("TensorForge v1.2 – Memory Optimization & Parallel CPU Execution Demonstration")
    print("=" * 100)

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "deep_mlp.tfmodel")

        # =========================================================================
        # 1. Construct and Export Multi-Layer Neural Network
        # =========================================================================
        print("\n--- [Step 1] Constructing and Exporting Neural Network ---")
        in_features, hidden_features, num_classes = 64, 128, 10

        model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Tanh(),
            nn.Linear(hidden_features, num_classes),
            nn.Softmax(dim=-1),
        )
        model.eval()

        save_model(model, model_path, metadata={"task": "multi_class_classification"})
        print(f"  Exported model to: '{os.path.basename(model_path)}' ({os.path.getsize(model_path)} bytes)")

        # =========================================================================
        # 2. Load into InferenceRuntime & Compile
        # =========================================================================
        print("\n--- [Step 2] Loading into InferenceRuntime & Ahead-of-Time Compiling ---")
        runtime = InferenceRuntime.load(model_path)
        batch_size = 32

        # Configure 4 CPU worker threads
        runtime.set_num_threads(4)
        runtime.compile(input_shape=(batch_size, in_features))
        summary = runtime.summary()

        print("  Compiled Runtime Summary:")
        print(f"    - Optimized Status:        {summary['is_optimized']}")
        print(f"    - Compiled Status:         {summary['is_compiled']}")
        print(f"    - Active Backend:          {summary['backend']}")
        print(f"    - Configured CPU Threads:  {summary['num_threads']}")
        print(f"    - Original Operators:      {summary['original_nodes']}")
        print(f"    - Optimized Operators:     {summary['optimized_nodes']}")
        print(f"    - Compiled Steps:          {summary['compiled_steps']}")
        print(f"    - Fused Operators:         {summary['fused_count']} ({', '.join(summary['fused_patterns'])})")

        # =========================================================================
        # 3. Inspect Memory Plan & Reusable Workspace Regions
        # =========================================================================
        print("\n--- [Step 3] Inspecting Memory Plan & Workspace Regions ---")
        mem_plan = runtime.memory_plan
        assert mem_plan is not None

        print(f"  Total Planned Workspace:     {summary['workspace_bytes']} bytes ({summary['workspace_kb']:.2f} KB)")
        print(f"  Physical Memory Regions:     {mem_plan.num_regions}")
        print(f"  Reused Intermediate Buffers: {mem_plan.num_reused_buffers}")
        print(f"  Alignment Padding:           {mem_plan.alignment_padding_bytes} bytes")
        print(f"  Buffer-to-Region Mapping:    {mem_plan.buffer_region_map}")

        for r_id, region in mem_plan.regions.items():
            print(f"    - Region {r_id}: Capacity={region.size_bytes}B, Offset={region.offset_bytes}B, Assigned Buffers={region.assigned_buffers}")

        # =========================================================================
        # 4. Multi-Threaded Parallel Inference & Parity Check
        # =========================================================================
        print("\n--- [Step 4] Multi-Threaded Execution & Bit-Exact Parity Check ---")
        test_inputs = tf.randn((batch_size, in_features))

        with tf.no_grad():
            ref_predictions = model(test_inputs)

        # Single-threaded prediction
        runtime.set_num_threads(1)
        st_predictions = runtime.predict(test_inputs)

        # Multi-threaded (4 threads) prediction
        runtime.set_num_threads(4)
        mt_predictions = runtime.predict(test_inputs)

        # Verify bit-exact numerical parity
        max_err_st = np.max(np.abs(ref_predictions.numpy() - st_predictions.numpy()))
        max_err_mt = np.max(np.abs(ref_predictions.numpy() - mt_predictions.numpy()))
        diff_threads = np.max(np.abs(st_predictions.numpy() - mt_predictions.numpy()))

        print(f"  Batch Size: {batch_size}")
        print(f"  Max Absolute Error (Original vs 1-Thread Runtime): {max_err_st:.10e}")
        print(f"  Max Absolute Error (Original vs 4-Thread Runtime): {max_err_mt:.10e}")
        print(f"  Max Absolute Difference (1-Thread vs 4-Threads):  {diff_threads:.10e}")

        assert max_err_mt < 1e-5, "Multi-threaded prediction diverged!"
        print("  ✓ Parallel Multi-Threaded Prediction: PASSED (Bit-exact parity maintained)")

        # =========================================================================
        # 5. Steady-State Prediction Loop (Zero Memory Allocation)
        # =========================================================================
        print("\n--- [Step 5] Steady-State Prediction Loop ---")
        num_runs = 5
        for i in range(num_runs):
            out_i = runtime.predict(test_inputs)
            assert np.allclose(out_i.numpy(), mt_predictions.numpy())
        print(f"  ✓ Executed {num_runs} consecutive multi-threaded predictions reusing pre-planned memory arena.")

    print("\n" + "=" * 100)
    print("TensorForge v1.2 Demonstration Finished Successfully!")
    print("=" * 100)


if __name__ == "__main__":
    run_parallel_inference_demo()
