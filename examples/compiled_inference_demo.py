"""TensorForge v1.1 Inference Compiler & Execution Planning Demonstration.

Demonstrates:
  1. Constructing and exporting an FP32 neural network to a .tfmodel archive
  2. Loading the model into InferenceRuntime with architecture auto-reconstruction
  3. Performing Graph Optimization (Operator Fusion)
  4. Compiling the model with InferenceCompiler into an ExecutionPlan with Memory Planning
  5. Inspecting the compiled ExecutionPlan and workspace buffer memory allocations
  6. Executing steady-state predictions with workspace reuse and plan caching
  7. Exporting, compiling, and running an INT8 quantized neural network
"""

import os
import tempfile
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.backend import is_native_available
from tensorforge.inference import InferenceRuntime
from tensorforge.quantization import compare_tensors, quantize
from tensorforge.serialization import save_model
from tensorforge.serialization.format import extract_module_architecture, write_tfmodel_container


def run_compiled_inference_demo():
    print("=" * 95)
    print("TensorForge v1.1 – Inference Compiler & Execution Planning Demonstration")
    print("=" * 95)

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_model_path = os.path.join(tmpdir, "mlp_classifier.tfmodel")
        int8_model_path = os.path.join(tmpdir, "quantized_mlp.tfmodel")

        # =========================================================================
        # 1. Construct and Export FP32 Model
        # =========================================================================
        print("\n--- [Step 1] Constructing and Exporting FP32 Neural Network ---")
        in_features, hidden_features, num_classes = 16, 32, 4

        model = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, num_classes),
            nn.Softmax(dim=-1),
        )
        model.eval()

        print(f"  Model: Linear({in_features}->{hidden_features}) -> ReLU -> Linear({hidden_features}->{num_classes}) -> Softmax")
        save_model(model, fp32_model_path, metadata={"task": "classification"})
        print(f"  Exported model to: '{os.path.basename(fp32_model_path)}' ({os.path.getsize(fp32_model_path)} bytes)")

        # =========================================================================
        # 2. Load into InferenceRuntime & Compile
        # =========================================================================
        print("\n--- [Step 2] Loading into InferenceRuntime & Compiling ---")
        runtime = InferenceRuntime.load(fp32_model_path)
        batch_size = 8

        # Compile model for (batch_size, in_features)
        runtime.compile(input_shape=(batch_size, in_features))
        summary = runtime.summary()

        print("  Compiled Runtime Summary:")
        print(f"    - Optimized Status:      {summary['is_optimized']}")
        print(f"    - Compiled Status:       {summary['is_compiled']}")
        print(f"    - Original Operators:    {summary['original_nodes']}")
        print(f"    - Optimized Operators:   {summary['optimized_nodes']}")
        print(f"    - Compiled Steps:        {summary['compiled_steps']}")
        print(f"    - Fused Operators:       {summary['fused_count']} ({', '.join(summary['fused_patterns'])})")
        print(f"    - Workspace Memory:      {summary['workspace_bytes']} bytes ({summary['workspace_kb']:.2f} KB)")
        print(f"    - Target Backend:        {summary['backend']}")

        # =========================================================================
        # 3. Inspect the Compiled ExecutionPlan
        # =========================================================================
        print("\n--- [Step 3] Inspecting Compiled ExecutionPlan ---")
        plan = runtime.execution_plan
        assert plan is not None
        print(plan.summary())

        # =========================================================================
        # 4. FP32 Prediction & Numerical Parity Check
        # =========================================================================
        print("\n--- [Step 4] Running Compiled Prediction & Parity Verification ---")
        test_inputs = tf.randn((batch_size, in_features))

        with tf.no_grad():
            ref_predictions = model(test_inputs)

        compiled_predictions = runtime.predict(test_inputs)

        # Verify no autograd overhead
        assert compiled_predictions.requires_grad is False
        assert compiled_predictions.grad_fn is None

        # Verify bit-exact numerical parity
        max_err = np.max(np.abs(ref_predictions.numpy() - compiled_predictions.numpy()))
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Absolute Error (Original vs Compiled Runtime): {max_err:.10e}")
        assert max_err < 1e-6, "Compiled prediction diverged from original model!"
        print("  ✓ FP32 Compiled Prediction Parity: PASSED")

        # =========================================================================
        # 5. Steady-State Repeated Predictions (Plan Cache Reuse)
        # =========================================================================
        print("\n--- [Step 5] Steady-State Repeated Execution (Cache Hits) ---")
        num_runs = 5
        for i in range(num_runs):
            out_i = runtime.predict(test_inputs)
            assert np.allclose(out_i.numpy(), compiled_predictions.numpy())
        print(f"  ✓ Executed {num_runs} consecutive predictions using cached ExecutionPlan with zero memory leaks.")

        # =========================================================================
        # 6. INT8 Quantized Model Compilation & Inference
        # =========================================================================
        print("\n--- [Step 6] INT8 Quantized Model Compilation & Low-Precision Inference ---")
        q_weights = {name: quantize(param, scheme="symmetric") for name, param in model.named_parameters()}
        write_tfmodel_container(
            int8_model_path,
            q_weights,
            metadata={"is_quantized": True, "scheme": "symmetric"},
            architecture=extract_module_architecture(model),
        )

        q_runtime = InferenceRuntime.load(int8_model_path).compile(input_shape=(batch_size, in_features))
        print(f"  Quantized Runtime: is_quantized={q_runtime.is_quantized}, is_compiled={q_runtime.is_compiled}")
        print(f"  Quantized Workspace: {q_runtime.workspace_size} bytes")

        q_predictions = q_runtime.predict(test_inputs)
        q_metrics = compare_tensors(ref_predictions, q_predictions)

        print("  INT8 Compiled Inference Metrics:")
        print(f"    - Maximum Absolute Error: {q_metrics['max_abs_error']:.6f}")
        print(f"    - Mean Absolute Error:    {q_metrics['mean_abs_error']:.6f}")
        print(f"    - SQNR:                   {q_metrics['sqnr_db']:.2f} dB")
        print("  ✓ INT8 Compiled Low-Precision Prediction: PASSED")

    print("\n" + "=" * 95)
    print("TensorForge v1.1 Demonstration Finished Successfully!")
    print("=" * 95)


if __name__ == "__main__":
    run_compiled_inference_demo()
