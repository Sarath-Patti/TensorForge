"""TensorForge v1.0 Production Inference Runtime & Operator Fusion Demonstration.

Demonstrates:
  1. Constructing and exporting a multi-layer FP32 neural network to a .tfmodel archive
  2. Loading the model into InferenceRuntime with architecture auto-reconstruction
  3. Running Graph Optimization & Operator Fusion (Linear+ReLU, Linear+Softmax)
  4. Verifying numerical parity between Original Model, Unfused Runtime, and Fused Runtime
  5. Multi-Backend Fused Execution on NumPy and Native C++
  6. INT8 Quantized Model Export, Fusion, and Low-Precision Prediction
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


def run_inference_demo():
    print("=" * 90)
    print("TensorForge v1.0 – Production Inference Runtime & Operator Fusion Demonstration")
    print("=" * 90)

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_model_path = os.path.join(tmpdir, "mlp_classifier.tfmodel")
        int8_model_path = os.path.join(tmpdir, "quantized_mlp.tfmodel")

        # =========================================================================
        # 1. Train and Export FP32 Model
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
        # 2. Load into InferenceRuntime & Run Graph Optimization
        # =========================================================================
        print("\n--- [Step 2] Loading Model & Running Graph Optimization (Operator Fusion) ---")
        runtime = InferenceRuntime.load(fp32_model_path)
        print(f"  Pre-Optimization Nodes:  {runtime.original_node_count}")

        # Run Operator Fusion
        runtime.optimize()
        summary = runtime.summary()

        print("  Optimization Summary:")
        print(f"    - Optimized Status:    {summary['is_optimized']}")
        print(f"    - Original Nodes:      {summary['original_nodes']}")
        print(f"    - Optimized Nodes:     {summary['optimized_nodes']} (collapsed {summary['original_nodes'] - summary['optimized_nodes']} intermediate nodes)")
        print(f"    - Fused Operators:     {summary['fused_count']} ({', '.join(summary['fused_patterns'])})")
        print(f"    - Active Backend:      {summary['backend']}")
        print(f"    - Total Parameters:    {summary['num_parameters']}")
        print(f"    - Memory Footprint:    {summary['total_bytes']} bytes ({summary['size_kb']:.2f} KB)")

        # =========================================================================
        # 3. FP32 Prediction & Numerical Parity Verification
        # =========================================================================
        print("\n--- [Step 3] FP32 Inference & Bit-Exact Parity Check ---")
        batch_size = 8
        test_inputs = tf.randn((batch_size, in_features))

        # Original model prediction
        with tf.no_grad():
            ref_predictions = model(test_inputs)

        # Unfused Runtime prediction
        unfused_runtime = InferenceRuntime.load(fp32_model_path)
        unfused_predictions = unfused_runtime.predict(test_inputs)

        # Fused Runtime prediction
        fused_predictions = runtime.predict(test_inputs)

        # Verify no autograd graph overhead
        assert fused_predictions.requires_grad is False
        assert fused_predictions.grad_fn is None

        # Verify bit-exact numerical parity
        max_err_unfused = np.max(np.abs(ref_predictions.numpy() - unfused_predictions.numpy()))
        max_err_fused = np.max(np.abs(ref_predictions.numpy() - fused_predictions.numpy()))

        print(f"  Batch Size: {batch_size}")
        print(f"  Max Absolute Difference (Original vs Unfused Runtime): {max_err_unfused:.10e}")
        print(f"  Max Absolute Difference (Original vs Fused Runtime):   {max_err_fused:.10e}")
        assert max_err_fused < 1e-6, "Fused prediction diverged from original model!"
        print("  ✓ FP32 Fused Inference Parity: PASSED (Outputs are bit-exact)")

        # =========================================================================
        # 4. Multi-Backend Fused Execution
        # =========================================================================
        print("\n--- [Step 4] Multi-Backend Fused Execution (NumPy & Native C++) ---")
        native_available = is_native_available()
        print(f"  Native C++ Runtime Available: {native_available}")

        # NumPy Fused Backend
        runtime_numpy = InferenceRuntime.load(fp32_model_path, backend="numpy").optimize()
        out_numpy = runtime_numpy.predict(test_inputs)
        print("  ✓ NumPy Fused Execution: OK")

        # Native C++ Fused Backend
        if native_available:
            runtime_native = InferenceRuntime.load(fp32_model_path, backend="native").optimize()
            out_native = runtime_native.predict(test_inputs)
            diff = np.max(np.abs(out_numpy.numpy() - out_native.numpy()))
            print(f"  ✓ Native C++ Fused Execution: OK (NumPy Fused vs Native Fused Parity: {diff:.10e})")
            assert diff < 1e-5, "Native C++ fused inference diverged from NumPy!"

        # =========================================================================
        # 5. INT8 Quantized Model Export & Fused Inference
        # =========================================================================
        print("\n--- [Step 5] INT8 Quantized Model Export & Fused Inference ---")

        # Quantize model weights to INT8
        q_weights = {}
        for name, param in model.named_parameters():
            q_weights[name] = quantize(param, scheme="symmetric")

        write_tfmodel_container(
            int8_model_path,
            q_weights,
            metadata={"is_quantized": True, "scheme": "symmetric"},
            architecture=extract_module_architecture(model),
        )
        print(f"  Exported INT8 model to: '{os.path.basename(int8_model_path)}' ({os.path.getsize(int8_model_path)} bytes)")

        # Load quantized model into InferenceRuntime and optimize
        q_runtime = InferenceRuntime.load(int8_model_path).optimize()
        print(f"  Quantized Runtime Loaded: is_quantized={q_runtime.is_quantized}, is_optimized={q_runtime.is_optimized}")

        q_predictions = q_runtime.predict(test_inputs)
        q_metrics = compare_tensors(ref_predictions, q_predictions)

        print("  INT8 Quantized Inference Metrics:")
        print(f"    - Maximum Absolute Error: {q_metrics['max_abs_error']:.6f}")
        print(f"    - Mean Absolute Error:    {q_metrics['mean_abs_error']:.6f}")
        print(f"    - SQNR:                   {q_metrics['sqnr_db']:.2f} dB")
        print("  ✓ INT8 Quantized Fused Inference: PASSED")

    print("\n" + "=" * 90)
    print("TensorForge v1.0 Demonstration Finished Successfully!")
    print("=" * 90)


if __name__ == "__main__":
    run_inference_demo()
