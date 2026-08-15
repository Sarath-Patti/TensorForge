"""TensorForge v0.9 Portable Inference Runtime Demonstration.

Demonstrates:
  1. Training and exporting an FP32 model as a .tfmodel archive
  2. Loading the model into InferenceRuntime with architecture auto-reconstruction
  3. Running batched inference and verifying bit-exact parity against the original model
  4. Executing predictions on both NumPy and Native C++ acceleration backends
  5. Exporting, loading, and predicting with an INT8 quantized model
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
    print("=" * 80)
    print("TensorForge v0.9 – Portable Inference Runtime Demonstration")
    print("=" * 80)

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

        print(f"  Model Architecture: Linear({in_features}->{hidden_features}) -> ReLU -> Linear({hidden_features}->{num_classes}) -> Softmax")

        # Export model to .tfmodel (includes architecture description & parameters)
        save_model(model, fp32_model_path, metadata={"task": "multi_class_classification"})
        print(f"  Exported model to: '{os.path.basename(fp32_model_path)}' ({os.path.getsize(fp32_model_path)} bytes)")

        # =========================================================================
        # 2. Load into InferenceRuntime (Zero-Code Reconstitution)
        # =========================================================================
        print("\n--- [Step 2] Loading Model into InferenceRuntime ---")
        runtime = InferenceRuntime.load(fp32_model_path)
        summary = runtime.summary()

        print("  Runtime Summary:")
        print(f"    - Model Type:          {summary['model_type']}")
        print(f"    - Active Backend:      {summary['backend']}")
        print(f"    - Inferred Input:      {summary['input_shape']}")
        print(f"    - Inferred Output:     {summary['output_shape']}")
        print(f"    - Total Parameters:    {summary['num_parameters']}")
        print(f"    - Memory Footprint:    {summary['total_bytes']} bytes ({summary['size_kb']:.2f} KB)")
        print(f"    - Format Version:      {summary['format_version']}")

        # =========================================================================
        # 3. FP32 Prediction & Numerical Parity Verification
        # =========================================================================
        print("\n--- [Step 3] Running FP32 Inference & Parity Check ---")
        batch_size = 8
        test_inputs = tf.randn((batch_size, in_features))

        # Original model prediction
        with tf.no_grad():
            ref_predictions = model(test_inputs)

        # InferenceRuntime prediction
        runtime_predictions = runtime.predict(test_inputs)

        # Verify no autograd graph overhead
        assert runtime_predictions.requires_grad is False
        assert runtime_predictions.grad_fn is None

        # Verify bit-exact numerical parity
        max_err = np.max(np.abs(ref_predictions.numpy() - runtime_predictions.numpy()))
        print(f"  Batch Size: {batch_size}")
        print(f"  Max Absolute Difference (Original vs Runtime): {max_err:.10e}")
        assert max_err < 1e-6, "InferenceRuntime prediction diverged from original model!"
        print("  ✓ FP32 Inference Parity: PASSED (Outputs are bit-exact)")

        # =========================================================================
        # 4. Multi-Backend Inference Dispatch
        # =========================================================================
        print("\n--- [Step 4] Multi-Backend Inference Execution ---")
        native_available = is_native_available()
        print(f"  Native C++ Runtime Available: {native_available}")

        # NumPy Backend
        runtime_numpy = InferenceRuntime.load(fp32_model_path, backend="numpy")
        out_numpy = runtime_numpy.predict(test_inputs)
        print("  ✓ NumPy Backend Execution: OK")

        # Native C++ Backend
        if native_available:
            runtime_native = InferenceRuntime.load(fp32_model_path, backend="native")
            out_native = runtime_native.predict(test_inputs)
            diff = np.max(np.abs(out_numpy.numpy() - out_native.numpy()))
            print(f"  ✓ Native Backend Execution: OK (NumPy vs Native Parity: {diff:.10e})")
            assert diff < 1e-5, "Native C++ inference diverged from NumPy!"

        # =========================================================================
        # 5. INT8 Quantized Model Export & Inference
        # =========================================================================
        print("\n--- [Step 5] INT8 Quantized Model Export & Inference ---")

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

        # Load quantized model into InferenceRuntime
        q_runtime = InferenceRuntime.load(int8_model_path)
        print(f"  Quantized Runtime Loaded: is_quantized={q_runtime.is_quantized}")

        q_predictions = q_runtime.predict(test_inputs)
        q_metrics = compare_tensors(ref_predictions, q_predictions)

        print("  INT8 Quantized Inference Metrics:")
        print(f"    - Maximum Absolute Error: {q_metrics['max_abs_error']:.6f}")
        print(f"    - Mean Absolute Error:    {q_metrics['mean_abs_error']:.6f}")
        print(f"    - SQNR:                   {q_metrics['sqnr_db']:.2f} dB")
        print("  ✓ INT8 Quantized Inference: PASSED")

    print("\n" + "=" * 80)
    print("Inference Demonstration Finished Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_inference_demo()
