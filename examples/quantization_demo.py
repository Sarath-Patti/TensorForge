"""TensorForge v0.7 End-to-End Quantization and INT8 Inference Demonstration.

Demonstrates:
  1. Defining an FP32 Neural Network
  2. Range calibration on representative data
  3. Post-training INT8 quantization of parameters
  4. INT8 Quantized Inference execution
  5. Error metrics and memory savings reporting
"""

import time
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
from tensorforge.quantization import (
    MinMaxCalibrator,
    compare_tensors,
    dequantize,
    qmatmul,
    quantize,
)


def run_quantization_demo():
    print("=" * 75)
    print("TensorForge v0.7 – Post-Training INT8 Quantization Demonstration")
    print("=" * 75)

    np.random.seed(42)

    # 1. Define FP32 Model
    print("\n[Step 1] Constructing FP32 Neural Network...")
    in_features = 16
    hidden_features = 32
    out_features = 4

    model = nn.Sequential(
        nn.Linear(in_features, hidden_features, bias=True),
        nn.ReLU(),
        nn.Linear(hidden_features, out_features, bias=True),
    )
    model.eval()

    # Calculate FP32 parameter memory
    fp32_param_bytes = sum(p.nbytes for p in model.parameters())
    print(f"  Model Architecture: Linear({in_features}->{hidden_features}) -> ReLU -> Linear({hidden_features}->{out_features})")
    print(f"  FP32 Parameter Memory: {fp32_param_bytes:,d} bytes ({fp32_param_bytes / 1024.0:.2f} KB)")

    # 2. Calibration on Representative Dataset
    print("\n[Step 2] Calibrating Dynamic Range on Representative Data...")
    num_samples = 64
    calibration_data = tf.randn((num_samples, in_features))

    calibrator = MinMaxCalibrator()
    calibrator.update(calibration_data)
    min_val, max_val = calibrator.compute_range()
    print(f"  Observed Calibration Range: [{min_val:.4f}, {max_val:.4f}]")

    # 3. Quantize Model Weights to INT8
    print("\n[Step 3] Quantizing Model Weights to INT8 (Symmetric)...")
    q_weights = []
    int8_param_bytes = 0

    for name, param in model.named_parameters():
        q_p = quantize(param, scheme="symmetric")
        q_weights.append((name, q_p))
        int8_param_bytes += q_p.nbytes
        print(f"  - Quantized '{name}': shape={q_p.shape}, scale={q_p.scale:.6f}, memory={q_p.nbytes}B (vs FP32 {param.nbytes}B)")

    compression_ratio = fp32_param_bytes / int8_param_bytes
    print(f"  Total INT8 Parameter Memory: {int8_param_bytes:,d} bytes ({int8_param_bytes / 1024.0:.2f} KB)")
    print(f"  Memory Reduction: {compression_ratio:.2f}x ({(1.0 - 1.0/compression_ratio)*100:.1f}% savings)")

    # 4. Run FP32 vs INT8 Inference
    print("\n[Step 4] Running Inference Comparison (Batch Size = 8)...")
    batch_size = 8
    test_input = tf.randn((batch_size, in_features))

    # A. FP32 Reference Inference
    start_fp = time.perf_counter()
    with tf.no_grad():
        fp32_output = model(test_input)
    fp32_latency_us = (time.perf_counter() - start_fp) * 1e6

    # B. INT8 Quantized Inference Pipeline
    start_int8 = time.perf_counter()
    with tf.no_grad():
        # Layer 1: Quantized Matmul
        w1_t_q = quantize(model[0].weight.transpose(), scheme="symmetric")
        b1 = model[0].bias

        x_q = quantize(test_input, scheme="symmetric")
        h1 = qmatmul(x_q, w1_t_q) + b1
        h1_act = tf.relu(h1)

        # Layer 2: Quantized Matmul
        w2_t_q = quantize(model[2].weight.transpose(), scheme="symmetric")
        b2 = model[2].bias

        h1_q = quantize(h1_act, scheme="symmetric")
        int8_output = qmatmul(h1_q, w2_t_q) + b2
    int8_latency_us = (time.perf_counter() - start_int8) * 1e6

    # 5. Measure Accuracy & Error Metrics
    print("\n[Step 5] Quantization Accuracy & Fidelity Analysis:")
    metrics = compare_tensors(fp32_output, int8_output)

    print(f"  - Maximum Absolute Error: {metrics['max_abs_error']:.6f}")
    print(f"  - Mean Absolute Error (MAE): {metrics['mean_abs_error']:.6f}")
    print(f"  - Mean Squared Error (MSE):  {metrics['mean_sq_error']:.6e}")
    print(f"  - Normalized Relative Error: {metrics['rel_error']:.6f}")
    print(f"  - Signal-to-Noise Ratio:     {metrics['sqnr_db']:.2f} dB")
    print(f"  - FP32 Latency:              {fp32_latency_us:.2f} µs")
    print(f"  - INT8 Latency:              {int8_latency_us:.2f} µs")

    print("\n[Output Comparison Sample (First 2 Predictions)]:")
    print("  FP32 Output:\n ", fp32_output.numpy()[:2])
    print("  INT8 Output:\n ", int8_output.numpy()[:2])
    print("=" * 75)


if __name__ == "__main__":
    run_quantization_demo()
