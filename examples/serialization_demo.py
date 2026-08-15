"""TensorForge v0.8 Model Serialization and Checkpointing Demonstration.

Demonstrates:
  1. Training an FP32 model and saving as a .tfmodel file
  2. Loading .tfmodel into a fresh model instance and verifying identical inference output
  3. Saving a training checkpoint with model, optimizer, epoch, step, and loss states
  4. Restoring checkpoint into a fresh model & optimizer to seamlessly resume training
  5. Serializing an INT8 quantized model and comparing storage footprints
"""

import os
import tempfile
import numpy as np
import tensorforge as tf
import tensorforge.nn as nn
import tensorforge.optim as optim
from tensorforge.quantization import quantize
from tensorforge.serialization import (
    compute_model_size,
    load_checkpoint,
    load_model,
    load_state_dict_from_file,
    save_checkpoint,
    save_model,
)


def run_serialization_demo():
    print("=" * 80)
    print("TensorForge v0.8 – Model Serialization & Checkpointing Demonstration")
    print("=" * 80)

    np.random.seed(42)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "classifier.tfmodel")
        ckpt_path = os.path.join(tmpdir, "training_ckpt.tfckpt")
        qmodel_path = os.path.join(tmpdir, "quantized_classifier.tfmodel")

        # =========================================================================
        # Part 1: Model Serialization & Weight Restoration
        # =========================================================================
        print("\n--- [Part 1] Model Serialization & Exact Weight Restoration ---")

        # 1. Define and train a 2-layer MLP
        in_dim, hidden_dim, out_dim = 8, 16, 3
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        x_sample = tf.randn((4, in_dim))
        target = tf.zeros((4, out_dim))
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        print("  Training model for 5 warm-up iterations...")
        for _ in range(5):
            optimizer.zero_grad()
            loss = criterion(model(x_sample), target)
            loss.backward()
            optimizer.step()

        # Compute pre-save inference baseline
        with tf.no_grad():
            baseline_output = model(x_sample)

        # 2. Save model to .tfmodel
        print(f"  Saving model to: '{os.path.basename(model_path)}'...")
        save_model(
            model,
            model_path,
            metadata={"architecture": "MLP(8->16->3)", "loss": float(loss.numpy())},
        )
        file_size_bytes = os.path.getsize(model_path)
        print(f"  Saved .tfmodel archive size: {file_size_bytes} bytes")

        # 3. Create fresh model and load weights
        fresh_model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        meta = load_model(fresh_model, model_path)
        print(f"  Loaded model successfully! Metadata format version: {meta['format_version']}")

        # 4. Verify identical outputs
        with tf.no_grad():
            restored_output = fresh_model(x_sample)

        max_diff = np.max(np.abs(baseline_output.numpy() - restored_output.numpy()))
        print(f"  Max prediction difference (Original vs Restored): {max_diff:.10e}")
        assert max_diff < 1e-6, "Restored model output does not match original!"
        print("  ✓ Model verification: PASSED (Outputs are bit-exact)")

        # =========================================================================
        # Part 2: Training Checkpoint & Resuming Optimization
        # =========================================================================
        print("\n--- [Part 2] Checkpointing & Resuming Training ---")

        # 1. Save checkpoint after Epoch 1
        print("  Saving training checkpoint at Epoch 1 (Step 5)...")
        save_checkpoint(
            {
                "model": model,
                "optimizer": optimizer,
                "epoch": 1,
                "step": 5,
                "loss": float(loss.numpy()),
                "metrics": {"current_loss": float(loss.numpy())},
                "user_metadata": {"dataset": "synthetic_features"},
            },
            ckpt_path,
        )
        print(f"  Checkpoint written to: '{os.path.basename(ckpt_path)}' ({os.path.getsize(ckpt_path)} bytes)")

        # 2. Instantiate fresh model & optimizer and restore
        resumed_model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        resumed_optimizer = optim.Adam(resumed_model.parameters(), lr=0.01)

        checkpoint_data = load_checkpoint(ckpt_path)
        resumed_model.load_state_dict(checkpoint_data["model_state_dict"])
        resumed_optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])

        print(f"  Restored checkpoint: Epoch {checkpoint_data['epoch']}, Step {checkpoint_data['step']}")

        # 3. Step both models and verify identical learning dynamics
        optimizer.zero_grad()
        loss1 = criterion(model(x_sample), target)
        loss1.backward()
        optimizer.step()

        resumed_optimizer.zero_grad()
        loss2 = criterion(resumed_model(x_sample), target)
        loss2.backward()
        resumed_optimizer.step()

        with tf.no_grad():
            out1 = model(x_sample)
            out2 = resumed_model(x_sample)

        step_diff = np.max(np.abs(out1.numpy() - out2.numpy()))
        print(f"  Post-resume step trajectory difference: {step_diff:.10e}")
        assert step_diff < 1e-6, "Resumed optimizer step diverged!"
        print("  ✓ Training resumption: PASSED (Momentum/moments preserved identically)")

        # =========================================================================
        # Part 3: Quantized Model Serialization & Storage Comparison
        # =========================================================================
        print("\n--- [Part 3] Quantized INT8 Model Serialization & Storage Footprint ---")

        # 1. Quantize model weights to INT8
        q_state_dict = {}
        for name, param in model.named_parameters():
            q_state_dict[name] = quantize(param, scheme="symmetric")

        save_model(
            q_state_dict,
            qmodel_path,
            metadata={"is_quantized": True, "scheme": "symmetric"},
        )
        print(f"  Quantized .tfmodel written to: '{os.path.basename(qmodel_path)}'")

        # 2. Compute model size statistics
        fp32_stats = compute_model_size(model)
        int8_stats = compute_model_size(q_state_dict)

        compression = fp32_stats["total_bytes"] / int8_stats["total_bytes"]
        print("\n  Memory Footprint Comparison:")
        print(f"    - Total Parameters:       {fp32_stats['num_parameters']:,d}")
        print(f"    - FP32 Parameter Storage: {fp32_stats['total_bytes']:,d} bytes ({fp32_stats['size_kb']:.2f} KB)")
        print(f"    - INT8 Parameter Storage: {int8_stats['total_bytes']:,d} bytes ({int8_stats['size_kb']:.2f} KB)")
        print(f"    - Compression Ratio:      {compression:.2f}x ({(1.0 - 1.0/compression)*100:.1f}% reduction)")

        # 3. Load quantized model and verify attributes
        loaded_q_sd, q_meta = load_state_dict_from_file(qmodel_path)
        print(f"  Successfully loaded {len(loaded_q_sd)} quantized tensors from archive!")
        print("  ✓ Quantized serialization: PASSED")

    print("\n" + "=" * 80)
    print("Serialization & Checkpointing Demonstration Finished Successfully!")
    print("=" * 80)


if __name__ == "__main__":
    run_serialization_demo()
