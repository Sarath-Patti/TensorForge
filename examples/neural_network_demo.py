"""Neural Network demonstration for TensorForge v0.3.

Demonstrates:
1. Constructing a multi-layer perceptron using Sequential, Linear, and ReLU.
2. Inspecting model architecture and parameters.
3. Performing a forward pass with input data.
4. Calculating multi-class classification loss using CrossEntropyLoss.
5. Computing analytical gradients via loss.backward().
6. Inspecting gradients across all layer parameters.
7. Clearing gradients with model.zero_grad().
"""

import tensorforge as tf
from tensorforge.nn import CrossEntropyLoss, Linear, ReLU, Sequential


def main() -> None:
    print("=== TensorForge v0.3 - Neural Network Modules Demo ===")
    print(f"TensorForge Version: {tf.__version__}\n")

    # 1. Build Model Architecture
    print("1. Creating Neural Network Model (Sequential)")
    model = Sequential(
        Linear(in_features=4, out_features=8, bias=True),
        ReLU(),
        Linear(in_features=8, out_features=3, bias=True),
    )
    print("Model Architecture:")
    print(model)
    print()

    # 2. Inspect Model Parameters
    print("2. Inspecting Named Parameters:")
    for name, param in model.named_parameters():
        print(f"  - {name:12s} shape={str(param.shape):12s} dtype={param.dtype} requires_grad={param.requires_grad}")
    print()

    # 3. Create Synthetic Input Data and Targets
    print("3. Creating Input Batch (3 samples, 4 features) and Targets:")
    x = tf.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [1.0, 0.5, -0.2, 0.8],
            [-0.5, 1.2, 0.0, -0.3],
        ],
        dtype=tf.float32,
    )
    targets = [0, 2, 1]  # Target class indices
    print(f"Input X shape: {x.shape}")
    print(f"Target classes: {targets}\n")

    # 4. Forward Pass
    print("4. Executing Forward Pass:")
    logits = model(x)
    print(f"Output Logits (shape={logits.shape}):\n{logits}\n")

    # 5. Compute Loss
    print("5. Computing Multiclass Cross-Entropy Loss:")
    criterion = CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, targets)
    print(f"Cross-Entropy Loss: {loss.item():.4f}\n")

    # 6. Backward Pass
    print("6. Executing Backpropagation (loss.backward()):")
    loss.backward()
    print("Computed Parameter Gradients:")
    for name, param in model.named_parameters():
        print(f"  - {name} grad (shape={param.grad.shape}):\n{param.grad}\n")

    # 7. Resetting Gradients
    print("7. Clearing Gradients with model.zero_grad():")
    model.zero_grad()
    all_cleared = all(p.grad is None for p in model.parameters())
    print(f"All parameter gradients reset to None: {all_cleared}")


if __name__ == "__main__":
    main()
