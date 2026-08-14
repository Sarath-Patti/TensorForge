"""Automatic Differentiation demonstration for TensorForge v0.2.

Demonstrates:
1. Creating leaf tensors with requires_grad=True.
2. Building a dynamic computation graph through forward mathematical operations.
3. Inspecting graph nodes (grad_fn) on intermediate non-leaf tensors.
4. Executing reverse-mode automatic differentiation with loss.backward().
5. Inspecting accumulated leaf gradients.
6. A simple manual gradient descent parameter update step using zero_grad() and detach().
"""

import tensorforge as tf


def main() -> None:
    print("=== TensorForge v0.2 - Automatic Differentiation Engine ===")
    print(f"TensorForge Version: {tf.__version__}\n")

    # 1. Leaf Tensor Creation
    print("1. Creating Leaf Tensors with requires_grad=True")
    x = tf.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32, requires_grad=False)
    w = tf.tensor([[0.5, -1.0], [2.0, 1.5]], dtype=tf.float32, requires_grad=True)
    b = tf.tensor([0.1, 0.2], dtype=tf.float32, requires_grad=True)

    print(f"Input X (is_leaf={x.is_leaf}, requires_grad={x.requires_grad}):\n  {x}")
    print(f"Weights W (is_leaf={w.is_leaf}, requires_grad={w.requires_grad}):\n  {w}")
    print(f"Bias B (is_leaf={b.is_leaf}, requires_grad={b.requires_grad}):\n  {b}\n")

    # 2. Forward Pass: Building the DAG
    print("2. Forward Pass & Dynamic Graph Construction")
    # Linear projection: h = X @ W + b
    proj = x @ w           # MatmulBackward
    h = proj + b           # AddBackward
    loss = (h * h).mean()  # MulBackward -> MeanBackward

    print(f"Projection (grad_fn={proj.grad_fn}, is_leaf={proj.is_leaf}):\n  {proj}")
    print(f"Hidden h (grad_fn={h.grad_fn}, is_leaf={h.is_leaf}):\n  {h}")
    print(f"Loss (grad_fn={loss.grad_fn}, is_leaf={loss.is_leaf}):\n  {loss}\n")

    # 3. Backward Pass
    print("3. Executing Reverse-Mode Autograd (loss.backward())")
    loss.backward()

    print("Computed analytical gradients on leaf tensors:")
    print(f"dL/dW (w.grad):\n  {w.grad}")
    print(f"dL/dB (b.grad):\n  {b.grad}")
    print(f"X.grad (requires_grad=False): {x.grad}\n")

    # 4. Manual Optimization Step
    print("4. Manual Gradient Descent Step (w = w - lr * grad)")
    lr = 0.01
    # Update weights in detached context
    with tf.no_grad():
        w_updated = w - w.grad * lr
        b_updated = b - b.grad * lr

    print(f"Updated W:\n  {w_updated}")
    print(f"Updated B:\n  {b_updated}\n")

    # 5. Gradient Reset
    print("5. Resetting Gradients with zero_grad()")
    w.zero_grad()
    b.zero_grad()
    print(f"w.grad after zero_grad(): {w.grad}")
    print(f"b.grad after zero_grad(): {b.grad}")


if __name__ == "__main__":
    main()
