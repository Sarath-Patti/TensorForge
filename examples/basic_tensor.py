"""Basic Tensor usage example for TensorForge v0.1.

Demonstrates:
1. Creating tensors from Python collections, factory functions, and NumPy arrays.
2. Inspecting tensor metadata (shape, ndim, dtype, numel, strides, nbytes).
3. Arithmetic operations and broadcasting.
4. Matrix multiplication.
5. Reshaping, transposing, and slicing.
6. Reductions (sum, mean).
"""

import tensorforge as tf


def main() -> None:
    print("=== TensorForge v0.1 - Project Foundation & Tensor Core ===")
    print(f"TensorForge Version: {tf.__version__}\n")

    # 1. Tensor Creation
    print("1. Tensor Creation")
    a = tf.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
    b = tf.ones((2, 3), dtype=tf.float32)
    c = tf.arange(1, 7, dtype=tf.float32).reshape(2, 3)

    print(f"Tensor A:\n  {a}")
    print(f"Tensor B (ones):\n  {b}")
    print(f"Tensor C (arange reshaped):\n  {c}\n")

    # 2. Inspecting Metadata
    print("2. Inspecting Tensor Metadata")
    print(f"Shape:         {a.shape}")
    print(f"Dimensions:    {a.ndim}")
    print(f"Data Type:     {a.dtype}")
    print(f"Total Elements:{a.numel}")
    print(f"Strides:       {a.strides} elements")
    print(f"Memory Usage:  {a.nbytes} bytes ({a.itemsize} bytes/element)")
    print(f"Is Contiguous: {a.is_contiguous}\n")

    # 3. Arithmetic Operations & Broadcasting
    print("3. Arithmetic Operations & Broadcasting")
    sum_ab = a + b
    scaled = a * 2.5
    bias = tf.tensor([10.0, 20.0, 30.0], dtype=tf.float32)  # (3,)
    broadcasted = a + bias                                   # (2, 3) + (3,) -> (2, 3)

    print(f"A + B:\n  {sum_ab}")
    print(f"A * 2.5:\n  {scaled}")
    print(f"A + [10, 20, 30] (Broadcasting):\n  {broadcasted}\n")

    # 4. Matrix Multiplication
    print("4. Matrix Multiplication")
    w = tf.tensor([[1.0, 0.5], [0.0, 2.0], [-1.0, 1.5]], dtype=tf.float32)  # (3, 2)
    # (2, 3) @ (3, 2) -> (2, 2)
    out = a @ w
    print(f"A (2x3) @ W (3x2) -> Result (2x2):\n  {out}\n")

    # 5. Transformations & Slicing
    print("5. Reshaping, Transposing & Slicing")
    reshaped = a.reshape(3, 2)
    transposed = a.T  # Transpose via .T property (3, 2)
    row_slice = a[0, :]
    elem = a[1, 2]

    print(f"Reshaped to (3, 2):\n  {reshaped}")
    print(f"Transposed A.T:\n  {transposed}")
    print(f"First row slice (a[0, :]):\n  {row_slice}")
    print(f"Element at [1, 2]: {elem.item()}\n")

    # 6. Reductions
    print("6. Reductions (Sum & Mean)")
    total_sum = a.sum()
    col_sum = a.sum(axis=0)
    row_mean = a.mean(axis=1, keepdims=True)

    print(f"Total Sum:  {total_sum.item()}")
    print(f"Column Sum (axis 0): {col_sum}")
    print(f"Row Mean (axis 1, keepdims=True):\n  {row_mean}\n")

    # 7. NumPy Interoperability
    print("7. NumPy Interoperability")
    np_array = a.numpy()
    print(f"Converted to NumPy array (type {type(np_array).__name__}, shape {np_array.shape}):")
    print(f"  {np_array}")
    reconstructed = tf.from_numpy(np_array)
    print(f"Reconstructed TensorForge Tensor:\n  {reconstructed}")


if __name__ == "__main__":
    main()
