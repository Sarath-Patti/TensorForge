"""Unit test suite for TensorForge Tensor Core (v0.1).

Covers:
- Tensor creation & factory methods
- Metadata (shape, ndim, numel, strides, itemsize, nbytes)
- Dtype abstraction & promotion
- Indexing & slicing (reads and writes)
- NumPy conversions & interop
- Arithmetic operations (add, sub, mul, truediv, neg)
- Matrix multiplication (2D, 1D, batched)
- NumPy-style broadcasting
- Shape transformations (reshape with -1 inference, transpose, .T)
- Reductions (sum, mean with axes and keepdims)
- Error handling & validation (mismatched shapes, invalid indices, invalid dtypes)
"""

import unittest
import numpy as np

import tensorforge as tf
from tensorforge import (
    DimensionError,
    DTypeError,
    IndexError_,
    ShapeError,
    Tensor,
    arange,
    float16,
    float32,
    float64,
    from_numpy,
    int8,
    int32,
    int64,
    ones,
    randn,
    tensor,
    zeros,
)


class TestTensorCreation(unittest.TestCase):
    """Tests for tensor creation and factory functions."""

    def test_create_from_list_1d(self):
        t = tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.ndim, 1)
        self.assertEqual(t.numel, 3)
        self.assertEqual(t.dtype, float32)
        np.testing.assert_allclose(t.numpy(), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_create_from_nested_list_2d(self):
        t = tensor([[1, 2, 3], [4, 5, 6]], dtype=int32)
        self.assertEqual(t.shape, (2, 3))
        self.assertEqual(t.ndim, 2)
        self.assertEqual(t.numel, 6)
        self.assertEqual(t.dtype, int32)
        self.assertEqual(t.strides, (3, 1))
        self.assertEqual(t.nbytes, 6 * 4)  # 6 elements * 4 bytes
        np.testing.assert_array_equal(t.numpy(), np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32))

    def test_create_scalar_0d(self):
        t = tensor(42.0)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.ndim, 0)
        self.assertEqual(t.numel, 1)
        self.assertEqual(t.item(), 42.0)
        self.assertEqual(t.strides, ())

    def test_create_zeros(self):
        t = zeros((2, 4), dtype=float64)
        self.assertEqual(t.shape, (2, 4))
        self.assertEqual(t.dtype, float64)
        self.assertEqual(t.nbytes, 8 * 8)
        np.testing.assert_array_equal(t.numpy(), np.zeros((2, 4), dtype=np.float64))

    def test_create_ones(self):
        t = ones((3, 2), dtype=int64)
        self.assertEqual(t.shape, (3, 2))
        self.assertEqual(t.dtype, int64)
        np.testing.assert_array_equal(t.numpy(), np.ones((3, 2), dtype=np.int64))

    def test_create_randn(self):
        t = randn(4, 5, dtype=float32)
        self.assertEqual(t.shape, (4, 5))
        self.assertEqual(t.dtype, float32)
        self.assertEqual(t.numpy().shape, (4, 5))

    def test_create_arange(self):
        t = arange(0, 10, 2, dtype=int32)
        self.assertEqual(t.shape, (5,))
        self.assertEqual(t.dtype, int32)
        np.testing.assert_array_equal(t.numpy(), np.array([0, 2, 4, 6, 8], dtype=np.int32))

    def test_from_numpy(self):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        t = from_numpy(arr)
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.dtype, float32)
        np.testing.assert_array_equal(t.numpy(), arr)


class TestTensorMetadata(unittest.TestCase):
    """Tests for tensor metadata, memory footprint, strides, and contiguity invariants."""

    def test_metadata_attributes(self):
        t = zeros((3, 4, 5), dtype=float32)
        self.assertEqual(t.shape, (3, 4, 5))
        self.assertEqual(t.ndim, 3)
        self.assertEqual(t.numel, 60)
        self.assertEqual(t.size, 60)
        self.assertEqual(t.itemsize, 4)
        self.assertEqual(t.nbytes, 240)
        self.assertEqual(t.strides, (20, 5, 1))
        self.assertTrue(t.is_contiguous)

    def test_is_contiguous_semantics(self):
        # 1. Standard created tensors are contiguous
        t1 = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float32)
        self.assertTrue(t1.is_contiguous)
        self.assertEqual(t1.strides, (3, 1))

        # 2. Transposed tensors materialize as contiguous tensors
        t_trans = t1.transpose()
        self.assertTrue(t_trans.is_contiguous)
        self.assertEqual(t_trans.shape, (3, 2))
        self.assertEqual(t_trans.strides, (2, 1))

        t_t = t1.T
        self.assertTrue(t_t.is_contiguous)
        self.assertEqual(t_t.shape, (3, 2))
        self.assertEqual(t_t.strides, (2, 1))

        # 3. Reshaped tensors are contiguous
        t_reshaped = t1.reshape(6)
        self.assertTrue(t_reshaped.is_contiguous)
        self.assertEqual(t_reshaped.strides, (1,))

        # 4. Slices materialize into new contiguous tensors
        t_slice = t1[:, 1:3]
        self.assertTrue(t_slice.is_contiguous)
        self.assertEqual(t_slice.shape, (2, 2))
        self.assertEqual(t_slice.strides, (2, 1))

        # 5. Non-contiguous NumPy arrays normalize into contiguous storage
        non_contig_np = np.arange(12).reshape(3, 4).T  # Fortran-order / non-C-contiguous
        self.assertFalse(non_contig_np.flags.c_contiguous)
        t_from_non_contig = from_numpy(non_contig_np)
        self.assertTrue(t_from_non_contig.is_contiguous)
        self.assertEqual(t_from_non_contig.shape, (4, 3))
        self.assertEqual(t_from_non_contig.strides, (3, 1))
        np.testing.assert_array_equal(t_from_non_contig.numpy(), non_contig_np)

        # 6. Strided sliced NumPy array
        strided_np = np.arange(10)[::2]
        t_strided = from_numpy(strided_np)
        self.assertTrue(t_strided.is_contiguous)
        self.assertEqual(t_strided.shape, (5,))
        self.assertEqual(t_strided.strides, (1,))

    def test_tolist_and_item(self):
        t = tensor([[10, 20], [30, 40]], dtype=int32)
        self.assertEqual(t.tolist(), [[10, 20], [30, 40]])

        single = tensor([99.5])
        self.assertEqual(single.item(), 99.5)

    def test_item_error_on_multielement(self):
        t = tensor([1, 2, 3])
        with self.assertRaises(DimensionError):
            t.item()

    def test_string_representation(self):
        t = tensor([1.0, 2.0], dtype=float32)
        repr_str = repr(t)
        str_str = str(t)
        self.assertIn("tensor(", repr_str)
        self.assertIn("float32", repr_str)
        self.assertIn("Tensor(", str_str)
        self.assertIn("shape=(2,)", str_str)


class TestDTypes(unittest.TestCase):
    """Tests for dtype handling, conversions, and promotion."""

    def test_supported_dtypes(self):
        for dt, expected_bytes in [(float32, 4), (float64, 8), (int32, 4), (int64, 8), (float16, 2), (int8, 1)]:
            t = zeros((2,), dtype=dt)
            self.assertEqual(t.dtype, dt)
            self.assertEqual(t.itemsize, expected_bytes)

    def test_int8_storage_and_quantization_semantics(self):
        # int8 is supported as a contiguous low-precision storage type in v0.1
        t = tensor([1, -2, 127], dtype=int8)
        self.assertEqual(t.dtype, int8)
        self.assertTrue(t.dtype.is_quantized)
        self.assertEqual(t.itemsize, 1)
        self.assertEqual(t.nbytes, 3)
        self.assertTrue(t.is_contiguous)
        np.testing.assert_array_equal(t.numpy(), np.array([1, -2, 127], dtype=np.int8))

        # Basic element-wise operations on int8
        t_add = t + t
        self.assertEqual(t_add.dtype, int8)
        np.testing.assert_array_equal(t_add.numpy(), np.array([2, -4, -2], dtype=np.int8))  # 127 + 127 wraps to -2 in int8

    def test_dtype_promotion_rules(self):
        from tensorforge.tensor.dtype import promote_dtypes

        # float32 + float32 -> float32
        self.assertEqual(promote_dtypes(float32, float32), float32)
        t_f32_1 = tensor([1.0], dtype=float32)
        t_f32_2 = tensor([2.0], dtype=float32)
        self.assertEqual((t_f32_1 + t_f32_2).dtype, float32)

        # float32 + int32 -> float32
        self.assertEqual(promote_dtypes(float32, int32), float32)
        self.assertEqual(promote_dtypes(int32, float32), float32)
        t_i32 = tensor([2], dtype=int32)
        self.assertEqual((t_f32_1 + t_i32).dtype, float32)

        # float64 + float32 -> float64
        self.assertEqual(promote_dtypes(float64, float32), float64)
        self.assertEqual(promote_dtypes(float32, float64), float64)
        t_f64 = tensor([3.0], dtype=float64)
        self.assertEqual((t_f64 + t_f32_1).dtype, float64)

        # int32 + int64 -> int64
        self.assertEqual(promote_dtypes(int32, int64), int64)
        self.assertEqual(promote_dtypes(int64, int32), int64)
        t_i64 = tensor([4], dtype=int64)
        self.assertEqual((t_i32 + t_i64).dtype, int64)

        # int8 + int8 -> int8
        self.assertEqual(promote_dtypes(int8, int8), int8)
        t_i8_1 = tensor([5], dtype=int8)
        t_i8_2 = tensor([6], dtype=int8)
        self.assertEqual((t_i8_1 + t_i8_2).dtype, int8)

        # int8 + int32 -> int32
        self.assertEqual(promote_dtypes(int8, int32), int32)
        self.assertEqual((t_i8_1 + t_i32).dtype, int32)

        # int8 + float32 -> float32
        self.assertEqual(promote_dtypes(int8, float32), float32)
        self.assertEqual((t_i8_1 + t_f32_1).dtype, float32)

        # integer + floating point -> floating point
        self.assertEqual(promote_dtypes(int64, float32), float32)
        self.assertEqual(promote_dtypes(int64, float64), float64)
        self.assertEqual((t_i64 + t_f64).dtype, float64)

    def test_dtype_conversion_astype(self):
        t = tensor([1.7, 2.4, 3.9], dtype=float32)
        t_int = t.astype(int32)
        self.assertEqual(t_int.dtype, int32)
        np.testing.assert_array_equal(t_int.numpy(), np.array([1, 2, 3], dtype=np.int32))

    def test_dtype_string_lookup(self):
        t = tensor([1, 2], dtype="float64")
        self.assertEqual(t.dtype, float64)

    def test_invalid_dtype(self):
        with self.assertRaises(DTypeError):
            tensor([1, 2], dtype="invalid_type")


class TestIndexing(unittest.TestCase):
    """Tests for tensor indexing, slicing, and mutation."""

    def test_1d_indexing(self):
        t = tensor([10, 20, 30, 40, 50], dtype=int64)
        self.assertEqual(t[0].item(), 10)
        self.assertEqual(t[-1].item(), 50)
        sliced = t[1:4]
        self.assertEqual(sliced.shape, (3,))
        np.testing.assert_array_equal(sliced.numpy(), np.array([20, 30, 40], dtype=np.int64))

    def test_2d_indexing(self):
        t = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float32)
        self.assertEqual(t[0, 0].item(), 1.0)
        self.assertEqual(t[1, 2].item(), 6.0)
        row = t[1]
        self.assertEqual(row.shape, (3,))
        np.testing.assert_array_equal(row.numpy(), np.array([4.0, 5.0, 6.0], dtype=np.float32))

        col = t[:, 1]
        self.assertEqual(col.shape, (3,))
        np.testing.assert_array_equal(col.numpy(), np.array([2.0, 5.0, 8.0], dtype=np.float32))

    def test_setitem(self):
        t = zeros((3, 3), dtype=float32)
        t[0, 0] = 5.0
        t[1, :] = tensor([1.0, 2.0, 3.0])
        expected = np.array([[5.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        np.testing.assert_array_equal(t.numpy(), expected)

    def test_index_out_of_bounds(self):
        t = tensor([1, 2, 3])
        with self.assertRaises(IndexError_):
            _ = t[10]


class TestArithmeticOperations(unittest.TestCase):
    """Tests for arithmetic operations and broadcasting."""

    def test_addition(self):
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([4.0, 5.0, 6.0])
        c = a + b
        self.assertEqual(c.shape, (3,))
        np.testing.assert_allclose(c.numpy(), [5.0, 7.0, 9.0])

    def test_scalar_addition(self):
        a = tensor([1.0, 2.0, 3.0])
        c1 = a + 10.0
        c2 = 10.0 + a
        np.testing.assert_allclose(c1.numpy(), [11.0, 12.0, 13.0])
        np.testing.assert_allclose(c2.numpy(), [11.0, 12.0, 13.0])

    def test_subtraction(self):
        a = tensor([10.0, 20.0, 30.0])
        b = tensor([1.0, 2.0, 3.0])
        c = a - b
        np.testing.assert_allclose(c.numpy(), [9.0, 18.0, 27.0])

    def test_scalar_subtraction(self):
        a = tensor([10.0, 20.0, 30.0])
        c1 = a - 5.0
        c2 = 50.0 - a
        np.testing.assert_allclose(c1.numpy(), [5.0, 15.0, 25.0])
        np.testing.assert_allclose(c2.numpy(), [40.0, 30.0, 20.0])

    def test_multiplication(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])
        b = tensor([[2.0, 0.5], [1.0, 3.0]])
        c = a * b
        np.testing.assert_allclose(c.numpy(), [[2.0, 1.0], [3.0, 12.0]])

    def test_scalar_multiplication(self):
        a = tensor([2.0, 4.0, 6.0])
        c1 = a * 2.5
        c2 = 2.5 * a
        np.testing.assert_allclose(c1.numpy(), [5.0, 10.0, 15.0])
        np.testing.assert_allclose(c2.numpy(), [5.0, 10.0, 15.0])

    def test_division(self):
        a = tensor([10.0, 20.0, 30.0])
        b = tensor([2.0, 4.0, 5.0])
        c = a / b
        np.testing.assert_allclose(c.numpy(), [5.0, 5.0, 6.0])

    def test_negation(self):
        a = tensor([1.0, -2.0, 3.5])
        c = -a
        np.testing.assert_allclose(c.numpy(), [-1.0, 2.0, -3.5])

    def test_broadcasting_arithmetic(self):
        a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        b = tensor([10.0, 20.0, 30.0])                  # (3,)
        c = a + b
        self.assertEqual(c.shape, (2, 3))
        expected = np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_incompatible_broadcast_error(self):
        a = zeros((2, 3))
        b = zeros((2, 4))
        with self.assertRaises(ShapeError):
            _ = a + b


class TestMatrixMultiplication(unittest.TestCase):
    """Tests for 2D and batched matrix multiplication."""

    def test_2d_matmul(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
        b = tensor([[5.0, 6.0], [7.0, 8.0]])  # (2, 2)
        c = a @ b
        self.assertEqual(c.shape, (2, 2))
        expected = np.array([[19.0, 22.0], [43.0, 50.0]], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_vector_matrix_matmul(self):
        v = tensor([1.0, 2.0])               # (2,)
        m = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        c = v @ m
        self.assertEqual(c.shape, (3,))
        expected = np.array([9.0, 12.0, 15.0], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_matrix_vector_matmul(self):
        m = tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # (3, 2)
        v = tensor([2.0, 3.0])                            # (2,)
        c = m @ v
        self.assertEqual(c.shape, (3,))
        expected = np.array([8.0, 18.0, 28.0], dtype=np.float32)
        np.testing.assert_allclose(c.numpy(), expected)

    def test_batched_matmul(self):
        a = randn(2, 3, 4)  # (2, 3, 4)
        b = randn(2, 4, 5)  # (2, 4, 5)
        c = a @ b
        self.assertEqual(c.shape, (2, 3, 5))
        np_expected = np.matmul(a.numpy(), b.numpy())
        np.testing.assert_allclose(c.numpy(), np_expected, rtol=1e-5)

    def test_incompatible_matmul_error(self):
        a = zeros((2, 3))
        b = zeros((4, 5))
        with self.assertRaises(DimensionError):
            _ = a @ b


class TestTransformationsAndReductions(unittest.TestCase):
    """Tests for reshape, transpose, sum, and mean operations."""

    def test_reshape(self):
        t = arange(12).astype(float32)
        r1 = t.reshape(3, 4)
        self.assertEqual(r1.shape, (3, 4))
        self.assertEqual(r1.strides, (4, 1))

        r2 = t.reshape(2, -1, 3)
        self.assertEqual(r2.shape, (2, 2, 3))

    def test_invalid_reshape_error(self):
        t = arange(10)
        with self.assertRaises(ShapeError):
            t.reshape(3, 4)  # 10 != 12

    def test_transpose_2d(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2, 3)
        t_trans = t.transpose()
        self.assertEqual(t_trans.shape, (3, 2))
        np.testing.assert_allclose(t_trans.numpy(), np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]))

        # Test .T property
        t_t = t.T
        self.assertEqual(t_t.shape, (3, 2))
        np.testing.assert_allclose(t_t.numpy(), t_trans.numpy())

    def test_transpose_nd(self):
        t = randn(2, 3, 4)
        t_perm = t.transpose(1, 2, 0)
        self.assertEqual(t_perm.shape, (3, 4, 2))
        np.testing.assert_allclose(t_perm.numpy(), np.transpose(t.numpy(), (1, 2, 0)))

    def test_invalid_transpose_error(self):
        t = randn(2, 3)
        with self.assertRaises(DimensionError):
            t.transpose(0, 0)  # Duplicate axes

    def test_sum_all(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        s = t.sum()
        self.assertEqual(s.shape, ())
        self.assertEqual(s.item(), 10.0)

    def test_sum_axis(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        s0 = t.sum(axis=0)
        s1 = t.sum(axis=1, keepdims=True)
        self.assertEqual(s0.shape, (2,))
        np.testing.assert_allclose(s0.numpy(), [4.0, 6.0])
        self.assertEqual(s1.shape, (2, 1))
        np.testing.assert_allclose(s1.numpy(), [[3.0], [7.0]])

    def test_mean(self):
        t = tensor([[2.0, 4.0], [6.0, 8.0]])
        m = t.mean()
        self.assertEqual(m.shape, ())
        self.assertEqual(m.item(), 5.0)

        m0 = t.mean(axis=0)
        np.testing.assert_allclose(m0.numpy(), [4.0, 6.0])


if __name__ == "__main__":
    unittest.main()
