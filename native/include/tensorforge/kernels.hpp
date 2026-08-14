#pragma once

#include "tensorforge/tensor.hpp"

namespace tensorforge {
namespace kernels {

/**
 * @brief Element-wise addition of two float32 tensors: out = a + b.
 * 
 * @param a First input tensor.
 * @param b Second input tensor.
 * @param out Output tensor (must have same numel and shape as inputs).
 */
void add(const Tensor& a, const Tensor& b, Tensor& out);

/**
 * @brief Element-wise subtraction of two float32 tensors: out = a - b.
 */
void sub(const Tensor& a, const Tensor& b, Tensor& out);

/**
 * @brief Element-wise multiplication of two float32 tensors: out = a * b.
 */
void mul(const Tensor& a, const Tensor& b, Tensor& out);

/**
 * @brief Element-wise scalar addition: out = a + scalar.
 */
void add_scalar(const Tensor& a, float scalar, Tensor& out);

/**
 * @brief Element-wise scalar multiplication: out = a * scalar.
 */
void mul_scalar(const Tensor& a, float scalar, Tensor& out);

/**
 * @brief Matrix multiplication of two 2D float32 matrices: out = a @ b.
 * 
 * Implements a cache-aware blocked/ordered matrix multiplication:
 *   a: (M, K)
 *   b: (K, N)
 *   out: (M, N)
 * 
 * Loop order: (i, k, j) or tiled block algorithm ensuring optimal sequential access
 * on the inner loop over b and out.
 * 
 * @param a Input matrix (M, K).
 * @param b Input matrix (K, N).
 * @param out Output matrix (M, N).
 */
void matmul(const Tensor& a, const Tensor& b, Tensor& out);

} // namespace kernels
} // namespace tensorforge
