#pragma once

#include <cstdint>
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

/**
 * @brief Quantized INT8 matrix multiplication producing float32 dequantized output:
 *   out(i, j) = scale_a * scale_b * sum_k((a(i, k) - zp_a) * (b(k, j) - zp_b))
 * 
 * @param a Input matrix (M, K) in Int8.
 * @param b Input matrix (K, N) in Int8.
 * @param out Output matrix (M, N) in Float32.
 * @param scale_a Quantization scale factor for matrix a.
 * @param zp_a Zero-point offset for matrix a.
 * @param scale_b Quantization scale factor for matrix b.
 * @param zp_b Zero-point offset for matrix b.
 */
void qmatmul_int8(
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    float scale_a,
    int32_t zp_a,
    float scale_b,
    int32_t zp_b
);

/**
 * @brief Dequantize INT8 tensor to Float32: out = (in - zero_point) * scale.
 * 
 * @param in Input Int8 tensor.
 * @param out Output Float32 tensor.
 * @param scale Quantization scale factor.
 * @param zero_point Quantization zero-point offset.
 */
void dequantize_int8(const Tensor& in, Tensor& out, float scale, int32_t zero_point);

/**
 * @brief Quantize Float32 tensor to INT8: out = clamp(round(in / scale) + zero_point, -128, 127).
 * 
 * @param in Input Float32 tensor.
 * @param out Output Int8 tensor.
 * @param scale Quantization scale factor.
 * @param zero_point Quantization zero-point offset.
 */
void quantize_float32(const Tensor& in, Tensor& out, float scale, int32_t zero_point);

} // namespace kernels
} // namespace tensorforge
