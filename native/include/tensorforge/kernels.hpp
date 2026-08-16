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

// ============================================================================
// TensorForge v1.0: Fused Inference Operators
// ============================================================================

/**
 * @brief Fused Linear forward: out = x @ weight.T + bias.
 * 
 * @param x Input tensor of shape (M, K).
 * @param weight Weight matrix of shape (N, K).
 * @param bias Optional bias vector of shape (N,) or nullptr.
 * @param out Output tensor of shape (M, N).
 */
void fused_linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
);

/**
 * @brief Fused Linear + ReLU forward: out = max(0, x @ weight.T + bias).
 */
void fused_linear_relu(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
);

/**
 * @brief Fused Linear + Sigmoid forward: out = 1 / (1 + exp(-(x @ weight.T + bias))).
 */
void fused_linear_sigmoid(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
);

/**
 * @brief Fused Linear + Tanh forward: out = tanh(x @ weight.T + bias).
 */
void fused_linear_tanh(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
);

/**
 * @brief Fused Linear + Softmax forward: out = softmax(x @ weight.T + bias, dim=-1).
 */
void fused_linear_softmax(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out,
    int64_t dim = -1
);

/**
 * @brief Fused Quantized INT8 Linear + ReLU forward with Float32 output:
 *   out = max(0, qmatmul(x_q, weight_q.T) + bias).
 */
void fused_qlinear_relu_int8(
    const Tensor& x_q,
    const Tensor& weight_q,
    const Tensor* bias,
    Tensor& out,
    float scale_x,
    int32_t zp_x,
    float scale_w,
    int32_t zp_w
);

} // namespace kernels
} // namespace tensorforge
