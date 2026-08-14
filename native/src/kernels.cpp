#include "tensorforge/kernels.hpp"

#include <cstring>
#include <stdexcept>

namespace tensorforge {
namespace kernels {

namespace {

void validate_elementwise_float32(const Tensor& a, const Tensor& b, const Tensor& out, const char* op_name) {
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32 || out.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(op_name) + " currently supports Float32 dtype only");
    }
    if (a.numel() != b.numel() || a.numel() != out.numel()) {
        throw std::invalid_argument(std::string(op_name) + " numel mismatch: a=" +
                                    std::to_string(a.numel()) + ", b=" + std::to_string(b.numel()) +
                                    ", out=" + std::to_string(out.numel()));
    }
}

void validate_scalar_float32(const Tensor& a, const Tensor& out, const char* op_name) {
    if (a.dtype() != DType::Float32 || out.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(op_name) + " currently supports Float32 dtype only");
    }
    if (a.numel() != out.numel()) {
        throw std::invalid_argument(std::string(op_name) + " numel mismatch: a=" +
                                    std::to_string(a.numel()) + ", out=" + std::to_string(out.numel()));
    }
}

} // anonymous namespace

void add(const Tensor& a, const Tensor& b, Tensor& out) {
    validate_elementwise_float32(a, b, out, "add");
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + b_ptr[i];
    }
}

void sub(const Tensor& a, const Tensor& b, Tensor& out) {
    validate_elementwise_float32(a, b, out, "sub");
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] - b_ptr[i];
    }
}

void mul(const Tensor& a, const Tensor& b, Tensor& out) {
    validate_elementwise_float32(a, b, out, "mul");
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] * b_ptr[i];
    }
}

void add_scalar(const Tensor& a, float scalar, Tensor& out) {
    validate_scalar_float32(a, out, "add_scalar");
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] + scalar;
    }
}

void mul_scalar(const Tensor& a, float scalar, Tensor& out) {
    validate_scalar_float32(a, out, "mul_scalar");
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] * scalar;
    }
}

void matmul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
        throw std::invalid_argument("matmul requires 2D matrices");
    }
    if (a.dtype() != DType::Float32 || b.dtype() != DType::Float32 || out.dtype() != DType::Float32) {
        throw std::invalid_argument("matmul currently supports Float32 dtype only");
    }

    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];
    const int64_t K2 = b.shape()[0];
    const int64_t N = b.shape()[1];

    if (K != K2) {
        throw std::invalid_argument("matmul inner dimension mismatch: (" + std::to_string(M) + ", " +
                                    std::to_string(K) + ") x (" + std::to_string(K2) + ", " +
                                    std::to_string(N) + ")");
    }
    if (out.shape()[0] != M || out.shape()[1] != N) {
        throw std::invalid_argument("matmul output shape mismatch: expected (" + std::to_string(M) +
                                    ", " + std::to_string(N) + "), got (" +
                                    std::to_string(out.shape()[0]) + ", " +
                                    std::to_string(out.shape()[1]) + ")");
    }

    const float* A = a.data_ptr<float>();
    const float* B = b.data_ptr<float>();
    float* C = out.data_ptr<float>();

    // Zero-initialize output buffer
    std::memset(C, 0, static_cast<size_t>(M * N) * sizeof(float));

    // Cache-aware loop ordering (i, k, j) for optimal spatial locality on rows of B and C
    for (int64_t i = 0; i < M; ++i) {
        const float* a_row = A + i * K;
        float* c_row = C + i * N;
        for (int64_t k = 0; k < K; ++k) {
            const float a_val = a_row[k];
            const float* b_row = B + k * N;
            for (int64_t j = 0; j < N; ++j) {
                c_row[j] += a_val * b_row[j];
            }
        }
    }
}

} // namespace kernels
} // namespace tensorforge
