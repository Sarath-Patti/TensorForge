#include "tensorforge/kernels.hpp"
#include "tensorforge/thread_pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

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

void validate_fused_linear_operands(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    const Tensor& out,
    const char* op_name
) {
    if (x.ndim() != 2 || weight.ndim() != 2 || out.ndim() != 2) {
        throw std::invalid_argument(std::string(op_name) + " requires 2D matrices for input, weight, and output");
    }
    if (x.dtype() != DType::Float32 || weight.dtype() != DType::Float32 || out.dtype() != DType::Float32) {
        throw std::invalid_argument(std::string(op_name) + " requires Float32 dtype for all operands");
    }

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];
    const int64_t Kw = weight.shape()[1];

    if (K != Kw) {
        throw std::invalid_argument(std::string(op_name) + " feature dimension mismatch: x has " +
                                    std::to_string(K) + " features, but weight expects " + std::to_string(Kw));
    }
    if (out.shape()[0] != M || out.shape()[1] != N) {
        throw std::invalid_argument(std::string(op_name) + " output shape mismatch: expected (" +
                                    std::to_string(M) + ", " + std::to_string(N) + "), got (" +
                                    std::to_string(out.shape()[0]) + ", " + std::to_string(out.shape()[1]) + ")");
    }
    if (bias != nullptr && bias->numel() > 0) {
        if (bias->dtype() != DType::Float32) {
            throw std::invalid_argument(std::string(op_name) + " bias must have Float32 dtype");
        }
        if (bias->numel() != N) {
            throw std::invalid_argument(std::string(op_name) + " bias length (" + std::to_string(bias->numel()) +
                                        ") must match out_features (" + std::to_string(N) + ")");
        }
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

void div(const Tensor& a, const Tensor& b, Tensor& out) {
    validate_elementwise_float32(a, b, out, "div");
    const float* a_ptr = a.data_ptr<float>();
    const float* b_ptr = b.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] / b_ptr[i];
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

void relu(const Tensor& a, Tensor& out) {
    validate_scalar_float32(a, out, "relu");
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = a_ptr[i] > 0.0f ? a_ptr[i] : 0.0f;
    }
}

void exp(const Tensor& a, Tensor& out) {
    validate_scalar_float32(a, out, "exp");
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = std::exp(a_ptr[i]);
    }
}

void log(const Tensor& a, Tensor& out) {
    validate_scalar_float32(a, out, "log");
    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = a.numel();

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = std::log(a_ptr[i]);
    }
}

void sum(const Tensor& a, Tensor& out, int64_t dim, bool keepdim) {
    if (a.dtype() != DType::Float32 || out.dtype() != DType::Float32) {
        throw std::invalid_argument("sum currently supports Float32 dtype only");
    }

    const float* a_ptr = a.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    if (dim == -1 && !keepdim && out.numel() == 1) {
        float total = 0.0f;
        const int64_t n = a.numel();
        for (int64_t i = 0; i < n; ++i) {
            total += a_ptr[i];
        }
        out_ptr[0] = total;
        return;
    }

    throw std::invalid_argument("Arbitrary axis sum reduction is not yet implemented in native kernel");
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

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            float* c_row = C + i * N;
            std::memset(c_row, 0, N * sizeof(float));
            const float* a_row = A + i * K;

            for (int64_t k = 0; k < K; ++k) {
                const float a_ik = a_row[k];
                const float* b_row = B + k * N;
                for (int64_t j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void qmatmul_int8(
    const Tensor& a,
    const Tensor& b,
    Tensor& out,
    float scale_a,
    int32_t zp_a,
    float scale_b,
    int32_t zp_b
) {
    if (a.ndim() != 2 || b.ndim() != 2 || out.ndim() != 2) {
        throw std::invalid_argument("qmatmul_int8 requires 2D matrices");
    }
    if (a.dtype() != DType::Int8 || b.dtype() != DType::Int8 || out.dtype() != DType::Float32) {
        throw std::invalid_argument("qmatmul_int8 requires Int8 inputs and Float32 output");
    }

    const int64_t M = a.shape()[0];
    const int64_t K = a.shape()[1];
    const int64_t K2 = b.shape()[0];
    const int64_t N = b.shape()[1];

    if (K != K2) {
        throw std::invalid_argument("qmatmul_int8 inner dimension mismatch: (" + std::to_string(M) + ", " +
                                    std::to_string(K) + ") x (" + std::to_string(K2) + ", " +
                                    std::to_string(N) + ")");
    }
    if (out.shape()[0] != M || out.shape()[1] != N) {
        throw std::invalid_argument("qmatmul_int8 output shape mismatch");
    }

    const int8_t* A = a.data_ptr<int8_t>();
    const int8_t* B = b.data_ptr<int8_t>();
    float* C = out.data_ptr<float>();

    const float combined_scale = scale_a * scale_b;

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        std::vector<int32_t> row_acc(static_cast<size_t>(N), 0);
        for (size_t i = r_start; i < r_end; ++i) {
            std::fill(row_acc.begin(), row_acc.end(), 0);
            const int8_t* a_row = A + i * K;
            float* c_row = C + i * N;

            for (int64_t k = 0; k < K; ++k) {
                const int32_t a_val = static_cast<int32_t>(a_row[k]) - zp_a;
                const int8_t* b_row = B + k * N;
                for (int64_t j = 0; j < N; ++j) {
                    const int32_t b_val = static_cast<int32_t>(b_row[j]) - zp_b;
                    row_acc[static_cast<size_t>(j)] += a_val * b_val;
                }
            }

            for (int64_t j = 0; j < N; ++j) {
                c_row[j] = static_cast<float>(row_acc[static_cast<size_t>(j)]) * combined_scale;
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void dequantize_int8(const Tensor& in, Tensor& out, float scale, int32_t zero_point) {
    if (in.dtype() != DType::Int8 || out.dtype() != DType::Float32) {
        throw std::invalid_argument("dequantize_int8 requires Int8 input and Float32 output");
    }
    if (in.numel() != out.numel()) {
        throw std::invalid_argument("dequantize_int8 numel mismatch");
    }

    const int8_t* in_ptr = in.data_ptr<int8_t>();
    float* out_ptr = out.data_ptr<float>();
    const int64_t n = in.numel();
    const float zp_f = static_cast<float>(zero_point);

    for (int64_t i = 0; i < n; ++i) {
        out_ptr[i] = (static_cast<float>(in_ptr[i]) - zp_f) * scale;
    }
}

void quantize_float32(const Tensor& in, Tensor& out, float scale, int32_t zero_point) {
    if (in.dtype() != DType::Float32 || out.dtype() != DType::Int8) {
        throw std::invalid_argument("quantize_float32 requires Float32 input and Int8 output");
    }
    if (in.numel() != out.numel()) {
        throw std::invalid_argument("quantize_float32 numel mismatch");
    }
    if (scale <= 0.0f) {
        throw std::invalid_argument("quantize_float32 scale must be positive");
    }

    const float* in_ptr = in.data_ptr<float>();
    int8_t* out_ptr = out.data_ptr<int8_t>();
    const int64_t n = in.numel();
    const float inv_scale = 1.0f / scale;
    const float zp_f = static_cast<float>(zero_point);

    for (int64_t i = 0; i < n; ++i) {
        float q_val = std::round(in_ptr[i] * inv_scale) + zp_f;
        q_val = std::max(-128.0f, std::min(127.0f, q_val));
        out_ptr[i] = static_cast<int8_t>(q_val);
    }
}

// ============================================================================
// TensorForge: Fused Forward Implementations with Parallel CPU Execution
// ============================================================================

void fused_linear(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
) {
    validate_fused_linear_operands(x, weight, bias, out, "fused_linear");

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];

    const float* X = x.data_ptr<float>();
    const float* W = weight.data_ptr<float>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const float* x_row = X + i * K;
            float* out_row = OUT + i * N;

            for (int64_t j = 0; j < N; ++j) {
                float acc = (B != nullptr) ? B[j] : 0.0f;
                const float* w_row = W + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                out_row[j] = acc;
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void fused_linear_relu(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
) {
    validate_fused_linear_operands(x, weight, bias, out, "fused_linear_relu");

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];

    const float* X = x.data_ptr<float>();
    const float* W = weight.data_ptr<float>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const float* x_row = X + i * K;
            float* out_row = OUT + i * N;

            for (int64_t j = 0; j < N; ++j) {
                float acc = (B != nullptr) ? B[j] : 0.0f;
                const float* w_row = W + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                out_row[j] = std::max(0.0f, acc);
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void fused_linear_sigmoid(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
) {
    validate_fused_linear_operands(x, weight, bias, out, "fused_linear_sigmoid");

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];

    const float* X = x.data_ptr<float>();
    const float* W = weight.data_ptr<float>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const float* x_row = X + i * K;
            float* out_row = OUT + i * N;

            for (int64_t j = 0; j < N; ++j) {
                float acc = (B != nullptr) ? B[j] : 0.0f;
                const float* w_row = W + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                out_row[j] = 1.0f / (1.0f + std::exp(-acc));
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void fused_linear_tanh(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out
) {
    validate_fused_linear_operands(x, weight, bias, out, "fused_linear_tanh");

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];

    const float* X = x.data_ptr<float>();
    const float* W = weight.data_ptr<float>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const float* x_row = X + i * K;
            float* out_row = OUT + i * N;

            for (int64_t j = 0; j < N; ++j) {
                float acc = (B != nullptr) ? B[j] : 0.0f;
                const float* w_row = W + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                out_row[j] = std::tanh(acc);
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void fused_linear_softmax(
    const Tensor& x,
    const Tensor& weight,
    const Tensor* bias,
    Tensor& out,
    int64_t /*dim*/
) {
    validate_fused_linear_operands(x, weight, bias, out, "fused_linear_softmax");

    const int64_t M = x.shape()[0];
    const int64_t K = x.shape()[1];
    const int64_t N = weight.shape()[0];

    const float* X = x.data_ptr<float>();
    const float* W = weight.data_ptr<float>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const float* x_row = X + i * K;
            float* out_row = OUT + i * N;

            // 1. Matrix multiplication + bias accumulation
            float max_val = -1e30f;
            for (int64_t j = 0; j < N; ++j) {
                float acc = (B != nullptr) ? B[j] : 0.0f;
                const float* w_row = W + j * K;
                for (int64_t k = 0; k < K; ++k) {
                    acc += x_row[k] * w_row[k];
                }
                out_row[j] = acc;
                if (acc > max_val) {
                    max_val = acc;
                }
            }

            // 2. Exponentiation with numerical stability shift
            float sum_exp = 0.0f;
            for (int64_t j = 0; j < N; ++j) {
                float exp_val = std::exp(out_row[j] - max_val);
                out_row[j] = exp_val;
                sum_exp += exp_val;
            }

            // 3. Normalization
            const float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 1.0f;
            for (int64_t j = 0; j < N; ++j) {
                out_row[j] *= inv_sum;
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

void fused_qlinear_relu_int8(
    const Tensor& x_q,
    const Tensor& weight_q,
    const Tensor* bias,
    Tensor& out,
    float scale_x,
    int32_t zp_x,
    float scale_w,
    int32_t zp_w
) {
    if (x_q.ndim() != 2 || weight_q.ndim() != 2 || out.ndim() != 2) {
        throw std::invalid_argument("fused_qlinear_relu_int8 requires 2D matrices");
    }
    if (x_q.dtype() != DType::Int8 || weight_q.dtype() != DType::Int8 || out.dtype() != DType::Float32) {
        throw std::invalid_argument("fused_qlinear_relu_int8 requires Int8 inputs and Float32 output");
    }

    const int64_t M = x_q.shape()[0];
    const int64_t K = x_q.shape()[1];
    const int64_t N = weight_q.shape()[0];
    const int64_t Kw = weight_q.shape()[1];

    if (K != Kw) {
        throw std::invalid_argument("fused_qlinear_relu_int8 dimension mismatch: K=" + std::to_string(K) +
                                    ", Kw=" + std::to_string(Kw));
    }
    if (out.shape()[0] != M || out.shape()[1] != N) {
        throw std::invalid_argument("fused_qlinear_relu_int8 output shape mismatch");
    }

    const int8_t* X = x_q.data_ptr<int8_t>();
    const int8_t* W = weight_q.data_ptr<int8_t>();
    const float* B = (bias != nullptr && bias->numel() > 0) ? bias->data_ptr<float>() : nullptr;
    float* OUT = out.data_ptr<float>();

    const float combined_scale = scale_x * scale_w;

    auto compute_row_range = [&](size_t r_start, size_t r_end) {
        for (size_t i = r_start; i < r_end; ++i) {
            const int8_t* x_row = X + i * K;
            float* out_row = OUT + i * N;

            for (int64_t j = 0; j < N; ++j) {
                float b_val = (B != nullptr) ? B[j] : 0.0f;
                const int8_t* w_row = W + j * K;
                int32_t acc_int = 0;

                for (int64_t k = 0; k < K; ++k) {
                    const int32_t x_val = static_cast<int32_t>(x_row[k]) - zp_x;
                    const int32_t w_val = static_cast<int32_t>(w_row[k]) - zp_w;
                    acc_int += x_val * w_val;
                }

                float val = static_cast<float>(acc_int) * combined_scale + b_val;
                out_row[j] = std::max(0.0f, val);
            }
        }
    };

    size_t total_work = static_cast<size_t>(M) * N * K;
    if (total_work >= PARALLEL_WORKLOAD_THRESHOLD && M > 1) {
        get_global_thread_pool().parallel_for(0, static_cast<size_t>(M), compute_row_range);
    } else {
        compute_row_range(0, static_cast<size_t>(M));
    }
}

} // namespace kernels
} // namespace tensorforge
