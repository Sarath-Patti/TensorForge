#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "tensorforge/allocator.hpp"
#include "tensorforge/dtype.hpp"
#include "tensorforge/kernels.hpp"
#include "tensorforge/shape.hpp"
#include "tensorforge/storage.hpp"
#include "tensorforge/tensor.hpp"

using namespace tensorforge;

void test_shape() {
    std::cout << "[Test] Shape abstraction..." << std::endl;
    Shape s1({2, 3, 4});
    assert(s1.ndim() == 3);
    assert(s1.numel() == 24);
    assert(s1[0] == 2 && s1[1] == 3 && s1[2] == 4);

    const auto& strides = s1.strides();
    assert(strides[0] == 12);
    assert(strides[1] == 4);
    assert(strides[2] == 1);
    (void)strides;

    Shape s2({2, 3, 4});
    Shape s3({2, 3, 5});
    assert(s1 == s2);
    assert(s1 != s3);
    (void)s2;
    (void)s3;
    std::cout << "  -> Shape PASSED" << std::endl;
}

void test_allocator_and_storage() {
    std::cout << "[Test] Allocator & Storage..." << std::endl;
    Allocator* alloc = get_default_cpu_allocator();
    size_t before = alloc->allocated_bytes();

    {
        Storage storage(DType::Float32, 100, alloc);
        assert(storage.numel() == 100);
        assert(storage.nbytes() == 100 * sizeof(float));
        assert(storage.data() != nullptr);
        assert(alloc->allocated_bytes() >= before + storage.nbytes());

        storage.fill_ones();
        auto* data = static_cast<float*>(storage.data());
        for (int i = 0; i < 100; ++i) {
            assert(data[i] == 1.0f);
        }

        storage.fill_zeros();
        for (int i = 0; i < 100; ++i) {
            assert(data[i] == 0.0f);
        }
        (void)data;
    }

    assert(alloc->allocated_bytes() == before);
    (void)before;
    std::cout << "  -> Allocator & Storage PASSED" << std::endl;
}

void test_tensor_and_kernels() {
    std::cout << "[Test] Tensor & Element-wise Kernels..." << std::endl;
    Shape shape({4});
    Tensor a = Tensor::ones(shape, DType::Float32);
    Tensor b = Tensor::ones(shape, DType::Float32);
    Tensor out = Tensor::empty(shape, DType::Float32);

    kernels::add(a, b, out);
    const float* out_ptr = out.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        assert(out_ptr[i] == 2.0f);
    }
    (void)out_ptr;

    kernels::mul(out, b, a);
    const float* a_ptr = a.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        assert(a_ptr[i] == 2.0f);
    }
    (void)a_ptr;

    kernels::add_scalar(a, 3.0f, out);
    for (int i = 0; i < 4; ++i) {
        assert(out_ptr[i] == 5.0f);
    }
    std::cout << "  -> Element-wise Kernels PASSED" << std::endl;
}

void test_matmul() {
    std::cout << "[Test] Matmul Kernel..." << std::endl;
    Tensor a = Tensor::empty(Shape({2, 3}), DType::Float32);
    Tensor b = Tensor::empty(Shape({3, 2}), DType::Float32);
    Tensor c = Tensor::empty(Shape({2, 2}), DType::Float32);

    float* a_ptr = a.data_ptr<float>();
    a_ptr[0] = 1.0f; a_ptr[1] = 2.0f; a_ptr[2] = 3.0f;
    a_ptr[3] = 4.0f; a_ptr[4] = 5.0f; a_ptr[5] = 6.0f;

    float* b_ptr = b.data_ptr<float>();
    b_ptr[0] = 7.0f; b_ptr[1] = 8.0f;
    b_ptr[2] = 9.0f; b_ptr[3] = 1.0f;
    b_ptr[4] = 2.0f; b_ptr[5] = 3.0f;

    kernels::matmul(a, b, c);

    const float* c_ptr = c.data_ptr<float>();
    assert(std::fabs(c_ptr[0] - 31.0f) < 1e-5f);
    assert(std::fabs(c_ptr[1] - 19.0f) < 1e-5f);
    assert(std::fabs(c_ptr[2] - 85.0f) < 1e-5f);
    assert(std::fabs(c_ptr[3] - 55.0f) < 1e-5f);
    (void)c_ptr;

    std::cout << "  -> Matmul Kernel PASSED" << std::endl;
}

void test_quantization_kernels() {
    std::cout << "[Test] Quantization & INT8 Matmul Kernels..." << std::endl;
    // Test quantize and dequantize
    Tensor fp32_in = Tensor::empty(Shape({4}), DType::Float32);
    Tensor int8_q = Tensor::empty(Shape({4}), DType::Int8);
    Tensor fp32_out = Tensor::empty(Shape({4}), DType::Float32);

    float* fp_in_ptr = fp32_in.data_ptr<float>();
    fp_in_ptr[0] = -1.0f;
    fp_in_ptr[1] = 0.0f;
    fp_in_ptr[2] = 0.5f;
    fp_in_ptr[3] = 1.0f;

    float scale = 1.0f / 127.0f;
    int32_t zero_point = 0;

    kernels::quantize_float32(fp32_in, int8_q, scale, zero_point);
    int8_t* q_ptr = int8_q.data_ptr<int8_t>();
    assert(q_ptr[0] == -127);
    assert(q_ptr[1] == 0);
    assert(q_ptr[2] == 64);
    assert(q_ptr[3] == 127);
    (void)q_ptr;

    kernels::dequantize_int8(int8_q, fp32_out, scale, zero_point);
    float* fp_out_ptr = fp32_out.data_ptr<float>();
    assert(std::fabs(fp_out_ptr[0] - (-1.0f)) < 1e-2f);
    assert(std::fabs(fp_out_ptr[1] - 0.0f) < 1e-2f);
    assert(std::fabs(fp_out_ptr[2] - 0.5f) < 1e-2f);
    assert(std::fabs(fp_out_ptr[3] - 1.0f) < 1e-2f);
    (void)fp_out_ptr;

    // Test INT8 qmatmul
    Tensor a_q = Tensor::empty(Shape({2, 2}), DType::Int8);
    Tensor b_q = Tensor::empty(Shape({2, 2}), DType::Int8);
    Tensor c_fp = Tensor::empty(Shape({2, 2}), DType::Float32);

    int8_t* a_q_ptr = a_q.data_ptr<int8_t>();
    int8_t* b_q_ptr = b_q.data_ptr<int8_t>();
    a_q_ptr[0] = 10; a_q_ptr[1] = 20;
    a_q_ptr[2] = 30; a_q_ptr[3] = 40;
    b_q_ptr[0] = 1;  b_q_ptr[1] = 2;
    b_q_ptr[2] = 3;  b_q_ptr[3] = 4;

    float s_a = 0.1f;
    float s_b = 0.2f;
    kernels::qmatmul_int8(a_q, b_q, c_fp, s_a, 0, s_b, 0);

    float* c_fp_ptr = c_fp.data_ptr<float>();
    assert(std::fabs(c_fp_ptr[0] - 1.4f) < 1e-5f);
    assert(std::fabs(c_fp_ptr[1] - 2.0f) < 1e-5f);
    assert(std::fabs(c_fp_ptr[2] - 3.0f) < 1e-5f);
    assert(std::fabs(c_fp_ptr[3] - 4.4f) < 1e-5f);
    (void)c_fp_ptr;

    std::cout << "  -> Quantization Kernels PASSED" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Running TensorForge Native C++ Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;

    test_shape();
    test_allocator_and_storage();
    test_tensor_and_kernels();
    test_matmul();
    test_quantization_kernels();

    std::cout << "All Native C++ tests passed successfully!" << std::endl;
    return 0;
}
