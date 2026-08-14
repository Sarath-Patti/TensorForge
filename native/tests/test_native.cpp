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

    Shape s2({2, 3, 4});
    Shape s3({2, 3, 5});
    assert(s1 == s2);
    assert(s1 != s3);
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
    }

    assert(alloc->allocated_bytes() == before);
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

    kernels::mul(out, b, a);
    const float* a_ptr = a.data_ptr<float>();
    for (int i = 0; i < 4; ++i) {
        assert(a_ptr[i] == 2.0f);
    }

    kernels::add_scalar(a, 3.0f, out);
    for (int i = 0; i < 4; ++i) {
        assert(out_ptr[i] == 5.0f);
    }
    std::cout << "  -> Element-wise Kernels PASSED" << std::endl;
}

void test_matmul() {
    std::cout << "[Test] Matmul Kernel..." << std::endl;
    // A: (2, 3), B: (3, 2), Out: (2, 2)
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7, 8],
    //      [9, 1],
    //      [2, 3]]
    // Expected C = [[31, 19],
    //               [85, 55]]
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

    std::cout << "  -> Matmul Kernel PASSED" << std::endl;
}

int main() {
    std::cout << "==========================================" << std::endl;
    std::cout << "Running TensorForge Native C++ Test Suite" << std::endl;
    std::cout << "==========================================" << std::endl;

    test_shape();
    test_allocator_and_storage();
    test_tensor_and_kernels();
    test_matmul();

    std::cout << "All Native C++ tests passed successfully!" << std::endl;
    return 0;
}
