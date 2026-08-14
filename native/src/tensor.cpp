#include "tensorforge/tensor.hpp"

#include <stdexcept>

namespace tensorforge {

Tensor::Tensor(Shape shape, DType dtype, Allocator* allocator)
    : shape_(std::move(shape)),
      storage_(std::make_shared<Storage>(dtype, shape_.numel(), allocator)) {}

Tensor::Tensor(Shape shape, std::shared_ptr<Storage> storage)
    : shape_(std::move(shape)), storage_(std::move(storage)) {
    if (!storage_) {
        throw std::invalid_argument("Cannot initialize Tensor with null storage pointer");
    }
    if (storage_->numel() < shape_.numel()) {
        throw std::invalid_argument("Storage capacity (" + std::to_string(storage_->numel()) +
                                    ") is smaller than Tensor element count (" +
                                    std::to_string(shape_.numel()) + ")");
    }
}

Tensor Tensor::zeros(Shape shape, DType dtype) {
    Tensor t(std::move(shape), dtype);
    t.storage()->fill_zeros();
    return t;
}

Tensor Tensor::ones(Shape shape, DType dtype) {
    Tensor t(std::move(shape), dtype);
    t.storage()->fill_ones();
    return t;
}

Tensor Tensor::empty(Shape shape, DType dtype) {
    return Tensor(std::move(shape), dtype);
}

} // namespace tensorforge
