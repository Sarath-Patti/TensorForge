#pragma once

#include <memory>
#include <vector>

#include "tensorforge/dtype.hpp"
#include "tensorforge/shape.hpp"
#include "tensorforge/storage.hpp"

namespace tensorforge {

class Tensor {
public:
    explicit Tensor(Shape shape, DType dtype = DType::Float32, Allocator* allocator = nullptr);
    Tensor(Shape shape, std::shared_ptr<Storage> storage);

    const Shape& shape() const noexcept { return shape_; }
    size_t ndim() const noexcept { return shape_.ndim(); }
    int64_t numel() const noexcept { return shape_.numel(); }
    DType dtype() const noexcept { return storage_->dtype(); }
    const std::vector<int64_t>& strides() const noexcept { return shape_.strides(); }
    std::shared_ptr<Storage> storage() const noexcept { return storage_; }

    void* data() noexcept { return storage_->data(); }
    const void* data() const noexcept { return storage_->data(); }

    template <typename T>
    T* data_ptr() noexcept {
        return reinterpret_cast<T*>(storage_->data());
    }

    template <typename T>
    const T* data_ptr() const noexcept {
        return reinterpret_cast<const T*>(storage_->data());
    }

    size_t nbytes() const noexcept { return storage_->nbytes(); }
    bool is_contiguous() const noexcept { return true; }

    static Tensor zeros(Shape shape, DType dtype = DType::Float32);
    static Tensor ones(Shape shape, DType dtype = DType::Float32);
    static Tensor empty(Shape shape, DType dtype = DType::Float32);

private:
    Shape shape_;
    std::shared_ptr<Storage> storage_;
};

} // namespace tensorforge
