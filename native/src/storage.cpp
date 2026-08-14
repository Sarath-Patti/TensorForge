#include "tensorforge/storage.hpp"

#include <cstring>
#include <stdexcept>

namespace tensorforge {

Storage::Storage(DType dtype, int64_t numel, Allocator* allocator)
    : dtype_(dtype),
      numel_(numel),
      nbytes_(static_cast<size_t>(numel) * dtype_itemsize(dtype)),
      allocator_(allocator ? allocator : get_default_cpu_allocator()),
      data_(nullptr) {
    if (numel_ < 0) {
        throw std::invalid_argument("Storage numel cannot be negative: " + std::to_string(numel_));
    }
    if (nbytes_ > 0) {
        data_ = allocator_->allocate(nbytes_);
    }
}

Storage::~Storage() {
    if (data_ && allocator_) {
        allocator_->deallocate(data_, nbytes_);
        data_ = nullptr;
    }
}

Storage::Storage(Storage&& other) noexcept
    : dtype_(other.dtype_),
      numel_(other.numel_),
      nbytes_(other.nbytes_),
      allocator_(other.allocator_),
      data_(other.data_) {
    other.data_ = nullptr;
    other.numel_ = 0;
    other.nbytes_ = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
        if (data_ && allocator_) {
            allocator_->deallocate(data_, nbytes_);
        }
        dtype_ = other.dtype_;
        numel_ = other.numel_;
        nbytes_ = other.nbytes_;
        allocator_ = other.allocator_;
        data_ = other.data_;

        other.data_ = nullptr;
        other.numel_ = 0;
        other.nbytes_ = 0;
    }
    return *this;
}

void Storage::fill_zeros() {
    if (data_ && nbytes_ > 0) {
        std::memset(data_, 0, nbytes_);
    }
}

void Storage::fill_ones() {
    if (!data_ || numel_ <= 0) {
        return;
    }

    switch (dtype_) {
        case DType::Float32: {
            auto* ptr = static_cast<float*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1.0f;
            break;
        }
        case DType::Float64: {
            auto* ptr = static_cast<double*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1.0;
            break;
        }
        case DType::Int32: {
            auto* ptr = static_cast<int32_t*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1;
            break;
        }
        case DType::Int64: {
            auto* ptr = static_cast<int64_t*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1;
            break;
        }
        case DType::Int8: {
            auto* ptr = static_cast<int8_t*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1;
            break;
        }
        case DType::UInt8: {
            auto* ptr = static_cast<uint8_t*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = 1;
            break;
        }
        case DType::Bool: {
            auto* ptr = static_cast<bool*>(data_);
            for (int64_t i = 0; i < numel_; ++i) ptr[i] = true;
            break;
        }
    }
}

std::shared_ptr<Storage> Storage::clone() const {
    auto copy = std::make_shared<Storage>(dtype_, numel_, allocator_);
    if (data_ && copy->data() && nbytes_ > 0) {
        std::memcpy(copy->data(), data_, nbytes_);
    }
    return copy;
}

} // namespace tensorforge
