#pragma once

#include <cstdint>
#include <memory>
#include <string_view>

#include "tensorforge/allocator.hpp"
#include "tensorforge/dtype.hpp"

namespace tensorforge {

class Storage : public std::enable_shared_from_this<Storage> {
public:
    Storage(DType dtype, int64_t numel, Allocator* allocator = nullptr);
    ~Storage();

    // Disable copy to prevent accidental shallow/double-free issues
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    // Enable move
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;

    DType dtype() const noexcept { return dtype_; }
    int64_t numel() const noexcept { return numel_; }
    size_t nbytes() const noexcept { return nbytes_; }
    size_t itemsize() const noexcept { return dtype_itemsize(dtype_); }
    std::string_view device() const noexcept { return "cpu"; }

    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    intptr_t data_ptr() const noexcept { return reinterpret_cast<intptr_t>(data_); }

    void fill_zeros();
    void fill_ones();
    std::shared_ptr<Storage> clone() const;

private:
    DType dtype_;
    int64_t numel_;
    size_t nbytes_;
    Allocator* allocator_;
    void* data_;
};

} // namespace tensorforge
