#include "tensorforge/arena.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace tensorforge {

WorkspaceArena::WorkspaceArena(
    size_t initial_capacity_bytes,
    Allocator* allocator
) : allocator_(allocator ? allocator : get_default_cpu_allocator()) {
    if (initial_capacity_bytes > 0) {
        reserve(initial_capacity_bytes);
    }
}

WorkspaceArena::~WorkspaceArena() {
    if (data_ && allocator_) {
        allocator_->deallocate(data_, capacity_bytes_);
        data_ = nullptr;
        capacity_bytes_ = 0;
        used_bytes_ = 0;
    }
}

WorkspaceArena::WorkspaceArena(WorkspaceArena&& other) noexcept
    : data_(other.data_),
      capacity_bytes_(other.capacity_bytes_),
      used_bytes_(other.used_bytes_),
      allocator_(other.allocator_) {
    other.data_ = nullptr;
    other.capacity_bytes_ = 0;
    other.used_bytes_ = 0;
    other.allocator_ = nullptr;
}

WorkspaceArena& WorkspaceArena::operator=(WorkspaceArena&& other) noexcept {
    if (this != &other) {
        if (data_ && allocator_) {
            allocator_->deallocate(data_, capacity_bytes_);
        }
        data_ = other.data_;
        capacity_bytes_ = other.capacity_bytes_;
        used_bytes_ = other.used_bytes_;
        allocator_ = other.allocator_;

        other.data_ = nullptr;
        other.capacity_bytes_ = 0;
        other.used_bytes_ = 0;
        other.allocator_ = nullptr;
    }
    return *this;
}

void WorkspaceArena::reserve(size_t num_bytes) {
    if (num_bytes <= capacity_bytes_) {
        return;
    }

    if (!allocator_) {
        allocator_ = get_default_cpu_allocator();
    }

    void* new_data = allocator_->allocate(num_bytes, 64);
    if (!new_data) {
        throw std::bad_alloc();
    }

    if (data_) {
        allocator_->deallocate(data_, capacity_bytes_);
    }

    data_ = new_data;
    capacity_bytes_ = num_bytes;
    used_bytes_ = 0;
}

void WorkspaceArena::reset() {
    used_bytes_ = 0;
}

uintptr_t WorkspaceArena::allocate_slice(size_t num_bytes, size_t alignment) {
    if (!data_ || capacity_bytes_ == 0) {
        throw std::runtime_error("WorkspaceArena has no allocated memory capacity. Call reserve() first.");
    }

    uintptr_t base_addr = reinterpret_cast<uintptr_t>(data_);
    uintptr_t current_addr = base_addr + used_bytes_;
    uintptr_t aligned_addr = (current_addr + (alignment - 1)) & ~(alignment - 1);
    size_t new_used = (aligned_addr - base_addr) + num_bytes;

    if (new_used > capacity_bytes_) {
        throw std::runtime_error(
            "WorkspaceArena out of capacity: requested total " + std::to_string(new_used) +
            " bytes, but capacity is " + std::to_string(capacity_bytes_) + " bytes."
        );
    }

    used_bytes_ = new_used;
    return aligned_addr;
}

uintptr_t WorkspaceArena::data_ptr() const {
    return reinterpret_cast<uintptr_t>(data_);
}

} // namespace tensorforge
