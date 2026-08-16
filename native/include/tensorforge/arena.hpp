#pragma once

#include <cstddef>
#include <cstdint>
#include "tensorforge/allocator.hpp"

namespace tensorforge {

/**
 * @brief High-performance contiguous workspace memory arena for compiled inference execution.
 * 
 * Provides 64-byte aligned, reusable CPU memory storage managed with RAII lifetime guarantees.
 * Eliminates dynamic memory allocations during steady-state neural network predictions.
 */
class WorkspaceArena {
public:
    explicit WorkspaceArena(
        size_t initial_capacity_bytes = 0,
        Allocator* allocator = nullptr
    );

    ~WorkspaceArena();

    // Disable copy semantics to guarantee unique ownership
    WorkspaceArena(const WorkspaceArena&) = delete;
    WorkspaceArena& operator=(const WorkspaceArena&) = delete;

    // Enable move semantics
    WorkspaceArena(WorkspaceArena&& other) noexcept;
    WorkspaceArena& operator=(WorkspaceArena&& other) noexcept;

    /**
     * @brief Reserve memory capacity in the arena.
     * 
     * If requested bytes exceed current capacity, reallocates a 64-byte aligned buffer.
     * 
     * @param num_bytes Total capacity in bytes.
     */
    void reserve(size_t num_bytes);

    /**
     * @brief Reset the bump-allocation offset to reuse arena memory.
     */
    void reset();

    /**
     * @brief Allocate a contiguous slice of memory from the arena.
     * 
     * @param num_bytes Number of bytes to allocate.
     * @param alignment Byte alignment (default: 64).
     * @return Virtual memory address of the allocated slice.
     */
    uintptr_t allocate_slice(size_t num_bytes, size_t alignment = 64);

    /**
     * @brief Return the starting virtual memory address of the arena.
     */
    uintptr_t data_ptr() const;

    /**
     * @brief Total allocated capacity in bytes.
     */
    size_t capacity_bytes() const { return capacity_bytes_; }

    /**
     * @brief Number of bytes currently consumed by active slices.
     */
    size_t used_bytes() const { return used_bytes_; }

    /**
     * @brief Check if the arena buffer is currently allocated.
     */
    bool is_allocated() const { return data_ != nullptr; }

private:
    void* data_{nullptr};
    size_t capacity_bytes_{0};
    size_t used_bytes_{0};
    Allocator* allocator_{nullptr};
};

} // namespace tensorforge
