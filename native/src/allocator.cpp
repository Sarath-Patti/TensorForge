#include "tensorforge/allocator.hpp"

#include <cstdlib>
#include <new>
#include <stdexcept>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace tensorforge {

void* DefaultCPUAllocator::allocate(size_t nbytes, size_t alignment) {
    if (nbytes == 0) {
        return nullptr;
    }

    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(nbytes, alignment);
    if (!ptr) {
        throw std::bad_alloc();
    }
#else
    // posix_memalign requires alignment to be a power of two and a multiple of sizeof(void*)
    size_t align = (alignment < sizeof(void*)) ? sizeof(void*) : alignment;
    int ret = posix_memalign(&ptr, align, nbytes);
    if (ret != 0 || !ptr) {
        throw std::bad_alloc();
    }
#endif

    allocated_bytes_.fetch_add(nbytes, std::memory_order_relaxed);
    return ptr;
}

void DefaultCPUAllocator::deallocate(void* ptr, size_t nbytes) {
    if (!ptr) {
        return;
    }

#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif

    allocated_bytes_.fetch_sub(nbytes, std::memory_order_relaxed);
}

size_t DefaultCPUAllocator::allocated_bytes() const noexcept {
    return allocated_bytes_.load(std::memory_order_relaxed);
}

Allocator* get_default_cpu_allocator() {
    static DefaultCPUAllocator default_allocator;
    return &default_allocator;
}

} // namespace tensorforge
