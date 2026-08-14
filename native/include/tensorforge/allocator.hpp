#pragma once

#include <atomic>
#include <cstddef>
#include <string_view>

namespace tensorforge {

class Allocator {
public:
    virtual ~Allocator() = default;

    virtual void* allocate(size_t nbytes, size_t alignment = 64) = 0;
    virtual void deallocate(void* ptr, size_t nbytes) = 0;
    virtual size_t allocated_bytes() const noexcept = 0;
    virtual std::string_view name() const noexcept = 0;
};

class DefaultCPUAllocator : public Allocator {
public:
    DefaultCPUAllocator() : allocated_bytes_(0) {}
    ~DefaultCPUAllocator() override = default;

    void* allocate(size_t nbytes, size_t alignment = 64) override;
    void deallocate(void* ptr, size_t nbytes) override;
    size_t allocated_bytes() const noexcept override;
    std::string_view name() const noexcept override { return "DefaultCPUAllocator"; }

private:
    std::atomic<size_t> allocated_bytes_;
};

Allocator* get_default_cpu_allocator();

} // namespace tensorforge
