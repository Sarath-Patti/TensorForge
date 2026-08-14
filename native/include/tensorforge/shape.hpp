#pragma once

#include <cstdint>
#include <initializer_list>
#include <string>
#include <vector>
#include <iostream>

namespace tensorforge {

class Shape {
public:
    Shape();
    explicit Shape(std::vector<int64_t> dims);
    Shape(std::initializer_list<int64_t> dims);

    size_t ndim() const noexcept;
    int64_t numel() const noexcept;
    const std::vector<int64_t>& dims() const noexcept;
    const std::vector<int64_t>& strides() const noexcept;

    int64_t operator[](size_t idx) const;
    bool operator==(const Shape& other) const noexcept;
    bool operator!=(const Shape& other) const noexcept;

    std::string to_string() const;

private:
    std::vector<int64_t> dims_;
    std::vector<int64_t> strides_;
    int64_t numel_;

    void compute_strides_and_numel();
};

std::ostream& operator<<(std::ostream& os, const Shape& shape);

} // namespace tensorforge
