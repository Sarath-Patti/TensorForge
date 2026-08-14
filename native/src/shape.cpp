#include "tensorforge/shape.hpp"

#include <numeric>
#include <sstream>
#include <stdexcept>

namespace tensorforge {

Shape::Shape() : dims_({}), strides_({}), numel_(1) {}

Shape::Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {
    compute_strides_and_numel();
}

Shape::Shape(std::initializer_list<int64_t> dims) : dims_(dims) {
    compute_strides_and_numel();
}

void Shape::compute_strides_and_numel() {
    if (dims_.empty()) {
        numel_ = 1;
        strides_.clear();
        return;
    }

    numel_ = 1;
    for (int64_t d : dims_) {
        if (d < 0) {
            throw std::invalid_argument("Shape dimension cannot be negative: " + std::to_string(d));
        }
        numel_ *= d;
    }

    strides_.resize(dims_.size());
    int64_t current_stride = 1;
    for (int i = static_cast<int>(dims_.size()) - 1; i >= 0; --i) {
        strides_[i] = current_stride;
        current_stride *= dims_[i];
    }
}

size_t Shape::ndim() const noexcept {
    return dims_.size();
}

int64_t Shape::numel() const noexcept {
    return numel_;
}

const std::vector<int64_t>& Shape::dims() const noexcept {
    return dims_;
}

const std::vector<int64_t>& Shape::strides() const noexcept {
    return strides_;
}

int64_t Shape::operator[](size_t idx) const {
    if (idx >= dims_.size()) {
        throw std::out_of_range("Shape dimension index out of range");
    }
    return dims_[idx];
}

bool Shape::operator==(const Shape& other) const noexcept {
    return dims_ == other.dims_;
}

bool Shape::operator!=(const Shape& other) const noexcept {
    return dims_ != other.dims_;
}

std::string Shape::to_string() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < dims_.size(); ++i) {
        oss << dims_[i];
        if (dims_.size() == 1) {
            oss << ",";
        } else if (i + 1 < dims_.size()) {
            oss << ", ";
        }
    }
    oss << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    os << shape.to_string();
    return os;
}

} // namespace tensorforge
