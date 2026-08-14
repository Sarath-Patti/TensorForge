#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace tensorforge {

enum class DType : uint8_t {
    Float32 = 0,
    Float64 = 1,
    Int32 = 2,
    Int64 = 3,
    Int8 = 4,
    UInt8 = 5,
    Bool = 6,
};

inline size_t dtype_itemsize(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
        case DType::Int8:    return sizeof(int8_t);
        case DType::UInt8:   return sizeof(uint8_t);
        case DType::Bool:    return sizeof(bool);
        default:             return 0;
    }
}

inline std::string_view dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::Int8:    return "int8";
        case DType::UInt8:   return "uint8";
        case DType::Bool:    return "bool";
        default:             return "unknown";
    }
}

} // namespace tensorforge
