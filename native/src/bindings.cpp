#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensorforge/allocator.hpp"
#include "tensorforge/dtype.hpp"
#include "tensorforge/kernels.hpp"
#include "tensorforge/shape.hpp"
#include "tensorforge/storage.hpp"
#include "tensorforge/tensor.hpp"

namespace py = pybind11;
using namespace tensorforge;

PYBIND11_MODULE(_tensorforge_native, m) {
    m.doc() = "TensorForge Native C++17 Runtime & Kernel Extension";

    m.def("is_native_available", []() { return true; }, "Check if C++ native runtime is compiled and available.");
    m.def("native_allocated_bytes", []() {
        return get_default_cpu_allocator()->allocated_bytes();
    }, "Return total active memory allocated via native CPU allocator in bytes.");

    py::enum_<DType>(m, "DType")
        .value("Float32", DType::Float32)
        .value("Float64", DType::Float64)
        .value("Int32", DType::Int32)
        .value("Int64", DType::Int64)
        .value("Int8", DType::Int8)
        .value("UInt8", DType::UInt8)
        .value("Bool", DType::Bool)
        .export_values();

    py::class_<Shape>(m, "Shape")
        .def(py::init<std::vector<int64_t>>())
        .def("ndim", &Shape::ndim)
        .def("numel", &Shape::numel)
        .def("dims", &Shape::dims)
        .def("strides", &Shape::strides)
        .def("__repr__", &Shape::to_string)
        .def("__eq__", &Shape::operator==);

    py::class_<Storage, std::shared_ptr<Storage>>(m, "Storage")
        .def(py::init<DType, int64_t>(), py::arg("dtype"), py::arg("numel"))
        .def("dtype", &Storage::dtype)
        .def("numel", &Storage::numel)
        .def("nbytes", &Storage::nbytes)
        .def("itemsize", &Storage::itemsize)
        .def("device", &Storage::device)
        .def("data_ptr", &Storage::data_ptr)
        .def("fill_zeros", &Storage::fill_zeros)
        .def("fill_ones", &Storage::fill_ones);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<Shape, DType>(), py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("zeros", &Tensor::zeros, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("ones", &Tensor::ones, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def_static("empty", &Tensor::empty, py::arg("shape"), py::arg("dtype") = DType::Float32)
        .def("shape", &Tensor::shape)
        .def("ndim", &Tensor::ndim)
        .def("numel", &Tensor::numel)
        .def("dtype", &Tensor::dtype)
        .def("strides", &Tensor::strides)
        .def("nbytes", &Tensor::nbytes)
        .def("is_contiguous", &Tensor::is_contiguous)
        .def("storage", &Tensor::storage);

    m.def("native_add", [](const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape(), a.dtype());
        kernels::add(a, b, out);
        return out;
    }, "Element-wise addition: out = a + b");

    m.def("native_sub", [](const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape(), a.dtype());
        kernels::sub(a, b, out);
        return out;
    }, "Element-wise subtraction: out = a - b");

    m.def("native_mul", [](const Tensor& a, const Tensor& b) {
        Tensor out = Tensor::empty(a.shape(), a.dtype());
        kernels::mul(a, b, out);
        return out;
    }, "Element-wise multiplication: out = a * b");

    m.def("native_matmul", [](const Tensor& a, const Tensor& b) {
        Shape out_shape({a.shape()[0], b.shape()[1]});
        Tensor out = Tensor::empty(out_shape, a.dtype());
        kernels::matmul(a, b, out);
        return out;
    }, "Matrix multiplication: out = a @ b");

    m.def("native_qmatmul", [](
        const Tensor& a,
        const Tensor& b,
        float scale_a,
        int32_t zp_a,
        float scale_b,
        int32_t zp_b
    ) {
        Shape out_shape({a.shape()[0], b.shape()[1]});
        Tensor out = Tensor::empty(out_shape, DType::Float32);
        kernels::qmatmul_int8(a, b, out, scale_a, zp_a, scale_b, zp_b);
        return out;
    }, py::arg("a"), py::arg("b"), py::arg("scale_a"), py::arg("zp_a") = 0, py::arg("scale_b"), py::arg("zp_b") = 0,
       "Quantized INT8 matrix multiplication producing float32 output: out = a @ b");

    m.def("native_dequantize", [](const Tensor& in, float scale, int32_t zero_point) {
        Tensor out = Tensor::empty(in.shape(), DType::Float32);
        kernels::dequantize_int8(in, out, scale, zero_point);
        return out;
    }, py::arg("in"), py::arg("scale"), py::arg("zero_point") = 0,
       "Dequantize INT8 tensor to Float32");

    m.def("native_quantize", [](const Tensor& in, float scale, int32_t zero_point) {
        Tensor out = Tensor::empty(in.shape(), DType::Int8);
        kernels::quantize_float32(in, out, scale, zero_point);
        return out;
    }, py::arg("in"), py::arg("scale"), py::arg("zero_point") = 0,
       "Quantize Float32 tensor to INT8");
}
