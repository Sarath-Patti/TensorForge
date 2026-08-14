cmake_minimum_required(VERSION 3.15)
project(tensorforge_native VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Core native library sources
set(NATIVE_SOURCES
    src/dtype.cpp
    src/shape.cpp
    src/allocator.cpp
    src/storage.cpp
    src/tensor.cpp
    src/kernels.cpp
)

add_library(tensorforge_native STATIC ${NATIVE_SOURCES})

target_include_directories(tensorforge_native PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Optional Python bindings via pybind11
option(BUILD_PYTHON_BINDINGS "Build Python pybind11 bindings" ON)

if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG QUIET)
    if(pybind11_FOUND)
        pybind11_add_module(_tensorforge_native src/bindings.cpp)
        target_link_libraries(_tensorforge_native PRIVATE tensorforge_native)
    else()
        message(STATUS "pybind11 not found; skipping _tensorforge_native Python extension target.")
    endif()
endif()

# Standalone C++ test suite
option(BUILD_TESTS "Build standalone native C++ tests" ON)

if(BUILD_TESTS)
    enable_testing()
    add_executable(native_tests tests/test_native.cpp)
    target_link_libraries(native_tests PRIVATE tensorforge_native)
    add_test(NAME NativeTests COMMAND native_tests)
endif()
