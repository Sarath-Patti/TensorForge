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
    src/arena.cpp
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
    # Prefer active python3 from environment PATH if not explicitly overridden
    if(NOT DEFINED Python3_EXECUTABLE)
        find_program(_PYTHON_IN_PATH NAMES python3 python)
        if(_PYTHON_IN_PATH)
            set(Python3_EXECUTABLE "${_PYTHON_IN_PATH}" CACHE FILEPATH "Path to Python3 executable" FORCE)
        endif()
    endif()

    find_package(Python3 COMPONENTS Interpreter Development.Module QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
            OUTPUT_VARIABLE _pybind11_cmakedir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _pybind11_res
        )
        if(_pybind11_res EQUAL 0 AND EXISTS "${_pybind11_cmakedir}")
            set(pybind11_DIR "${_pybind11_cmakedir}")
        endif()
    endif()

    find_package(pybind11 CONFIG QUIET)

    if(pybind11_FOUND)
        message(STATUS "Found pybind11: ${pybind11_DIR} (version \"${pybind11_VERSION}\")")
        pybind11_add_module(_tensorforge_native src/bindings.cpp)
        target_link_libraries(_tensorforge_native PRIVATE tensorforge_native)

        # Output the compiled extension directly into tensorforge/ directory so Python can import it immediately
        set_target_properties(_tensorforge_native PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../tensorforge"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../tensorforge"
        )
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
