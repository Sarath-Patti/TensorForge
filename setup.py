"""Build configuration for TensorForge Python package and optional native C++ extension."""

from setuptools import setup, find_packages

ext_modules = []
cmdclass = {}

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "tensorforge._tensorforge_native",
            sources=[
                "native/src/dtype.cpp",
                "native/src/shape.cpp",
                "native/src/allocator.cpp",
                "native/src/storage.cpp",
                "native/src/arena.cpp",
                "native/src/tensor.cpp",
                "native/src/kernels.cpp",
                "native/src/bindings.cpp",
            ],
            include_dirs=["native/include"],
            cxx_std=17,
        ),
    ]
    cmdclass = {"build_ext": build_ext}
except ImportError:
    pass

setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
