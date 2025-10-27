import os

from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension as Pybind11Extension


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
        ]
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        os.system(f"cmake -S . -B {build_temp} {' '.join(cmake_args)}")
        os.system(f"cmake --build {build_temp}")


setup(
    name="spatial_utils",
    version="0.1",
    packages=find_packages(),
    package_data={"geospatial_neural_adapter.cpp_extensions": ["*.so"]},
    include_package_data=True,
    zip_safe=False,
    ext_modules=[
        Pybind11Extension(
            "spatial_utils",
            ["spatial_utils.cpp"],
            # ... existing code ...
        ),
    ],
)
