cmake_minimum_required(VERSION 3.10)
project(spatial_utils)

# === C++ Config ===
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# === Armadillo: Disable MKL fallback ===
add_definitions(-DARMA_DONT_USE_WRAPPER)
set(ENV{ARMA_DONT_USE_WRAPPER} 1)

# === Python & Pybind11 ===
find_package(Python3 3.10 EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)

# === Armadillo ===
if(APPLE AND EXISTS "/opt/homebrew/lib/libarmadillo.dylib")
    # Prefer Homebrew installation if available
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew")
    set(ARMADILLO_INCLUDE_DIRS "/opt/homebrew/include")
    set(ARMADILLO_LIBRARIES "/opt/homebrew/lib/libarmadillo.dylib")
    set(BLA_VENDOR OpenBLAS)
else()
    # Use Conda environment
    set(ARMADILLO_INCLUDE_DIRS "$ENV{CONDA_PREFIX}/include")
    if(APPLE)
        set(ARMADILLO_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libarmadillo.dylib")
    else()
        set(ARMADILLO_LIBRARIES "$ENV{CONDA_PREFIX}/lib/libarmadillo.so")
    endif()
endif()

# === BLAS / LAPACK ===
# Set OpenBLAS library path based on platform
if(APPLE)
    set(OPENBLAS_LIB_HINT "$ENV{CONDA_PREFIX}/lib/libopenblas.dylib")
else()
    set(OPENBLAS_LIB_HINT "$ENV{CONDA_PREFIX}/lib/libopenblas.so")
endif()

if(EXISTS "${OPENBLAS_LIB_HINT}")
    message(STATUS "✅ Found OpenBLAS: ${OPENBLAS_LIB_HINT}")
    set(OPENBLAS_LIB ${OPENBLAS_LIB_HINT})
else()
    message(FATAL_ERROR "❌ OpenBLAS not found at ${OPENBLAS_LIB_HINT}. Please install it via Conda.")
endif()

find_library(LAPACK_LIB NAMES lapack PATHS "$ENV{CONDA_PREFIX}/lib" NO_DEFAULT_PATH)
if(NOT LAPACK_LIB)
    message(WARNING "⚠️ LAPACK not found. Using OpenBLAS LAPACK implementation instead.")
    set(LAPACK_LIB ${OPENBLAS_LIB})
endif()

# === Module Build ===
pybind11_add_module(spatial_utils spatial_utils.cpp)

# === Compiler/Linker Flags ===
target_include_directories(spatial_utils PRIVATE ${ARMADILLO_INCLUDE_DIRS})
target_compile_definitions(spatial_utils PRIVATE ARMA_DONT_USE_WRAPPER)
target_link_libraries(spatial_utils PRIVATE
    ${ARMADILLO_LIBRARIES}
    ${OPENBLAS_LIB}
    ${LAPACK_LIB}
)

# === Output Location ===
set_target_properties(spatial_utils PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_NAME "spatial_utils"
    SUFFIX ".so"
)
