cmake_minimum_required(VERSION 3.24)

project(forgenpu_kernels LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(forgenpu_kernels_core
  cpp/src/version.cpp
)

target_include_directories(forgenpu_kernels_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
  message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
else()
  message(STATUS "CUDA toolkit not found; building C++ bridge only")
endif()
