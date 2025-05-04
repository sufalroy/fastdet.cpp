cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(fastdet_cpp
        VERSION 0.1.0
        DESCRIPTION "Fast Object Detection Inference Engine"
        LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(cmake/CPM.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Sanitizers.cmake)

cpmaddpackage("gh:fmtlib/fmt#11.1.4")
cpmaddpackage(
        NAME spdlog
        VERSION 1.15.2
        GITHUB_REPOSITORY "gabime/spdlog"
        OPTIONS "SPDLOG_FMT_EXTERNAL ON"
)

find_package(CUDAToolkit REQUIRED)
find_package(cuDNN REQUIRED)
find_package(TensorRT 10 REQUIRED COMPONENTS OnnxParser Plugin)

add_subdirectory(src)