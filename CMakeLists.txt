cmake_minimum_required(VERSION 3.22)
project(mp-dsp-python LANGUAGES CXX)

# C++20 required for sw::dsp and Universal
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Phase 2+: nanobind module will be added here
# find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
# find_package(nanobind CONFIG REQUIRED)
# find_package(dsp CONFIG REQUIRED)
#
# nanobind_add_module(mpdsp_core
#     src/bindings.cpp
#     src/signal_bindings.cpp
#     src/filter_bindings.cpp
#     src/spectral_bindings.cpp
#     src/conditioning_bindings.cpp
#     src/estimation_bindings.cpp
#     src/image_bindings.cpp
#     src/quantization_bindings.cpp
#     src/analysis_bindings.cpp
#     src/io_bindings.cpp
# )
# target_link_libraries(mpdsp_core PRIVATE sw::dsp)

message(STATUS "mp-dsp-python: CMake configured (nanobind module pending Phase 2)")
