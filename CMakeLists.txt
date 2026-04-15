cmake_minimum_required(VERSION 3.22)
project(mp-dsp-python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and nanobind
find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)
find_package(nanobind CONFIG REQUIRED)

# Find or fetch mixed-precision-dsp (header-only)
# Try local peer directory first, then FetchContent
set(DSP_PEER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../dsp")
if(EXISTS "${DSP_PEER_DIR}/include/sw/dsp/dsp.hpp")
  message(STATUS "Found mixed-precision-dsp at: ${DSP_PEER_DIR}")
  add_library(sw_dsp INTERFACE)
  target_include_directories(sw_dsp INTERFACE "${DSP_PEER_DIR}/include")
else()
  message(STATUS "mixed-precision-dsp not found locally, fetching from GitHub")
  include(FetchContent)
  FetchContent_Declare(dsp
    GIT_REPOSITORY https://github.com/stillwater-sc/mixed-precision-dsp.git
    GIT_TAG main
    GIT_SHALLOW TRUE
  )
  FetchContent_Populate(dsp)
  add_library(sw_dsp INTERFACE)
  target_include_directories(sw_dsp INTERFACE "${dsp_SOURCE_DIR}/include")
endif()

# Find or fetch Universal (header-only)
find_package(universal CONFIG QUIET)
if(universal_FOUND)
  message(STATUS "Found Universal via find_package")
  target_link_libraries(sw_dsp INTERFACE universal::universal)
else()
  set(UNIVERSAL_PEER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../universal")
  if(EXISTS "${UNIVERSAL_PEER_DIR}/include/sw/universal/universal")
    message(STATUS "Found Universal at: ${UNIVERSAL_PEER_DIR}")
    target_include_directories(sw_dsp INTERFACE
      "${UNIVERSAL_PEER_DIR}/include"
      "${UNIVERSAL_PEER_DIR}/include/sw"
    )
  else()
    message(STATUS "Universal not found locally, fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(universal
      GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
      GIT_TAG main
      GIT_SHALLOW TRUE
    )
    FetchContent_Populate(universal)
    target_include_directories(sw_dsp INTERFACE
      "${universal_SOURCE_DIR}/include"
      "${universal_SOURCE_DIR}/include/sw"
    )
  endif()
endif()

# Find or fetch MTL5 (header-only)
find_package(MTL5 CONFIG QUIET)
if(MTL5_FOUND)
  message(STATUS "Found MTL5 via find_package")
  target_link_libraries(sw_dsp INTERFACE MTL5::mtl5)
else()
  set(MTL5_PEER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../mtl5")
  if(EXISTS "${MTL5_PEER_DIR}/include/mtl/mtl.hpp")
    message(STATUS "Found MTL5 at: ${MTL5_PEER_DIR}")
    target_include_directories(sw_dsp INTERFACE "${MTL5_PEER_DIR}/include")
  else()
    message(STATUS "MTL5 not found locally, fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(mtl5
      GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
      GIT_TAG main
      GIT_SHALLOW TRUE
    )
    FetchContent_Populate(mtl5)
    target_include_directories(sw_dsp INTERFACE "${mtl5_SOURCE_DIR}/include")
  endif()
endif()

# Build the nanobind module
nanobind_add_module(_core
  src/bindings.cpp
  src/signal_bindings.cpp
  src/quantization_bindings.cpp
  src/spectral_bindings.cpp
)
target_link_libraries(_core PRIVATE sw_dsp)
target_compile_definitions(_core PRIVATE
  # Suppress Universal warnings in binding code
  $<$<CXX_COMPILER_ID:GNU>:>
  $<$<CXX_COMPILER_ID:Clang>:>
)

# Install the module into the Python package
install(TARGETS _core LIBRARY DESTINATION mpdsp)
