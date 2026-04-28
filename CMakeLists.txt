cmake_minimum_required(VERSION 3.22)

# Version tracks the mixed-precision-dsp C++ library in lockstep (see
# docs convention matching mtl5 / mtl5-python). Python-only patches
# ship as PEP 440 post-releases (e.g. 0.4.1.post1) without bumping
# this value. scikit-build-core reads the version from this line via
# the regex provider configured in pyproject.toml — do not hardcode
# it anywhere else.
project(mp-dsp-python
	VERSION 0.6.0
	LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Peer dependency requirements.
#
# Two intentionally-decoupled concepts:
#
#   MPDSP_REQUIRED_*_VERSION — floors enforced at configure time on the
#       sibling-path branch (see _DSP_CANDIDATES / _UNI_CANDIDATES /
#       _MTL5_CANDIDATES loops below). When a sibling checkout is older
#       than the floor, configure aborts with a clear error. These move
#       to the latest released peer versions during a development cycle
#       so dev clones cannot silently be stale.
#
#   MPDSP_*_PIN — git refs (tags, branches, or SHAs) that FetchContent uses
#       on the fallback path. cibuildwheel has no siblings to find, so the
#       pin is what determines what gets compiled into the wheel. The DSP
#       pin moves in lockstep with `project(... VERSION ...)` only at
#       release time (see test_version.py::test_lockstep_prefix). Thus the
#       pin can lag the floor during a development cycle.
#
# Override at configure time:
#   -DMPDSP_REQUIRED_DSP_VERSION=0.5.0   (lower floor for experimentation)
#   -DMPDSP_DSP_PIN=main                 (build against unreleased upstream)
# ---------------------------------------------------------------------------
set(MPDSP_REQUIRED_DSP_VERSION       "0.6.0"  CACHE STRING
	"Minimum mixed-precision-dsp version required at configure time")
set(MPDSP_REQUIRED_UNIVERSAL_VERSION "4.6.11" CACHE STRING
	"Minimum universal version required at configure time")
set(MPDSP_REQUIRED_MTL5_VERSION      "5.2.1"  CACHE STRING
	"Minimum mtl5 version required at configure time")

set(MPDSP_DSP_PIN       "v0.6.0" CACHE STRING
	"mixed-precision-dsp git tag / branch / SHA to fetch")
# universal and mtl5 pins move freely (only DSP is in the lockstep test);
# track the latest released versions of both so CI matches the
# sibling-path floor.
set(MPDSP_UNIVERSAL_PIN "v4.6.11" CACHE STRING
	"universal git tag / branch / SHA to fetch")
set(MPDSP_MTL5_PIN      "v5.2.1"  CACHE STRING
	"mtl5 git tag / branch / SHA to fetch")

# Shallow-fetch is a big speedup for tags and branches, but git's shallow
# protocol doesn't support fetching an arbitrary commit SHA — it needs a
# refname. Detect SHA-looking pins (7–40 hex chars, no non-hex characters)
# and turn off GIT_SHALLOW for those cases only.
foreach(_pin IN ITEMS DSP UNIVERSAL MTL5)
	if(MPDSP_${_pin}_PIN MATCHES "^[a-fA-F0-9]+$" AND
	   NOT MPDSP_${_pin}_PIN MATCHES "^[a-fA-F0-9]{1,6}$")
		set(_MPDSP_${_pin}_SHALLOW FALSE)
	else()
		set(_MPDSP_${_pin}_SHALLOW TRUE)
	endif()
endforeach()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python — prefer the local .venv if it exists.
# This lets CMake GUI users avoid setting Python_EXECUTABLE manually.
if(NOT Python_EXECUTABLE)
  # Check for .venv in the source directory
  if(WIN32)
    set(_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/Scripts/python.exe")
  else()
    set(_VENV_PYTHON "${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python")
  endif()
  if(EXISTS "${_VENV_PYTHON}")
    set(Python_EXECUTABLE "${_VENV_PYTHON}" CACHE FILEPATH
        "Python interpreter (auto-detected from .venv)")
    message(STATUS "Auto-detected Python venv: ${_VENV_PYTHON}")
  endif()
endif()

find_package(Python 3.9
  REQUIRED COMPONENTS Interpreter Development.Module
)

# Find nanobind — try multiple discovery methods:
# 1. User-provided -Dnanobind_DIR=...
# 2. Auto-discover from Python: python -m nanobind --cmake_dir
# 3. CMAKE_PREFIX_PATH
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _NB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _NB_RESULT
  )
  if(_NB_RESULT EQUAL 0 AND EXISTS "${_NB_DIR}")
    set(nanobind_DIR "${_NB_DIR}" CACHE PATH "nanobind cmake dir (auto-detected)")
    message(STATUS "Auto-detected nanobind at: ${_NB_DIR}")
  endif()
endif()

find_package(nanobind CONFIG REQUIRED)

# ---------------------------------------------------------------------------
# Sibling-clone version floor checks.
#
# When a peer is resolved by sibling-path (rather than FetchContent), verify
# its version meets the floor and abort with a clear error if not. The check
# is permissive when the version cannot be determined (no .git directory,
# unparseable header) — STATUS message only, do not fail. This supports
# tarball checkouts and avoids spurious failures in unusual environments.
# ---------------------------------------------------------------------------
function(mpdsp_check_version_from_header NAME DIR HEADER_PATH REQUIRED CACHE_VAR)
	set(_FULL "${DIR}/${HEADER_PATH}")
	if(NOT EXISTS "${_FULL}")
		message(STATUS "${NAME}: ${HEADER_PATH} not found at ${DIR}; skipping floor check (required >= ${REQUIRED})")
		return()
	endif()
	file(READ "${_FULL}" _CONTENTS)
	set(_MAJOR "")
	set(_MINOR "")
	set(_PATCH "")
	if(_CONTENTS MATCHES "version_major[ \t]*=[ \t]*([0-9]+)")
		set(_MAJOR "${CMAKE_MATCH_1}")
	endif()
	if(_CONTENTS MATCHES "version_minor[ \t]*=[ \t]*([0-9]+)")
		set(_MINOR "${CMAKE_MATCH_1}")
	endif()
	if(_CONTENTS MATCHES "version_patch[ \t]*=[ \t]*([0-9]+)")
		set(_PATCH "${CMAKE_MATCH_1}")
	endif()
	if(_MAJOR STREQUAL "" OR _MINOR STREQUAL "" OR _PATCH STREQUAL "")
		message(STATUS "${NAME}: could not parse version from ${HEADER_PATH}; skipping floor check (required >= ${REQUIRED})")
		return()
	endif()
	set(_VERSION "${_MAJOR}.${_MINOR}.${_PATCH}")
	if(_VERSION VERSION_LESS "${REQUIRED}")
		message(FATAL_ERROR
			"${NAME} at ${DIR} is version ${_VERSION}, but ${PROJECT_NAME} requires >= ${REQUIRED}.\n"
			"Update the sibling clone:\n"
			"  cd ${DIR} && git fetch --tags && git checkout v${REQUIRED}\n"
			"Or lower the floor: -D${CACHE_VAR}=${_VERSION}")
	endif()
	message(STATUS "${NAME} version: ${_VERSION} (required >= ${REQUIRED})")
endfunction()

function(mpdsp_check_version_from_git NAME DIR REQUIRED CACHE_VAR)
	if(NOT EXISTS "${DIR}/.git")
		message(STATUS "${NAME}: no .git at ${DIR}; skipping floor check (required >= ${REQUIRED})")
		return()
	endif()
	execute_process(
		COMMAND git -C "${DIR}" describe --tags --abbrev=0
		OUTPUT_VARIABLE _TAG
		OUTPUT_STRIP_TRAILING_WHITESPACE
		ERROR_QUIET
		RESULT_VARIABLE _RES
	)
	if(NOT _RES EQUAL 0 OR NOT _TAG MATCHES "^v?([0-9]+)\\.([0-9]+)\\.([0-9]+)")
		message(STATUS "${NAME}: could not detect version from git tag at ${DIR}; skipping floor check (required >= ${REQUIRED})")
		return()
	endif()
	set(_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
	if(_VERSION VERSION_LESS "${REQUIRED}")
		message(FATAL_ERROR
			"${NAME} at ${DIR} is version ${_VERSION}, but ${PROJECT_NAME} requires >= ${REQUIRED}.\n"
			"Update the sibling clone:\n"
			"  cd ${DIR} && git fetch --tags && git checkout v${REQUIRED}\n"
			"Or lower the floor: -D${CACHE_VAR}=${_VERSION}")
	endif()
	message(STATUS "${NAME} version: ${_VERSION} (required >= ${REQUIRED})")
endfunction()

# Find or fetch mixed-precision-dsp (header-only)
# Try peer directory (../dsp or ../mixed-precision-dsp), then FetchContent
add_library(sw_dsp INTERFACE)

set(_DSP_CANDIDATES
  "${CMAKE_CURRENT_SOURCE_DIR}/../dsp"
  "${CMAKE_CURRENT_SOURCE_DIR}/../mixed-precision-dsp"
)
set(_DSP_FOUND FALSE)
foreach(_DIR IN LISTS _DSP_CANDIDATES)
  if(EXISTS "${_DIR}/include/sw/dsp/dsp.hpp")
    message(STATUS "Found mixed-precision-dsp at: ${_DIR}")
    mpdsp_check_version_from_header("mixed-precision-dsp" "${_DIR}"
      "include/sw/dsp/version.hpp" "${MPDSP_REQUIRED_DSP_VERSION}"
      "MPDSP_REQUIRED_DSP_VERSION")
    target_include_directories(sw_dsp INTERFACE "${_DIR}/include")
    set(_DSP_FOUND TRUE)
    break()
  endif()
endforeach()

if(NOT _DSP_FOUND)
  message(STATUS "mixed-precision-dsp not found locally, fetching from GitHub")
  include(FetchContent)
  FetchContent_Declare(dsp
    GIT_REPOSITORY https://github.com/stillwater-sc/mixed-precision-dsp.git
    GIT_TAG ${MPDSP_DSP_PIN}
    GIT_SHALLOW ${_MPDSP_DSP_SHALLOW}
  )
  FetchContent_Populate(dsp)
  target_include_directories(sw_dsp INTERFACE "${dsp_SOURCE_DIR}/include")
endif()

# Find or fetch Universal (header-only)
# Prefer peer directory over find_package — the installed Universal cmake
# config has a known bug that sets CMAKE_CXX_STANDARD=14 globally.
set(_UNI_CANDIDATES
  "${CMAKE_CURRENT_SOURCE_DIR}/../universal"
)
set(_UNI_FOUND FALSE)
foreach(_DIR IN LISTS _UNI_CANDIDATES)
  if(EXISTS "${_DIR}/include/sw/universal/number/cfloat/cfloat.hpp")
    message(STATUS "Found Universal at: ${_DIR}")
    mpdsp_check_version_from_git("universal" "${_DIR}"
      "${MPDSP_REQUIRED_UNIVERSAL_VERSION}"
      "MPDSP_REQUIRED_UNIVERSAL_VERSION")
    target_include_directories(sw_dsp INTERFACE
      "${_DIR}/include"
      "${_DIR}/include/sw"
    )
    set(_UNI_FOUND TRUE)
    break()
  endif()
endforeach()

if(NOT _UNI_FOUND)
  message(STATUS "Universal not found locally, fetching from GitHub")
  include(FetchContent)
  FetchContent_Declare(universal
    GIT_REPOSITORY https://github.com/stillwater-sc/universal.git
    GIT_TAG ${MPDSP_UNIVERSAL_PIN}
    GIT_SHALLOW ${_MPDSP_UNIVERSAL_SHALLOW}
  )
  FetchContent_Populate(universal)
  target_include_directories(sw_dsp INTERFACE
    "${universal_SOURCE_DIR}/include"
    "${universal_SOURCE_DIR}/include/sw"
  )
endif()

# Find or fetch MTL5 (header-only)
find_package(MTL5 ${MPDSP_REQUIRED_MTL5_VERSION} CONFIG QUIET)
if(MTL5_FOUND)
  message(STATUS "Found MTL5 ${MTL5_VERSION} via find_package (required >= ${MPDSP_REQUIRED_MTL5_VERSION})")
  target_link_libraries(sw_dsp INTERFACE MTL5::mtl5)
else()
  set(_MTL5_CANDIDATES
    "${CMAKE_CURRENT_SOURCE_DIR}/../mtl5"
  )
  set(_MTL5_FOUND FALSE)
  foreach(_DIR IN LISTS _MTL5_CANDIDATES)
    if(EXISTS "${_DIR}/include/mtl/mtl.hpp")
      message(STATUS "Found MTL5 at: ${_DIR}")
      mpdsp_check_version_from_git("mtl5" "${_DIR}"
        "${MPDSP_REQUIRED_MTL5_VERSION}"
        "MPDSP_REQUIRED_MTL5_VERSION")
      target_include_directories(sw_dsp INTERFACE "${_DIR}/include")
      set(_MTL5_FOUND TRUE)
      break()
    endif()
  endforeach()

  if(NOT _MTL5_FOUND)
    message(STATUS "MTL5 not found locally, fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(mtl5
      GIT_REPOSITORY https://github.com/stillwater-sc/mtl5.git
      GIT_TAG ${MPDSP_MTL5_PIN}
      GIT_SHALLOW ${_MPDSP_MTL5_SHALLOW}
    )
    FetchContent_Populate(mtl5)
    target_include_directories(sw_dsp INTERFACE "${mtl5_SOURCE_DIR}/include")
  endif()
endif()

# Re-enforce C++20 after all find_package calls.
# Universal's cmake config sets CMAKE_CXX_STANDARD=14 globally, overriding
# our project-level setting. Force it back to 20.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build the nanobind module
nanobind_add_module(_core
  src/bindings.cpp
  src/signal_bindings.cpp
  src/quantization_bindings.cpp
  src/spectral_bindings.cpp
  src/filter_bindings.cpp
  src/conditioning_bindings.cpp
  src/estimation_bindings.cpp
  src/image_bindings.cpp
  src/types_bindings.cpp
  src/analysis_bindings.cpp
  src/acquisition_bindings.cpp
)
target_link_libraries(_core PRIVATE sw_dsp)

# C++20 is required for sw::dsp and MTL5 concepts.
# nanobind sets cxx_std_17 PUBLIC; we need to override to C++20.
# Use both target_compile_features AND explicit flags for MSVC.
target_compile_features(_core PUBLIC cxx_std_20)
if(MSVC)
  target_compile_options(_core PRIVATE /std:c++20)
endif()

# Install the module into the Python package.
# scikit-build-core's `wheel.install-dir = "mpdsp"` already prefixes all CMake
# install paths with `mpdsp/`, so DESTINATION must be `.` — using `mpdsp` here
# double-nests the extension to `mpdsp/mpdsp/_core.so` and mpdsp.__init__.py's
# `from mpdsp._core import ...` silently falls through to HAS_CORE=False.
install(TARGETS _core LIBRARY DESTINATION .)
