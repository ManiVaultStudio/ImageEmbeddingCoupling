cmake_minimum_required(VERSION 3.30...3.31)

option(IHP_USE_AVX "Use AVX if available - by default ON" ON)
option(IHP_BUILD_TESTS "Build Interactive-HSNE-Plugin tests" ON)
set(IHP_OPTIMIZATION_LEVEL "2" CACHE STRING "Optimization level for all IHP targets in release builds, e.g. 0, 1, 2")

# set vcpkg cmake toolchain
if(WIN32 AND NOT DEFINED VCPKG_LIBRARY_LINKAGE)
    set(VCPKG_HOST_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_TARGET_TRIPLET "x64-windows-static-md" CACHE STRING "")
    set(VCPKG_LIBRARY_LINKAGE "static" CACHE STRING "")
endif()

# set up project names
set(HSNE_INTERACTIVE "Interactive-HSNE-Plugin")
project(${HSNE_INTERACTIVE}    
    VERSION 0.0.1
    DESCRIPTION "Interactive image exploration with HSNE"
    LANGUAGES CXX)

# -----------------------------------------------------------------------------
# CMake Options
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(IHP_CMAKE_MODULES_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${IHP_CMAKE_MODULES_PATH})

set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/EHsc /MP /permissive- /Zc:__cplusplus /Zc:preprocessor )
endif()

# Test hardware avx capabilities
include(CMakeCheckSetAVX)

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm|Arm|ARM)")
    set(SPH_USE_AVX OFF CACHE BOOL "Enable some feature for ARM architecture" FORCE)
    message(STATUS "On Apple Silicon AVX instructions are not available, setting SPH_USE_AVX to ${SPH_USE_AVX}.")
endif()

# CMake package manager
include(get_cpm)

# re-applying patches is problematic without CPM_SOURCE_CACHE
# see https://github.com/cpm-cmake/CPM.cmake/issues/577
set(CPM_SOURCE_CACHE ${CMAKE_CURRENT_BINARY_DIR}/.cpm-cache)

# wrapper for cmake fetch utils
include(CMakeFetchUtils)

# -----------------------------------------------------------------------------
# Project targets
# -----------------------------------------------------------------------------
add_subdirectory(plugins/hsne-analysis)
add_subdirectory(plugins/imageviewer)

if(IHP_BUILD_TESTS)
    add_subdirectory(tests)
endif()
