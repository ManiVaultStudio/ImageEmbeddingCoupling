# -----------------------------------------------------------------------------
# Test Target
# -----------------------------------------------------------------------------
set(FUNCTION_TESTS FunctionTests)
project(${FUNCTION_TESTS})
message(STATUS "Configure test ${FUNCTION_TESTS}")

# -----------------------------------------------------------------------------
# Third Party libraries
# -----------------------------------------------------------------------------

find_package(Qt6 COMPONENTS Core Widgets WebEngineWidgets REQUIRED)
find_package(catch2 CONFIG REQUIRED)
find_package(ManiVault COMPONENTS Core CONFIG)

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------

set(TEST_SOURCES
    tests_utils.cpp
)

source_group(Tests FILES ${TEST_SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------

add_executable(${FUNCTION_TESTS} ${TEST_SOURCES})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
target_compile_features(${FUNCTION_TESTS} PRIVATE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${FUNCTION_TESTS} PRIVATE /bigobj)	# for Eigen
endif()

# Instruction sets
target_compile_options(${FUNCTION_TESTS} PRIVATE -DUSE_SSE)

ihp_check_and_set_AVX(${FUNCTION_TESTS} ${IHP_USE_AVX})
ihp_set_optimization_level(${FUNCTION_TESTS} ${IHP_OPTIMIZATION_LEVEL})

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------

target_include_directories(${FUNCTION_TESTS} PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../plugins/hsne-analysis/src")
target_include_directories(${FUNCTION_TESTS} PRIVATE "${ManiVault_INCLUDE_DIR}")

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------


target_link_libraries(${FUNCTION_TESTS} PRIVATE InteractiveHsnePlugin)
target_link_libraries(${FUNCTION_TESTS} PRIVATE Catch2::Catch2WithMain)
target_link_libraries(${FUNCTION_TESTS} PRIVATE ManiVault::Core)
target_link_libraries(${FUNCTION_TESTS} PRIVATE Qt6::Core)
