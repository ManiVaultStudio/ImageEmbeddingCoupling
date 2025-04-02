# -----------------------------------------------------------------------------
# HSNE Plugin Target
# -----------------------------------------------------------------------------
set(IHP_PLUGIN "InteractiveHsnePlugin")
project(${IHP_PLUGIN} C CXX)
message(STATUS "Configure plugin ${IHP_PLUGIN}")

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
find_package(Qt6 COMPONENTS Widgets WebEngineWidgets OpenGL OpenGLWidgets REQUIRED)
find_package(ManiVault COMPONENTS Core PointData ImageData CONFIG)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(OpenMP_RUNTIME_MSVC "llvm" FORCE)
endif()
find_package(OpenMP REQUIRED)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

if(UNIX AND NOT APPLE)
   find_package(TBB REQUIRED)
endif()

if(NOT eigen_SOURCE_DIR)
    set(BUILD_TESTING OFF CACHE BOOL "Enable testing for Eigen" FORCE)
    set(EIGEN_BUILD_TESTING  OFF CACHE BOOL "Enable creation of Eigen tests." FORCE)
    set(EIGEN_BUILD_DOC OFF CACHE BOOL "Enable creation of Eigen documentation" FORCE)
    set(EIGEN_BUILD_DEMOS OFF CACHE BOOL "Toggles the building of the Eigen demos" FORCE)
    fetch_content_url(eigen "https://gitlab.com/libeigen/eigen/-/archive/3866cbfbe8622f41b4f9fa17227aaa7a8de13890/eigen-3866cbfbe8622f41b4f9fa17227aaa7a8de13890.tar.gz") # master as of 26/03/25
endif()

if(TARGET hdislimdimensionalityreduction AND TARGET hdislimutils AND TARGET hdislimdata)
    if(NOT HDILibSlim_LINK_LIBS)
        set(HDILibSlim_LINK_LIBS hdislimdimensionalityreduction hdislimutils hdislimdata)
    endif()
    
    if(NOT HDILibSlim_INCLUDE_DIR AND HDILibSlim_SOURCE_DIR)
        set(HDILibSlim_INCLUDE_DIR ${HDILibSlim_SOURCE_DIR})
    endif()
else()
    set(HDILibSlim_INSTALL OFF)
    set(HDILibSlim_ENABLE_AVX ${IHP_USE_AVX})
    set(HDILibSlim_ENABLE_PID ON)
    set(HDILibSlim_OPTIMIZATION_LEVEL ${IHP_OPTIMIZATION_LEVEL})

    CPMAddPackage(
     NAME              HDILibSlim
     GIT_REPOSITORY    https://github.com/alxvth/HDILibSlim.git
     GIT_TAG           8eb510528dc4169079fa97cb324c13b54d060ec4
     EXCLUDE_FROM_ALL  YES
     OPTIONS "HDILibSlim_INSTALL ${HDILibSlim_INSTALL}"
             "HDILibSlim_ENABLE_AVX ${HDILibSlim_ENABLE_AVX}"
             "HDILibSlim_ENABLE_PID ${HDILibSlim_ENABLE_PID}"
             "HDILibSlim_OPTIMIZATION_LEVEL ${HDILibSlim_OPTIMIZATION_LEVEL}"
    )

    set(HDILibSlim_INCLUDE_DIR ${HDILibSlim_SOURCE_DIR})
    set(HDILibSlim_LINK_LIBS hdislimdimensionalityreduction hdislimutils hdislimdata)

    message(STATUS "HDILibSlim_SOURCE_DIR: ${HDILibSlim_SOURCE_DIR}")

    set_target_properties(hdislimdata hdislimdimensionalityreduction hdislimutils
        PROPERTIES 
        FOLDER HDILibSlim
    )
endif()

if (APPLE)
    target_compile_definitions(hdislimdimensionalityreduction PRIVATE -DGL_SILENCE_DEPRECATION)
    target_compile_definitions(hdislimdata PRIVATE -DGL_SILENCE_DEPRECATION)
    target_compile_definitions(hdislimutils PRIVATE -DGL_SILENCE_DEPRECATION)
endif()

# -----------------------------------------------------------------------------
# Source files
# -----------------------------------------------------------------------------
set(IHP_PLUGIN_HEADERS
    src/InteractiveHsnePlugin.h
    src/HsneHierarchy.h
    src/HsneParameters.h
    src/HsneScaleUpdate.h
)

set(IHP_PLUGIN_SOURCES
    src/InteractiveHsnePlugin.cpp
    src/InteractiveHsnePlugin.json
    src/HsneHierarchy.cpp
    src/HsneScaleUpdate.cpp
)

set(HSNE_ACTIONS_HEADERS
    src/HsneSettingsAction.h
    src/GeneralHsneSettingsAction.h
    src/AdvancedHsneSettingsAction.h
    src/HsneScaleAction.h
    src/ViewportSequence.h
    src/RegularHsneAction.h
    src/RecolorAction.h
    src/ViewportSharingActions.h
)

set(HSNE_ACTIONS_SOURCES
    src/HsneSettingsAction.cpp
    src/GeneralHsneSettingsAction.cpp
    src/AdvancedHsneSettingsAction.cpp
    src/HsneScaleAction.cpp
    src/ViewportSequence.cpp
    src/RegularHsneAction.cpp
    src/RecolorAction.cpp
    src/ViewportSharingActions.cpp
)

set(UTILS_HEADERS
    src/Utils.h
    src/UtilsScale.h
    src/CommonTypes.h
    src/PCA.h
    src/Logger.h
)

set(UTILS_SOURCES
    src/Utils.cpp
    src/UtilsScale.cpp
    src/Logger.cpp
)

set(TSNE_COMMON_SOURCES
    src/TsneAnalysis.h
    src/TsneAnalysis.cpp
    src/TsneData.h
    src/TsneParameters.h
    src/OffscreenBuffer.h
    src/OffscreenBuffer.cpp
)

set(TSNE_ACTIONS_SOURCES
    src/TsneComputationAction.h
    src/TsneComputationAction.cpp
    src/TsneSettingsAction.h
    src/TsneSettingsAction.cpp
    src/GeneralTsneSettingsAction.h
    src/GeneralTsneSettingsAction.cpp
)

set(DIMENSION_SELECTION_ACTION_SOURCES
    src/DimensionSelectionAction.h
    src/DimensionSelectionAction.cpp
)

set(MEAN_SHIFT_SOURCES
    src/MeanShiftAction.h
    src/MeanShiftAction.cpp
)

set(HSNE_SOURCES
    ${HSNE_ACTIONS_HEADERS}
    ${HSNE_ACTIONS_SOURCES}
    ${TSNE_ACTIONS_SOURCES}
    ${DIMENSION_SELECTION_ACTION_SOURCES}
    ${TSNE_COMMON_SOURCES}
    ${IHP_PLUGIN_HEADERS}
    ${IHP_PLUGIN_SOURCES}
    ${UTILS_HEADERS}
    ${UTILS_SOURCES}
    ${MEAN_SHIFT_SOURCES}
)

source_group(Plugin FILES ${IHP_PLUGIN_HEADERS} ${IHP_PLUGIN_SOURCES})
source_group(Actions FILES ${HSNE_ACTIONS_HEADERS} ${HSNE_ACTIONS_SOURCES} ${DIMENSION_SELECTION_ACTION_SOURCES} ${TSNE_ACTIONS_SOURCES} ${MEAN_SHIFT_SOURCES})
source_group(TSNE FILES ${TSNE_COMMON_SOURCES})
source_group(Utils FILES ${UTILS_HEADERS} ${UTILS_SOURCES})

# -----------------------------------------------------------------------------
# CMake Target
# -----------------------------------------------------------------------------
add_library(${IHP_PLUGIN} SHARED ${HSNE_SOURCES})

# -----------------------------------------------------------------------------
# Target properties
# -----------------------------------------------------------------------------
target_compile_features(${IHP_PLUGIN} PRIVATE cxx_std_20)
target_compile_features(${IHP_PLUGIN} PRIVATE c_std_17)

set_target_properties(${IHP_PLUGIN} PROPERTIES AUTOMOC TRUE)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${IHP_PLUGIN} PRIVATE $<$<CONFIG:DEBUG>:/MDd> $<$<CONFIG:RELWITHDEBINFO>:/MD> $<$<CONFIG:RELEASE>:/MD>)
    target_compile_options(${IHP_PLUGIN} PRIVATE /bigobj)	# for Eigen
endif()

# Warning levels
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${IHP_PLUGIN} PRIVATE /W3)
else()
    target_compile_options(${IHP_PLUGIN} PRIVATE -Wall)
endif()

# Suppress warning from biovault_bfloat16 in ManiVault core
target_compile_definitions(${IHP_PLUGIN} PRIVATE _SILENCE_CXX20_IS_POD_DEPRECATION_WARNING)

# Instruction sets and optimization
ihp_check_and_set_AVX(${IHP_PLUGIN} ${IHP_USE_AVX})
ihp_set_optimization_level(${IHP_PLUGIN} ${IHP_OPTIMIZATION_LEVEL})

set_target_properties(${IHP_PLUGIN} PROPERTIES DEBUG_POSTFIX "d")

# -----------------------------------------------------------------------------
# Target include directories
# -----------------------------------------------------------------------------
# Include ManiVault core headers
target_include_directories(${IHP_PLUGIN} PRIVATE "${ManiVault_INCLUDE_DIR}")

target_include_directories(${IHP_PLUGIN} PUBLIC  "${HDILibSlim_SOURCE_DIR}")
target_include_directories(${IHP_PLUGIN} PUBLIC  "${eigen_SOURCE_DIR}")

# -----------------------------------------------------------------------------
# Target library linking
# -----------------------------------------------------------------------------
target_link_libraries(${IHP_PLUGIN} PRIVATE Qt6::Widgets)
target_link_libraries(${IHP_PLUGIN} PRIVATE Qt6::WebEngineWidgets)

target_link_libraries(${IHP_PLUGIN} PRIVATE nlohmann_json::nlohmann_json)
target_link_libraries(${IHP_PLUGIN} PRIVATE Eigen3::Eigen)
target_link_libraries(${IHP_PLUGIN} PRIVATE spdlog::spdlog_header_only)
target_link_libraries(${IHP_PLUGIN} PRIVATE hdislimdimensionalityreduction hdislimutils hdislimdata ${CMAKE_DL_LIBS})
target_link_libraries(${IHP_PLUGIN} PRIVATE OpenMP::OpenMP_CXX)

target_link_libraries(${IHP_PLUGIN} PRIVATE ManiVault::Core)
target_link_libraries(${IHP_PLUGIN} PRIVATE ManiVault::PointData)
target_link_libraries(${IHP_PLUGIN} PRIVATE ManiVault::ClusterData)
target_link_libraries(${IHP_PLUGIN} PRIVATE ManiVault::ImageData)

if(UNIX AND NOT APPLE)
   target_link_libraries(${IHP_PLUGIN} PRIVATE TBB::tbb)
endif()

# -----------------------------------------------------------------------------
# Target installation
# -----------------------------------------------------------------------------
install(TARGETS ${IHP_PLUGIN}
    RUNTIME DESTINATION Plugins COMPONENT INTERACTIVE_IHP_PLUGIN # Windows .dll
    LIBRARY DESTINATION Plugins COMPONENT INTERACTIVE_IHP_PLUGIN # Linux/Mac .so
)

add_custom_command(TARGET ${IHP_PLUGIN} POST_BUILD
    COMMAND "${CMAKE_COMMAND}"
    --install ${CMAKE_CURRENT_BINARY_DIR}
    --component INTERACTIVE_IHP_PLUGIN
    --config $<CONFIGURATION>
    --prefix ${ManiVault_INSTALL_DIR}/$<CONFIGURATION>
    --verbose
)

# -----------------------------------------------------------------------------
# Miscellaneous
# -----------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Automatically set the debug environment (command + working directory) for MSVC 
    set_property(TARGET ${IHP_PLUGIN} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY $<IF:$<CONFIG:DEBUG>,${ManiVault_INSTALL_DIR}/Debug,$<IF:$<CONFIG:RELWITHDEBINFO>,${ManiVault_INSTALL_DIR}/RelWithDebInfo,${ManiVault_INSTALL_DIR}/Release>>)
    set_property(TARGET ${IHP_PLUGIN} PROPERTY VS_DEBUGGER_COMMAND $<IF:$<CONFIG:DEBUG>,"${ManiVault_INSTALL_DIR}/Debug/ManiVault Studio.exe",$<IF:$<CONFIG:RELWITHDEBINFO>,"${ManiVault_INSTALL_DIR}/RelWithDebInfo/ManiVault Studio.exe","${ManiVault_INSTALL_DIR}/Release/ManiVault Studio.exe">>)
endif()
