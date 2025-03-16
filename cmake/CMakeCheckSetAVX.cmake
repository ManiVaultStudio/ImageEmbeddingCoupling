# -----------------------------------------------------------------------------
# Check for and link to AVX instruction sets
# -----------------------------------------------------------------------------
# usage: 
#  ihp_check_and_set_AVX(${TARGET} ${USE_AVX})
#  ihp_check_and_set_AVX(${TARGET} ${USE_AVX} 1)    # optional argument, only use AVX (not AVX2 even if available)

macro(ihp_check_and_set_AVX target useavx)
    message(STATUS "Set instruction sets for ${target}, USE_AVX is ${useavx}")

    if(${useavx})
        # Use cmake hardware checks to see whether AVX should be activated
        include(CheckCXXCompilerFlag)

        if(MSVC)
            set(Check_AXV_CompileOption /arch:AVX)
            set(Check_AXV2_CompileOption /arch:AVX2)
            set(Set_AXV_CompileOption /arch:AVX)
            set(Set_AXV2_CompileOption /arch:AVX2)
        else()
            set(Check_AXV_CompileOption -mavx)
            set(Check_AXV2_CompileOption -mavx2)
            set(Set_AXV_CompileOption -mavx -mfma -ftree-vectorize)
            set(Set_AXV2_CompileOption -mavx2 -mfma -ftree-vectorize)
        endif()

        if(NOT DEFINED COMPILER_OPT_AVX_SUPPORTED OR NOT DEFINED COMPILER_OPT_AVX2_SUPPORTED)
            check_cxx_compiler_flag(${Check_AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
            check_cxx_compiler_flag(${Check_AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)
        endif()

        if(${COMPILER_OPT_AVX2_SUPPORTED} AND ${ARGC} EQUAL 2)
            message( STATUS "Use AXV2 for ${target}: ${Set_AXV2_CompileOption}")
            target_compile_options(${target} PRIVATE ${Set_AXV2_CompileOption})
        elseif(${COMPILER_OPT_AVX_SUPPORTED})
            message( STATUS "Use AXV for ${target}: ${Set_AXV_CompileOption}")
            target_compile_options(${target} PRIVATE ${Set_AXV_CompileOption})
        endif()
    endif()
endmacro()

macro(checkAVX)
    include(CheckCXXCompilerFlag)
    
    if(MSVC)
        set(AXV_CompileOption "/arch:AVX")
    else()
        set(AXV_CompileOption "-mavx")
    endif()

    if(NOT DEFINED COMPILER_OPT_AVX_SUPPORTED)
        check_cxx_compiler_flag(${AXV_CompileOption} COMPILER_OPT_AVX_SUPPORTED)
    endif()
endmacro()

macro(checkAVX2)
    include(CheckCXXCompilerFlag)
    
    if(MSVC)
        set(AXV2_CompileOption "/arch:AVX2")
    else()
        set(AXV2_CompileOption "-mavx2")
    endif()

    if(NOT DEFINED COMPILER_OPT_AVX2_SUPPORTED)
        check_cxx_compiler_flag(${AXV2_CompileOption} COMPILER_OPT_AVX2_SUPPORTED)
    endif()
endmacro()

# -----------------------------------------------------------------------------
# Sets the optimization level
# -----------------------------------------------------------------------------
macro(ihp_set_optimization_level target level)
    message(STATUS "Set optimization level in release for ${target} to ${level}")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(OPTIMIZATION_LEVEL_FLAG "-O${level}")
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        if(${level} EQUAL 0)
            set(OPTIMIZATION_LEVEL_FLAG "/Od")
        else()
            set(OPTIMIZATION_LEVEL_FLAG "/O${level}")
        endif()
    endif()

    target_compile_options(${target} PRIVATE "$<$<CONFIG:Release>:${OPTIMIZATION_LEVEL_FLAG}>")

    message( STATUS "Optimization level for ${target} (release) is ${OPTIMIZATION_LEVEL_FLAG}")
endmacro()
