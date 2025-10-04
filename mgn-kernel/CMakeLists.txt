# cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
# project(mgn-kernel LANGUAGES CXX CUDA)

# # CMake
# cmake_policy(SET CMP0169 OLD)
# include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
# set(CMAKE_COLOR_DIAGNOSTICS ON)
# set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# set(CMAKE_SHARED_LIBRARY_PREFIX "")

# # Python
# # SKBUILD_SABI_COMPONENT is typically passed by scikit-build.
# find_package(Python COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT} REQUIRED)

# # CXX
# set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# # CUDA
# enable_language(CUDA)
# find_package(CUDAToolkit REQUIRED)
# set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
#     message("CUDA_VERSION ${CUDA_VERSION} >= 13.0")
# elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
#     message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
# elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
#     message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
# elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
#     message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
# elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
#     message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
# endif()

# # Torch
# find_package(Torch REQUIRED)
# # The 'clear_cuda_arches' function comes from the included utils.cmake.
# # It's used to clean up default CUDA architectures set by PyTorch.
# clear_cuda_arches(CMAKE_FLAG)

# # Dependencies via FetchContent
# include(FetchContent)

# # cutlass
# FetchContent_Declare(
#     repo-cutlass
#     GIT_REPOSITORY https://github.com/NVIDIA/cutlass
#     # Pinned to a specific commit for reproducibility
#     GIT_TAG        f115c3f85467d5d9619119d1dbeb9c03c3d73864
#     GIT_SHALLOW    OFF
# )
# FetchContent_Populate(repo-cutlass)

# # ccache (optional)
# option(ENABLE_CCACHE "Whether to use ccache" ON)
# find_program(CCACHE_FOUND ccache)
# if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
#     message(STATUS "Building with CCACHE enabled")
#     set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
#     set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
# endif()


# # ==================== Project Configuration ====================

# # Enable gencode below SM90
# option(ENABLE_BELOW_SM90 "Enable below SM90" ON)

# # For aarch64, disable older architectures by default as they are often not relevant
# if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
#     set(ENABLE_BELOW_SM90 OFF)
#     message(STATUS "For aarch64, disabling gencode below SM90 by default.")
# endif()

# # Include directories
# include_directories(
#     ${PROJECT_SOURCE_DIR}/include
#     ${PROJECT_SOURCE_DIR}/csrc
#     ${repo-cutlass_SOURCE_DIR}/include
#     ${repo-cutlass_SOURCE_DIR}/tools/util/include
# )


# # Set up CUDA compiler flags
# set(MGN_KERNEL_CUDA_FLAGS
#     "-DNDEBUG"
#     "-DOPERATOR_NAMESPACE=mgn-kernel"
#     "-O3"
#     "-Xcompiler"
#     "-fPIC"
#     "-gencode=arch=compute_90,code=sm_90" # Hopper support by default
#     "-std=c++17"
#     "-DCUTE_USE_PACKED_TUPLE=1"
#     "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
#     "-DCUTLASS_VERSIONS_GENERATED"
#     "-DCUTLASS_TEST_LEVEL=0"
#     "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
#     "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
#     "--expt-relaxed-constexpr"
#     "--expt-extended-lambda"
#     "--threads=32"

#     # Supress warnings
#     "-Xcompiler=-Wconversion"
#     "-Xcompiler=-fno-strict-aliasing"
# )

# option(MGN_KERNEL_ENABLE_SM100A           "Enable SM100A"           OFF)
# option(MGN_KERNEL_ENABLE_SM90A            "Enable SM90A"            OFF)
# option(MGN_KERNEL_ENABLE_BF16             "Enable BF16"             ON)
# option(MGN_KERNEL_ENABLE_FP8              "Enable FP8"              ON)
# option(MGN_KERNEL_ENABLE_FP4              "Enable FP4"              OFF)
# option(MGN_KERNEL_ENABLE_FA3              "Enable FA3"              OFF)


# # Add GPU architecture flags based on options
# if (ENABLE_BELOW_SM90)
#     list(APPEND MGN_KERNEL_CUDA_FLAGS
#         "-gencode=arch=compute_75,code=sm_75"
#         "-gencode=arch=compute_80,code=sm_80"
#         "-gencode=arch=compute_89,code=sm_89"
#     )
# endif()

# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR MGN_KERNEL_ENABLE_SM100A)
#     list(APPEND MGN_KERNEL_CUDA_FLAGS
#         "-gencode=arch=compute_100,code=sm_100"
#         "-gencode=arch=compute_100a,code=sm_100a"
#     )
# endif()

# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR MGN_KERNEL_ENABLE_SM90A)
#     list(APPEND MGN_KERNEL_CUDA_FLAGS
#         "-gencode=arch=compute_100,code=sm_100"
#         "-gencode=arch=compute_100a,code=sm_100a"
#         "-gencode=arch=compute_101,code=sm_101"
#         "-gencode=arch=compute_101a,code=sm_101a"
#         "-gencode=arch=compute_120,code=sm_120"
#         "-gencode=arch=compute_120a,code=sm_120a"
#     )
# else()
#     list(APPEND MGN_KERNEL_CUDA_FLAGS
#         "-use_fast_math"
#     )
# endif()

# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR MGN_KERNEL_ENABLE_SM90A)
#     set(MGN_KERNEL_ENABLE_FA3 ON)
#     list(APPEND MGN_KERNEL_CUDA_FLAGS
#         "-gencode=arch=compute_90a,code=sm_90a"
#     )
# endif()


# if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR MGN_KERNEL_ENABLE_FP4)
#     list(APPEND MGN_KERNEL_CUDA_FLAGS "-DENABLE_NVFP4=1")
# endif()


# # Remove problematic flags that PyTorch might add by default
# string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__"      "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__"    "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
# string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__"     "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")


# # ==================== Build Target ====================

# set(SOURCES
#     "csrc/common_extension.cc"
#     "csrc/moe/moe_fused_gate.cu"
#     "csrc/moe/expert_bin_count.cu"
#     "csrc/moe/fused_moe_token_dispatch.cu"
#     "csrc/elementwise/rmsnorm.cu"
# )

# # Create the Python module
# Python_add_library(common_ops MODULE USE_SABI ${SKBUILD_SABI_VERSION} WITH_SOABI ${SOURCES})

# # Apply the CUDA compile flags to our target
# target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MGN_KERNEL_CUDA_FLAGS}>)

# # Add include directories needed by the target
# target_include_directories(common_ops PRIVATE
#     # These CUTLASS example paths can be useful for reference
#     ${repo-cutlass_SOURCE_DIR}/examples/77_blackwell_fmha
#     ${repo-cutlass_SOURCE_DIR}/examples/common
# )

# find_package(Python3 COMPONENTS Interpreter REQUIRED)

# # Match PyTorch's C++ ABI
# execute_process(
#     COMMAND ${Python3_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
#     OUTPUT_VARIABLE TORCH_CXX11_ABI
#     OUTPUT_STRIP_TRAILING_WHITESPACE
# )
# if(TORCH_CXX11_ABI STREQUAL "0")
#     message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
# else()
#     message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
#     set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
# endif()

# target_compile_options(common_ops PRIVATE ${CXX11_ABI_FLAG})


# # Link against PyTorch and CUDA libraries
# target_link_libraries(common_ops PRIVATE ${TORCH_LIBRARIES} c10 cuda cublas cublasLt)

# # Install the compiled library to the 'mgn_kernel' directory
# install(TARGETS common_ops LIBRARY DESTINATION mgn_kernel)



cmake_minimum_required(VERSION 3.26 FATAL_ERROR)
project(mgn-kernel LANGUAGES CXX CUDA)

# CMake
cmake_policy(SET CMP0169 OLD)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/utils.cmake)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_VERBOSE_MAKEFILE ON CACHE BOOL "ON")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_SHARED_LIBRARY_PREFIX "")

# Python - Find it but don't use Python_add_library
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# CXX
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Detected CUDA_VERSION=${CUDA_VERSION}")
if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "13.0")
    message("CUDA_VERSION ${CUDA_VERSION} >= 13.0")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.8")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.4")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.1")
    message("CUDA_VERSION ${CUDA_VERSION} >= 12.1")
elseif ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "11.8")
    message("CUDA_VERSION ${CUDA_VERSION} >= 11.8")
endif()

# Torch
find_package(Torch REQUIRED)
clear_cuda_arches(CMAKE_FLAG)

# CUTLASS
set(CUTLASS_DIR "${PROJECT_SOURCE_DIR}/csrc/cutlass")
if(NOT EXISTS "${CUTLASS_DIR}/include")
    message(FATAL_ERROR 
        "CUTLASS not found at ${CUTLASS_DIR}. "
        "Please clone it with: git clone https://github.com/NVIDIA/cutlass ${CUTLASS_DIR}")
endif()
message(STATUS "Using local CUTLASS at: ${CUTLASS_DIR}")

# ccache
option(ENABLE_CCACHE "Whether to use ccache" ON)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE AND DEFINED ENV{CCACHE_DIR})
    message(STATUS "Building with CCACHE enabled")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "ccache")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK "ccache")
endif()

# ==================== Project Configuration ====================

option(ENABLE_BELOW_SM90 "Enable below SM90" ON)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ENABLE_BELOW_SM90 OFF)
    message(STATUS "For aarch64, disabling gencode below SM90 by default.")
endif()

# CUDA compiler flags
set(MGN_KERNEL_CUDA_FLAGS
    "-DNDEBUG"
    "-DOPERATOR_NAMESPACE=mgn-kernel"
    "-O3"
    "-Xcompiler"
    "-fPIC"
    "-gencode=arch=compute_90,code=sm_90"
    "-std=c++17"
    "-DCUTE_USE_PACKED_TUPLE=1"
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1"
    "-DCUTLASS_VERSIONS_GENERATED"
    "-DCUTLASS_TEST_LEVEL=0"
    "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1"
    "-DCUTLASS_DEBUG_TRACE_LEVEL=0"
    "--expt-relaxed-constexpr"
    "--expt-extended-lambda"
    "--threads=32"
    "-Xcompiler=-Wconversion"
    "-Xcompiler=-fno-strict-aliasing"
)

option(MGN_KERNEL_ENABLE_SM100A "Enable SM100A" OFF)
option(MGN_KERNEL_ENABLE_SM90A "Enable SM90A" OFF)
option(MGN_KERNEL_ENABLE_BF16 "Enable BF16" ON)
option(MGN_KERNEL_ENABLE_FP8 "Enable FP8" ON)
option(MGN_KERNEL_ENABLE_FP4 "Enable FP4" OFF)
option(MGN_KERNEL_ENABLE_FA3 "Enable FA3" OFF)

if (ENABLE_BELOW_SM90)
    list(APPEND MGN_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_75,code=sm_75"
        "-gencode=arch=compute_80,code=sm_80"
        "-gencode=arch=compute_89,code=sm_89"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR MGN_KERNEL_ENABLE_SM100A)
    list(APPEND MGN_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100,code=sm_100"
        "-gencode=arch=compute_100a,code=sm_100a"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR MGN_KERNEL_ENABLE_SM90A)
    list(APPEND MGN_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_100,code=sm_100"
        "-gencode=arch=compute_100a,code=sm_100a"
        "-gencode=arch=compute_101,code=sm_101"
        "-gencode=arch=compute_101a,code=sm_101a"
        "-gencode=arch=compute_120,code=sm_120"
        "-gencode=arch=compute_120a,code=sm_120a"
    )
else()
    list(APPEND MGN_KERNEL_CUDA_FLAGS
        "-use_fast_math"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.4" OR MGN_KERNEL_ENABLE_SM90A)
    set(MGN_KERNEL_ENABLE_FA3 ON)
    list(APPEND MGN_KERNEL_CUDA_FLAGS
        "-gencode=arch=compute_90a,code=sm_90a"
    )
endif()

if ("${CUDA_VERSION}" VERSION_GREATER_EQUAL "12.8" OR MGN_KERNEL_ENABLE_FP4)
    list(APPEND MGN_KERNEL_CUDA_FLAGS "-DENABLE_NVFP4=1")
endif()

# Remove problematic flags
string(REPLACE "-D__CUDA_NO_HALF_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_BFLOAT16_CONVERSIONS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
string(REPLACE "-D__CUDA_NO_HALF2_OPERATORS__" "" CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

# ==================== Build Target ====================

set(SOURCES
    "csrc/common_extension.cc"
    "csrc/moe/moe_fused_gate.cu"
    "csrc/moe/expert_bin_count.cu"
    "csrc/moe/fused_moe_token_dispatch.cu"
    "csrc/elementwise/rmsnorm.cu"
)

# DON'T use Python_add_library - create a regular MODULE library instead
add_library(common_ops MODULE ${SOURCES})

# Set Python module properties manually
set_target_properties(common_ops PROPERTIES
    PREFIX ""  # No 'lib' prefix
    SUFFIX ".${Python_SOABI}.so"  # Use Python SOABI suffix
)

# Add includes in CORRECT order - Python FIRST as regular includes (NOT SYSTEM)
target_include_directories(common_ops PRIVATE
    ${Python_INCLUDE_DIRS}
)

# Now add Torch includes - but extract them WITHOUT the INTERFACE_SYSTEM property
# Get torch include dirs from the imported target
get_target_property(TORCH_INCLUDE_DIRS_RAW torch INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(common_ops PRIVATE
    ${TORCH_INCLUDE_DIRS_RAW}
)

# Add project includes
target_include_directories(common_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/csrc
    ${CUTLASS_DIR}/include
    ${CUTLASS_DIR}/tools/util/include
    ${CUTLASS_DIR}/examples/77_blackwell_fmha
    ${CUTLASS_DIR}/examples/common
)

# Apply CUDA compile flags
target_compile_options(common_ops PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${MGN_KERNEL_CUDA_FLAGS}>)

# Match PyTorch's C++ ABI
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(TORCH_CXX11_ABI STREQUAL "0")
    message(STATUS "Using old C++ ABI (-D_GLIBCXX_USE_CXX11_ABI=0)")
    target_compile_definitions(common_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=0)
else()
    message(STATUS "Using new C++11 ABI (-D_GLIBCXX_USE_CXX11_ABI=1)")
    target_compile_definitions(common_ops PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
endif()

# Get torch libraries without their interface includes
get_target_property(TORCH_LIBS_RAW torch INTERFACE_LINK_LIBRARIES)

# Link libraries - link to torch's libraries but not the torch target itself
target_link_libraries(common_ops PRIVATE
    ${TORCH_LIBS_RAW}
    c10
    CUDA::cuda_driver
    CUDA::cublas
    CUDA::cublasLt
    ${Python_LIBRARIES}
)

# Install
install(TARGETS common_ops LIBRARY DESTINATION mgn_kernel)