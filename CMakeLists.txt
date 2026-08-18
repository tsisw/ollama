cmake_minimum_required(VERSION 3.21)

project(Ollama C CXX)

include(CheckLanguage)
include(GNUInstallDirs)

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(BUILD_SHARED_LIBS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_SYSTEM_NAME Linux)

if (GGML_TSAVORITE AND ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    
    # Use toolbox for ARM toolchain configuration
    if (NOT DEFINED TOOLBOX_DIR)
        message(FATAL_ERROR "TOOLBOX_DIR not defined (pass -DTOOLBOX_DIR=<path> or set TOOLBOX_DIR environment variable)")
    endif()
    
    # Find toolbox package
    set(Toolbox_DIR "${TOOLBOX_DIR}/lib/cmake")
    find_package(Toolbox REQUIRED CONFIG)
    
    # Use toolbox's ARM toolchain file
    set(TOOLBOX_TOOLCHAIN_FILE "${TOOLBOX_DIR}/lib/cmake/toolchains/arm.cmake")
    if (EXISTS "${TOOLBOX_TOOLCHAIN_FILE}")
        # Include the ARM toolchain configuration
        # This sets CMAKE_C_COMPILER, CMAKE_CXX_COMPILER, CMAKE_AR, CMAKE_STRIP, and CMAKE_FIND_ROOT_PATH
        message(STATUS "Using ARM toolchain from toolbox: ${TOOLBOX_DIR}")
        message(STATUS "ARM C compiler: ${CMAKE_C_COMPILER}")
        message(STATUS "ARM CXX compiler: ${CMAKE_CXX_COMPILER}")
    else()
        message(FATAL_ERROR "ARM toolchain file not found at ${TOOLBOX_TOOLCHAIN_FILE}")
    endif()
endif()

set(GGML_BUILD ON)
set(GGML_SHARED ON)
set(GGML_CCACHE ON)
set(GGML_BACKEND_DL ON)
set(GGML_BACKEND_SHARED ON)
set(GGML_SCHED_MAX_COPIES 4)

set(GGML_LLAMAFILE ON)
set(GGML_CUDA_PEER_MAX_BATCH_SIZE 128)
set(GGML_CUDA_GRAPHS ON)
set(GGML_CUDA_FA ON)
set(GGML_CUDA_COMPRESSION_MODE default)

if((CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    OR (NOT CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64|ARM64|ARMv[0-9]+"))
    set(GGML_CPU_ALL_VARIANTS ON)
endif()

if (CMAKE_OSX_ARCHITECTURES MATCHES "x86_64")
    set(CMAKE_BUILD_RPATH "@loader_path")
    set(CMAKE_INSTALL_RPATH "@loader_path")
endif()

set(OLLAMA_BUILD_DIR ${CMAKE_BINARY_DIR}/lib/ollama)
set(OLLAMA ON)
set(OLLAMA_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/lib/ollama/${OLLAMA_RUNNER_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY         ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG   ${OLLAMA_BUILD_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${OLLAMA_BUILD_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cpu/amx)
if (GGML_TSAVORITE)
    if  (NOT DEFINED GGML_TSAVORITE_TARGET)
        set(GGML_TSAVORITE_TARGET "posix")
    endif()
    if (NOT ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        set(GGML_TSAVORITE_TARGET "posix")
    endif()

    if (NOT DEFINED MLIR_COMPILER_DIR)
        if (NOT DEFINED $ENV{MLIR_SDK_VERSION})
            set (MLIR_COMPILER_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.$ENV{SDK_VERSION}/${CMAKE_SYSTEM_PROCESSOR}/compiler)
            message("MLIR_SDK_VERSION not set defaulting to ${MLIR_COMPILER_DIR} ${CMAKE_OSX_ARCHITECTURES} ${CMAKE_SYSTEM_PROCESSOR}")
        else()
            set (MLIR_COMPILER_DIR $ENV{MLIR_SDK_VERSION}/compiler)
        endif()
    endif()

    if (NOT DEFINED RUNTIME_DIR)
        if (NOT DEFINED $ENV{MLIR_SDK_VERSION})
            set (RUNTIME_DIR /proj/rel/sw/tsi-sw/staging/sdk/sdk-r.$ENV{SDK_VERSION}/${CMAKE_SYSTEM_PROCESSOR}/${GGML_TSAVORITE_TARGET}/runtime)
            message("MLIR_SDK_VERSION not set defaulting to ${RUNTIME_DIR} ${CMAKE_OSX_ARCHITECTURES} ${CMAKE_SYSTEM_PROCESSOR}")
        else()
            set (RUNTIME_DIR $ENV{MLIR_SDK_VERSION}/${GGML_TSAVORITE_TARGET}/runtime)
        endif()
    endif()

    if (NOT DEFINED GGML_TSI_KERNEL_DIR)
       set (GGML_TSI_KERNEL_DIR ${CMAKE_SOURCE_DIR}/llama/vendor/ggml-tsi-kernel/${GGML_TSAVORITE_TARGET})
    endif()


    if (${GGML_TSAVORITE_TARGET} STREQUAL fpga)
        set(CMAKE_CROSSCOMPILING ON)
        set(ARCH_FLAGS -march=armv8-a)
        file(GLOB TLIBS  "${RUNTIME_DIR}/lib/*.so" "${GGML_TSI_KERNEL_DIR}/host/*.o" "${RUNTIME_DIR}/../utils/lib/TsavRTShimCAPI.cpp.o")
        # RUNTIME_DIR/lib is a directory shared with other targets' runtime
        # libraries (e.g. libTsavRTPosix.so lives alongside libTsavRTFPGA.so);
        # the glob above pulls in everything in that directory, not just what
        # this target needs. libTsavRTPosix.so and libTsavRTFPGA.so both
        # export a colliding tsi::runtime::TsavRT::allocate symbol, so linking
        # both into the same binary makes symbol resolution non-deterministic
        # (which implementation actually runs can vary by process/ASLR),
        # causing intermittent SIGSEGV crashes deep in TXE op execution.
        # Excluding the other target's runtime library here guarantees only
        # one implementation is ever linked in.
        list(FILTER TLIBS EXCLUDE REGEX "libTsavRTPosix\\.so")
        message("Setting target as fpga")
    elseif (${GGML_TSAVORITE_TARGET} STREQUAL "posix")
        file(GLOB TLIBS  "${RUNTIME_DIR}/lib/*.so" "${GGML_TSI_KERNEL_DIR}/host/*.o" "${MLIR_COMPILER_DIR}/lib/libFFMDeviceShim.so" "${MLIR_COMPILER_DIR}/lib/libTsavRTPosixShimCAPI.so" "${MLIR_COMPILER_DIR}/lib/libtxe-ffm-c-native.so" "${MLIR_COMPILER_DIR}/lib/libtxe-ffm-cpp.so" "${MLIR_COMPILER_DIR}/lib/libtvu_fau_full.so")
        # See the fpga branch above -- RUNTIME_DIR/lib for posix also ships
        # libTsavRTFPGA.so alongside libTsavRTPosix.so; exclude it so the
        # colliding TsavRT::allocate symbol always resolves to the Posix
        # implementation.
        list(FILTER TLIBS EXCLUDE REGEX "libTsavRTFPGA\\.so")
        message("Setting target as posix for tsavorite")
    endif()

    set(GGML_TSAVORITE_TARGET "${GGML_TSAVORITE_TARGET}" CACHE STRING "Target for tsavorite")
    set (TSAVORITE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/ggml/src/ggml-tsavorite/include)

    include_directories(${TSAVORITE_INCLUDE_DIR})
    include_directories(${MLIR_COMPILER_DIR}/include/tsi-rt/shim)
    include_directories(${RUNTIME_DIR}/include)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-tsavorite/include)
    include_directories(${MLIR_COMPILER_DIR}/../toolbox/build/install-fpga/include)
    message("tsavorite backend is enabled")
endif()

add_compile_definitions(NDEBUG GGML_VERSION=0x0 GGML_COMMIT=0x0)

set(GGML_CPU ON)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src)
set_property(TARGET ggml PROPERTY EXCLUDE_FROM_ALL TRUE)

if (GGML_TSAVORITE AND ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
    add_definitions(-DGGML_USE_TSAVORITE)
endif()
if (NOT GGML_TSAVORITE AND ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
get_target_property(CPU_VARIANTS ggml-cpu MANUALLY_ADDED_DEPENDENCIES)
if(NOT CPU_VARIANTS)
    set(CPU_VARIANTS "ggml-cpu")
endif()

install(TARGETS ggml-base ${CPU_VARIANTS}
    RUNTIME_DEPENDENCIES
        PRE_EXCLUDE_REGEXES ".*"
    RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
    FRAMEWORK DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CPU
)
endif()

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24" AND NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native")
    endif()

    find_package(CUDAToolkit)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-cuda)
    install(TARGETS ggml-cuda
        RUNTIME_DEPENDENCIES
            DIRECTORIES ${CUDAToolkit_BIN_DIR} ${CUDAToolkit_BIN_DIR}/x64 ${CUDAToolkit_LIBRARY_DIR}
            PRE_INCLUDE_REGEXES cublas cublasLt cudart
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT CUDA
    )
endif()

set(WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX "^gfx(908|90a|1200|1201):xnack[+-]$"
    CACHE STRING
    "Regular expression describing AMDGPU_TARGETS not supported on Windows. Override to force building these targets. Default \"^gfx(908|90a|1200|1201):xnack[+-]$\"."
)

check_language(HIP)
if(CMAKE_HIP_COMPILER)
    set(HIP_PLATFORM "amd")

    if(NOT AMDGPU_TARGETS)
        find_package(hip REQUIRED)
        list(FILTER AMDGPU_TARGETS INCLUDE REGEX "^gfx(94[012]|101[02]|1030|110[012]|120[01])$")
    endif()

    if(WIN32 AND WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX)
        list(FILTER AMDGPU_TARGETS EXCLUDE REGEX ${WINDOWS_AMDGPU_TARGETS_EXCLUDE_REGEX})
    endif()

    if(AMDGPU_TARGETS)
        find_package(hip REQUIRED)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-hip)

        if (WIN32)
            target_compile_definitions(ggml-hip PRIVATE GGML_CUDA_NO_PEER_COPY)
        endif()

        target_compile_definitions(ggml-hip PRIVATE GGML_HIP_NO_VMM)

        install(TARGETS ggml-hip
            RUNTIME_DEPENDENCY_SET rocm
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
        )
        install(RUNTIME_DEPENDENCY_SET rocm
                DIRECTORIES ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR}
                PRE_INCLUDE_REGEXES hipblas rocblas amdhip64 rocsolver amd_comgr hsa-runtime64 rocsparse tinfo rocprofiler-register drm drm_amdgpu numa elf
                PRE_EXCLUDE_REGEXES ".*"
                POST_EXCLUDE_REGEXES "system32"
            RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
            LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP
        )

        foreach(HIP_LIB_BIN_INSTALL_DIR IN ITEMS ${HIP_BIN_INSTALL_DIR} ${HIP_LIB_INSTALL_DIR})
            if(EXISTS ${HIP_LIB_BIN_INSTALL_DIR}/rocblas)
                install(DIRECTORY ${HIP_LIB_BIN_INSTALL_DIR}/rocblas DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT HIP)
                break()
            endif()
        endforeach()
    endif()
endif()

find_package(Vulkan)
if(Vulkan_FOUND)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/ml/backend/ggml/ggml/src/ggml-vulkan)
    install(TARGETS ggml-vulkan
        RUNTIME_DEPENDENCIES
            PRE_INCLUDE_REGEXES vulkan
            PRE_EXCLUDE_REGEXES ".*"
        RUNTIME DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
        LIBRARY DESTINATION ${OLLAMA_INSTALL_DIR} COMPONENT Vulkan
    )
endif()


set(GO_EXECUTABLE /proj/local/go/bin/go CACHE FILEPATH "Go executable")

if (GGML_TSAVORITE_TARGET STREQUAL "posix")

  add_custom_target(ollama ALL
      COMMAND
          ${CMAKE_COMMAND} -E env
          CGO_ENABLED=1
          GOOS=linux
          GOARCH=amd64
          CC=
          CXX=
          ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/ollama .
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Building Go binary for x86_64"
  )

elseif (GGML_TSAVORITE_TARGET STREQUAL "fpga")

  add_custom_target(ollama ALL
      COMMAND
          ${CMAKE_COMMAND} -E env
          CGO_ENABLED=1
          GOOS=linux
          GOARCH=arm64
          CC=${CMAKE_C_COMPILER}
          CXX=${CMAKE_CXX_COMPILER}
          ${GO_EXECUTABLE} build -o ${CMAKE_BINARY_DIR}/ollama .
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Building Go binary for ARM64"
  )

endif()
