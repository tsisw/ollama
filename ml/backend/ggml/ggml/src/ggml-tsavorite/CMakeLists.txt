message(STATUS "Tsavorite framework is found")
#
# tsavorite Kernel Library
add_compile_options(--std=c++20)
include_directories(../ggml-cpu)
ggml_add_backend_library(ggml-tsavorite
                         ggml-tsavorite.cpp
                        )

# Add compiler lib directory to link path for FFM libraries
if(DEFINED MLIR_COMPILER_DIR)
    target_link_directories(ggml-tsavorite PRIVATE "${MLIR_COMPILER_DIR}/lib")
endif()


if (NOT ${GGML_TSAVORITE_TARGET} STREQUAL fpga)
target_link_libraries(ggml-tsavorite PRIVATE ${TLIBS} dl rt)
else()

message(STATUS "OLLAMA = ${OLLAMA}")

if(OLLAMA)
target_link_libraries(ggml-tsavorite
  PRIVATE
    -Wl,--no-as-needed
    ${CMAKE_BINARY_DIR}/lib/ollama/libggml-cpu.so
    ${TLIBS}
    dl
    rt
)
else()
target_link_libraries(ggml-tsavorite
  PRIVATE
    -Wl,--no-as-needed
    ggml-cpu
    ${TLIBS}
    dl
    rt
)
endif()
set_target_properties(ggml-tsavorite PROPERTIES
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib/ollama"
)
endif()

