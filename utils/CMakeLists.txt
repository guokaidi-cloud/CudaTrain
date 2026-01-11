# utils 库
add_library(cuda_utils STATIC
    timer.cpp
    device_info.cpp
)

target_include_directories(cuda_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(cuda_utils PUBLIC
    CUDA::cudart
)

