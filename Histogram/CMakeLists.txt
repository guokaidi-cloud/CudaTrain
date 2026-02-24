# Histogram 子目录 CMakeLists.txt（参考 Matmul 结构）

file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h" "*.hpp" "*.cuh")

if(CUDA_SOURCES OR CPP_SOURCES)
    add_executable(histogram
        ${CUDA_SOURCES}
        ${CPP_SOURCES}
        ${HEADERS}
    )

    target_link_libraries(histogram PRIVATE
        CUDA::cudart
        cuda_utils
    )

    target_include_directories(histogram PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    set_target_properties(histogram PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    target_compile_options(histogram PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
else()
    message(WARNING "Histogram: 未找到源文件，请添加 .cu 或 .cpp 文件")
endif()
