# CUB block_wide_sort 子目录 CMakeLists.txt

# 收集所有源文件
file(GLOB_RECURSE CUDA_SOURCES "*.cu" "*.cpp")
file(GLOB_RECURSE HEADERS "*.h" "*.hpp" "*.cuh")

# 如果有源文件则创建可执行文件
if(CUDA_SOURCES)
    # 创建可执行文件
    add_executable(block_wide_sort
        ${CUDA_SOURCES}
        ${HEADERS}
    )

    # 设置 C++ 标准（block_wide_sort.cpp 使用 C++14）
    set_target_properties(block_wide_sort PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
    )

    # 链接 CUDA 运行时库
    target_link_libraries(block_wide_sort PRIVATE
        CUDA::cudart
    )

    # 设置包含目录（包含 CUB 头文件）
    # CUB 库位于 /usr/local/cuda-11.4/targets/x86_64-linux/include/
    target_include_directories(block_wide_sort PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
        /usr/local/cuda-11.4/targets/x86_64-linux/include
    )

    # 设置 CUDA 特定属性
    # 对于 .cpp 文件包含 CUDA 代码的情况，需要启用 CUDA 语言支持
    set_target_properties(block_wide_sort PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # 将包含 CUDA 代码的 .cpp 文件标记为 CUDA 语言
    # CMake 3.18+ 会自动检测，但我们可以显式设置以确保正确编译
    foreach(SOURCE_FILE ${CUDA_SOURCES})
        get_filename_component(EXT ${SOURCE_FILE} EXT)
        if(EXT STREQUAL ".cpp")
            set_source_files_properties(${SOURCE_FILE} PROPERTIES LANGUAGE CUDA)
        endif()
    endforeach()

    # 添加编译选项
    target_compile_options(block_wide_sort PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )

    message(STATUS "CUB block_wide_sort: 已配置")
else()
    message(WARNING "CUB: 未找到源文件")
endif()
