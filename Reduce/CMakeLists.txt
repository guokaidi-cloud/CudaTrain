# Reduce 子目录 CMakeLists.txt

# 收集所有CUDA源文件
file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h" "*.hpp" "*.cuh")

# 如果有源文件则创建可执行文件
if(CUDA_SOURCES OR CPP_SOURCES)
    add_executable(reduce
        ${CUDA_SOURCES}
        ${CPP_SOURCES}
        ${HEADERS}
    )

    # 链接CUDA运行时库和工具库
    target_link_libraries(reduce PRIVATE
        CUDA::cudart
        cuda_utils
    )

    # 设置包含目录
    target_include_directories(reduce PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # 设置CUDA特定属性
    set_target_properties(reduce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )

    # 添加 ptxas 详细输出选项（显示寄存器、共享内存使用信息）
    target_compile_options(reduce PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
    )
else()
    message(WARNING "Reduce: 未找到源文件，请添加 .cu 或 .cpp 文件")
endif()
