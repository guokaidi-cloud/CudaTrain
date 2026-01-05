# MatrixMultiply 子目录 CMakeLists.txt

# 收集所有CUDA源文件
file(GLOB_RECURSE CUDA_SOURCES "*.cu")
file(GLOB_RECURSE CPP_SOURCES "*.cpp")
file(GLOB_RECURSE HEADERS "*.h" "*.hpp" "*.cuh")

# 如果有源文件则创建可执行文件
if(CUDA_SOURCES OR CPP_SOURCES)
    add_executable(matrix_multiply
        ${CUDA_SOURCES}
        ${CPP_SOURCES}
        ${HEADERS}
    )

    # 链接CUDA运行时库
    target_link_libraries(matrix_multiply PRIVATE
        CUDA::cudart
    )

    # 设置包含目录
    target_include_directories(matrix_multiply PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # 设置CUDA特定属性
    set_target_properties(matrix_multiply PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
else()
    message(WARNING "MatrixMultiply: 未找到源文件，请添加 .cu 或 .cpp 文件")
endif()

