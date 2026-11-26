# Thor：从仓库根目录构建并运行基准测试
# cmake -S . -B build/softmax-thor -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=110
# cmake --build build/softmax-thor --target softmax_row
# ./build/softmax-thor/bin/softmax_row --bench
add_executable(softmax_row softmax_row.cu)
target_link_libraries(softmax_row PRIVATE CUDA::cudart)
