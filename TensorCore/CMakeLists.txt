add_executable(tensorcore_demo demo.cu)

target_link_libraries(tensorcore_demo PRIVATE CUDA::cudart)

target_compile_options(tensorcore_demo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# WMMA 需要 Volta+（sm_70）。父工程若为 52 等旧架构，设备端不会展开 nvcuda::wmma。
# 配置：cmake -B build -DTENSORCORE_CUDA_ARCH=86
if(NOT DEFINED TENSORCORE_CUDA_ARCH)
    set(TENSORCORE_CUDA_ARCH "75" CACHE STRING "TensorCore demo GPU arch number only, e.g. 70,75,80,86,89 (>=70)")
endif()

set_target_properties(tensorcore_demo PROPERTIES
    CUDA_ARCHITECTURES "${TENSORCORE_CUDA_ARCH}"
)
