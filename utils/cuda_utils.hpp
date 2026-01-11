#ifndef CUDA_UTILS_HPP
#define CUDA_UTILS_HPP

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 日志输出宏
#define LOG(format, ...)                                                       \
  do {                                                                         \
    printf(format "\n", ##__VA_ARGS__);                                        \
  } while (0)

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);              \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#endif // CUDA_UTILS_HPP
