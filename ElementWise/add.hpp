#ifndef ADD_HPP
#define ADD_HPP

#include <cuda_runtime.h>
#include "utils/cuda_utils.hpp"

// ==================== 宏定义 ====================

// 数组大小
#define N 4096 * 4096  // 16M 元素
#define BLOCK_SIZE 256

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// ==================== Kernel 声明 ====================

// 朴素版本：每个线程处理一个元素
__global__ void elementwise_add_naive(float *a, float *b, float *c, int n);

// 优化版本1：使用 float4 向量化，每个线程处理 4 个元素
__global__ void elementwise_add_float4(float *__restrict__ a, 
                                       float *__restrict__ b, 
                                       float *__restrict__ c, 
                                       int n);

// 优化版本2：假设数据已对齐的 float4 版本（避免边界检查）
__global__ void elementwise_add_float4_aligned(float *__restrict__ a, 
                                                 float *__restrict__ b, 
                                                 float *__restrict__ c, 
                                                 int n);

// 优化版本3：每个线程处理 8 个元素（2 个 float4）
__global__ void elementwise_add_float4_x2(float *__restrict__ a, 
                                          float *__restrict__ b, 
                                          float *__restrict__ c, 
                                          int n);

// ==================== 辅助函数声明 ====================

// CPU 参考实现（用于验证）
void elementwiseAddCPU(float *a, float *b, float *c, int n);

// 验证结果
bool verifyResult(float *result_gpu, float *result_cpu, int n,
                  float tolerance = 1e-5f);

// ==================== Kernel 启动器函数声明 ====================

// 朴素版本启动器
void launchElementwiseAddNaive(float *d_a, float *d_b, float *d_c, int n);

// float4 向量化版本启动器
void launchElementwiseAddFloat4(float *d_a, float *d_b, float *d_c, int n);

// float4 向量化对齐版本启动器（假设数据对齐）
void launchElementwiseAddFloat4Aligned(float *d_a, float *d_b, float *d_c, int n);

// float4 向量化 x2 版本启动器（每个线程处理 8 个元素）
void launchElementwiseAddFloat4X2(float *d_a, float *d_b, float *d_c, int n);

#endif  // ADD_HPP

