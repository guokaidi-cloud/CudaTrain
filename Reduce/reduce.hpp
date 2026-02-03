#ifndef REDUCE_HPP
#define REDUCE_HPP

#include <cuda_runtime.h>

// ==================== 宏定义 ====================

// 数组大小
#define ARRAY_SIZE (1024 * 1024) // 1M 元素 (1024 * 1024)
#define N ARRAY_SIZE // 兼容旧代码
#define BLOCK_SIZE 256

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// ==================== Kernel 声明 ====================

// 朴素的 reduce kernel（求和）
// 每个线程处理一个元素，然后使用原子操作累加
__global__ void reduceNaive(float *input, float *output, int n);


__global__ void reduceEliminaetWarpDivergence(float *input, float *output, int n);


__global__ void reduceEliminateBankConflicts(float *input, float *output, int n);

__global__ void reduceEliminatePallelSharedMemory(float *input, float *output, int n);


__global__ void reduceEliminateWrapReduce(float *input, float *output, int n);


__global__ void reduceEliminateWrapShuffleV1(float *input, float *output, int n);
__global__ void reduceEliminateWrapShuffleV2(float *input, float *output, int n);


__global__ void reduceBlockCUB(float *input, float *output, int n);
__global__ void reduceBlockCUBV2(float *input, float *output, int n);

// ==================== 辅助函数声明 ====================

// CPU reduce（用于验证）
float reduceCPU(float *input, int n);

// 验证结果（使用相对误差）
bool verifyResult(float result_gpu, float result_cpu);

// ==================== Kernel 启动器函数声明 ====================

// 朴素版本启动器
void launchReduceNaive(float *d_input, float *d_output, int n);

void launchReduceEliminaetWarpDivergence(float *d_input, float *d_output, int n);

void launchReduceEliminateBankConflicts(float *d_input, float *d_output, int n);

void launchReduceEliminatePallelSharedMemory(float *d_input, float *d_output, int n);

void launchReduceEliminateWrapReduce(float *d_input, float *d_output, int n);

void launchReduceEliminateWrapShuffleV1(float *d_input, float *d_output, int n);

void launchReduceEliminateWrapShuffleV2(float *d_input, float *d_output, int n);

void launchreduceBlockCUB(float *d_input, float *d_output, int n);

void launchreduceBlockCUBV2(float *d_input, float *d_output, int n);

#endif // REDUCE_HPP
