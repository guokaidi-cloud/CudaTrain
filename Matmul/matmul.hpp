#ifndef MATMUL_HPP
#define MATMUL_HPP

#include <cuda_runtime.h>

// ==================== 宏定义 ====================

// 矩阵大小
#define N 2048
#define BLOCK_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// 1D Block Tiling 参数
#define BM 64 // Block 处理的行数
#define BN 64 // Block 处理的列数
#define BK 8  // 沿 K 维度的分块大小
#define TM 8  // 每个线程计算的行数

// 2D Block Tiling 参数
#define BM2 128
#define BN2 128
#define BK2 8
#define TM2 8
#define TN2 8

// Vectorized 2D Block Tiling 参数
#define BMV 128
#define BNV 128
#define BKV 16
#define TMV 8
#define TNV 8

// Warp Tiling 参数
#define BMW 128
#define BNW 128
#define BKW 16
#define WM 32               // Warp 处理的行数
#define WN 64               // Warp 处理的列数
#define WSUBM 16            // Warp 子块行数
#define WSUBN 32            // Warp 子块列数
#define TMW 4               // 每个线程计算的行数
#define TNW 4               // 每个线程计算的列数
#define WMITER (WM / WSUBM) // Warp 行方向迭代次数: 2
#define WNITER (WN / WSUBN) // Warp 列方向迭代次数: 2
#define WARPSIZE 32
// 线程数: (WSUBM/TMW) * (WSUBN/TNW) = 4*8 = 32 (正好一个 warp)
// warp 数: (BMW/WM) * (BNW/WN) = 4*2 = 8 warps
// 总线程数: 8*32 = 256 threads

// ==================== Kernel 声明 ====================

// 朴素的矩阵乘法 kernel（非 coalesced 版本）
// 使用二维 block (32×32)，warp 内 threadIdx.x 连续变化，threadIdx.y 相同
// 同一 warp 内: x (行索引) 连续变化，y (列索引) 相同
// 写入 C[x * k + y] 时，x 变化导致地址间隔为 k，内存访问不连续
__global__ void matrixMulNaive(float *A, float *B, float *C, int k);

// Global Memory Coalescing 优化版本（coalesced 版本）
// 使用一维 block (1024 线程)，通过 threadIdx.x 计算 x 和 y
// 同一 warp 内: threadIdx.x=0-31，threadIdx.x/32=0，threadIdx.x%32=0-31
// 所以 x (行索引) 相同，y (列索引) 连续变化
// 写入 C[x * k + y] 时，y 连续导致地址连续，内存访问合并
__global__ void matrixMulGlobalMemoryCoalescing(float *A, float *B, float *C,
                                                int k);

// Shared Memory Cache-Blocking 优化版本
// 将数据块加载到共享内存，减少全局内存访问次数
// 方阵版本：A(k×k) × B(k×k) = C(k×k)
__global__ void matrixMulSharedMemory(float *A, float *B, float *C, int k);

// 1D Block Tiling 优化版本
// 每个线程计算 TM 个输出元素，提高计算密度
// 方阵版本：A(k×k) × B(k×k) = C(k×k)
__global__ void matrixMul1DBlockTiling(float *A, float *B, float *C, int k);

// 2D Block Tiling 优化版本
// 每个线程计算 TM×TN 个输出元素（2D 区域）
// 使用多次加载填充共享内存
__global__ void matrixMul2DBlockTiling(float *A, float *B, float *C, int k);

// Vectorized 2D Block Tiling 优化版本
// 使用 float4 向量化加载，提高内存带宽利用率
// A 在加载时转置，优化后续访问模式
__global__ void matrixMulVectorized(float *A, float *B, float *C, int k);

// Warp Tiling 优化版本
// 在 warp 级别进行 tiling，为 Tensor Core 优化做准备
__global__ void matrixMulWarpTiling(float *A, float *B, float *C, int k);

// ==================== 辅助函数声明 ====================

// CPU 矩阵乘法（用于验证）
void matrixMulCPU(float *A, float *B, float *C, int k);

// 验证结果（使用相对误差，对大矩阵更合理）
bool verifyResult(float *C_gpu, float *C_cpu, int k);

// ==================== Kernel 启动器函数声明 ====================
// 这些函数封装了 CUDA kernel 调用，可以从纯 C++ 代码中调用

// 朴素版本启动器
void launchMatrixMulNaive(float *d_A, float *d_B, float *d_C, int k);

// Global Memory Coalescing 版本启动器
void launchMatrixMulCoalescing(float *d_A, float *d_B, float *d_C, int k);

// Shared Memory 版本启动器
void launchMatrixMulSharedMemory(float *d_A, float *d_B, float *d_C, int k);

// 1D Block Tiling 版本启动器
void launchMatrixMul1DBlockTiling(float *d_A, float *d_B, float *d_C, int k);

// 2D Block Tiling 版本启动器
void launchMatrixMul2DBlockTiling(float *d_A, float *d_B, float *d_C, int k);

// Vectorized 版本启动器
void launchMatrixMulVectorized(float *d_A, float *d_B, float *d_C, int k);

// Warp Tiling 版本启动器
void launchMatrixMulWarpTiling(float *d_A, float *d_B, float *d_C, int k);

// GPU Warmup - 运行所有 kernel 一次
void warmupAllKernels(float *d_A, float *d_B, float *d_C, int k);

#endif // MATMUL_HPP
