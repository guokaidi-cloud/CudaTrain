#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "utils/cuda_utils.h"
#include "utils/device_info.h"
#include "utils/timer.hpp"

// 矩阵大小
#define N 2048
#define BLOCK_SIZE 32
#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// 朴素的矩阵乘法 kernel（非 coalesced 版本）
// 同一 warp 内: row 连续变化，col 相同
// 写入 C[row * k + col] 时地址间隔为 k，内存访问不连续
__global__ void matrixMulNaive(float *A, float *B, float *C, int k) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < k && y < k) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += A[x * k + i] * B[i * k + y];
    }
    C[x * k + y] = sum;
  }
}

// Global Memory Coalescing 优化版本（coalesced 版本）
// 同一 warp 内: row 相同，col 连续变化
// 写入 C[row * k + col] 时地址连续，内存访问合并
__global__ void matrixMulGlobalMemoryCoalescing(float *A, float *B, float *C,
                                                int k) {
  const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

  if (x < k && y < k) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += A[x * k + i] * B[i * k + y];
    }
    C[x * k + y] = sum;
  }
}

// Shared Memory Cache-Blocking 优化版本
// 将数据块加载到共享内存，减少全局内存访问次数
// 方阵版本：A(n×n) × B(n×n) = C(n×n)
__global__ void matrixMulSharedMemory(float *A, float *B, float *C, int n) {
  // 共享内存缓存
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  // 当前 block 在 C 中的位置
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // 当前线程在 block 内的位置
  const int threadRow = threadIdx.y;
  const int threadCol = threadIdx.x;

  // 移动指针到当前 block 的起始位置
  A += cRow * BLOCK_SIZE * n; // A 的第 cRow*BLOCK_SIZE 行
  B += cCol * BLOCK_SIZE;     // B 的第 cCol*BLOCK_SIZE 列
  C += cRow * BLOCK_SIZE * n + cCol * BLOCK_SIZE; // C 的对应位置

  float tmp = 0.0f;

  // 沿着 n 维度分块迭代
  for (int bkIdx = 0; bkIdx < n; bkIdx += BLOCK_SIZE) {
    // 将 A 和 B 的一个 block 加载到共享内存
    // threadCol 作为连续索引，实现 coalesced 访问
    As[threadRow][threadCol] = A[threadRow * n + threadCol];
    Bs[threadRow][threadCol] = B[threadRow * n + threadCol];

    // 等待所有线程加载完成
    __syncthreads();

    // 移动 A 和 B 指针到下一个 block
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * n;

    // 计算当前 block 的点积
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
    }

    // 等待所有线程计算完成，再加载下一个 block
    __syncthreads();
  }

  // 写回结果
  C[threadRow * n + threadCol] = tmp;
}

// CPU 矩阵乘法（用于验证）
void matrixMulCPU(float *A, float *B, float *C, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      float sum = 0.0f;
      for (int k = 0; k < n; k++) {
        sum += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = sum;
    }
  }
}

// 验证结果
bool verifyResult(float *C_gpu, float *C_cpu, int n) {
  float epsilon = 1e-3f;
  for (int i = 0; i < n * n; i++) {
    if (fabs(C_gpu[i] - C_cpu[i]) > epsilon) {
      printf("验证失败: C_gpu[%d] = %f, C_cpu[%d] = %f\n", i, C_gpu[i], i,
             C_cpu[i]);
      return false;
    }
  }
  return true;
}

int main() {
  // 打印 GPU 设备信息
  printDeviceInfo();

  printf("=== CUDA 矩阵乘法示例 ===\n");
  printf("矩阵大小: %d x %d\n", N, N);

  // 分配主机内存
  size_t bytes = N * N * sizeof(float);
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);
  float *h_C_ref = (float *)malloc(bytes);

  // 初始化矩阵
  for (int i = 0; i < N * N; i++) {
    h_A[i] = (float)(rand() % 100) / 100.0f;
    h_B[i] = (float)(rand() % 100) / 100.0f;
  }

  // 分配设备内存
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));

  // 拷贝数据到设备
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  // 配置kernel参数
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // Coalescing 版本使用一维 block
  dim3 blockDim1D(BLOCK_SIZE * BLOCK_SIZE);
  dim3 gridDim1D((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

  // 使用 Timer 计时
  Timer timer;

  // GPU Warmup - 避免首次运行的额外开销（JIT编译、内存分配等）
  printf("GPU Warmup...\n");
  matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  matrixMulGlobalMemoryCoalescing<<<gridDim1D, blockDim1D>>>(d_A, d_B, d_C, N);
  matrixMulSharedMemory<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  cudaDeviceSynchronize();

  // 运行朴素版本
  timer.start_gpu();
  matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("朴素版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // CPU 计算参考结果
  printf("正在计算CPU参考结果...\n");
  matrixMulCPU(h_A, h_B, h_C_ref, N);

  // 验证朴素版本
  printf("验证朴素版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 Global Memory Coalescing 版本
  timer.start_gpu();
  matrixMulGlobalMemoryCoalescing<<<gridDim1D, blockDim1D>>>(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("Global Memory Coalescing版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 Global Memory Coalescing 版本
  printf("验证Coalescing版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 Shared Memory Cache-Blocking 版本
  timer.start_gpu();
  matrixMulSharedMemory<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("Shared Memory版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 Shared Memory 版本
  printf("验证SharedMemory版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 清理
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  printf("=== 程序结束 ===\n");
  return 0;
}
