#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "matmul.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/device_info.hpp"
#include "utils/timer.hpp"

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

  // 初始化 cuBLAS
  cublasHandle_t cublasHandle;
  cublasCreate(&cublasHandle);
  const float alpha = 1.0f;
  const float beta = 0.0f;

  // 使用 Timer 计时
  Timer timer;

  // GPU Warmup - 避免首次运行的额外开销（JIT编译、内存分配等）
  printf("GPU Warmup...\n");
  warmupAllKernels(d_A, d_B, d_C, N);
  // cuBLAS warmup
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N,
              d_A, N, &beta, d_C, N);
  cudaDeviceSynchronize();

  // 运行朴素版本
  timer.start_gpu();
  launchMatrixMulNaive(d_A, d_B, d_C, N);
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
  launchMatrixMulCoalescing(d_A, d_B, d_C, N);
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
  launchMatrixMulSharedMemory(d_A, d_B, d_C, N);
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

  // 运行 1D Block Tiling 版本
  timer.start_gpu();
  launchMatrixMul1DBlockTiling(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("1D Block Tiling版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 1D Block Tiling 版本
  printf("验证1DBlockTiling版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 2D Block Tiling 版本
  timer.start_gpu();
  launchMatrixMul2DBlockTiling(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("2D Block Tiling版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 2D Block Tiling 版本
  printf("验证2DBlockTiling版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 Vectorized 版本
  timer.start_gpu();
  launchMatrixMulVectorized(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("Vectorized版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 Vectorized 版本
  printf("验证Vectorized版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 Warp Tiling 版本
  timer.start_gpu();
  launchMatrixMulWarpTiling(d_A, d_B, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("Warp Tiling版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 Warp Tiling 版本
  printf("验证WarpTiling版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 运行 cuBLAS SGEMM
  // 注意: cuBLAS 使用列优先存储，C = alpha*A*B + beta*C
  // 对于行优先的矩阵，使用 C^T = B^T * A^T，即交换 A 和 B
  timer.start_gpu();
  cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_B, N,
              d_A, N, &beta, d_C, N);
  timer.stop_gpu();
  timer.duration_gpu("cuBLAS SGEMM耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证 cuBLAS 版本
  printf("验证cuBLAS版本: ");
  if (verifyResult(h_C, h_C_ref, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 清理
  cublasDestroy(cublasHandle);
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
