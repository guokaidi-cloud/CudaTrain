#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "reduce.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/device_info.hpp"
#include "utils/timer.hpp"

int main() {
  // 打印 GPU 设备信息
  printDeviceInfo();

  printf("=== CUDA Reduce 算子示例 ===\n");
  printf("数组大小: %d 元素\n", N);

  // 分配主机内存
  size_t bytes = N * sizeof(float);
  float *h_input = (float *)malloc(bytes);
  float h_output = 0.0f;
  float h_output_ref = 0.0f;

  // 初始化数组
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)(rand() % 100) / 100.0f;
  }

  // 分配设备内存
  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, bytes));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(float)));

  // 拷贝数据到设备
  CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

  // 使用 Timer 计时
  Timer timer;

  // CPU 计算参考结果（只需要计算一次）
  printf("正在计算CPU参考结果...\n");
  h_output_ref = reduceCPU(h_input, N);

  // GPU Warmup
  printf("GPU Warmup...\n");
  launchReduceNaive(d_input, d_output, N);
  launchReduceEliminaetWarpDivergence(d_input, d_output, N);
  launchReduceEliminateBankConflicts(d_input, d_output, N);
  launchReduceEliminatePallelSharedMemory(d_input, d_output, N);
  launchReduceEliminateWrapReduce(d_input, d_output, N);
  launchReduceEliminateWrapShuffle(d_input, d_output, N);
  cudaDeviceSynchronize();

  // ========== 测试朴素版本 ==========
  printf("\n--- 测试朴素版本 ---\n");
  timer.start_gpu();
  launchReduceNaive(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("朴素版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证朴素版本
  printf("验证朴素版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试消除 Warp Divergence 版本 ==========
  printf("\n--- 测试消除 Warp Divergence 版本 ---\n");
  timer.start_gpu();
  launchReduceEliminaetWarpDivergence(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("消除WarpDivergence版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证消除 Warp Divergence 版本
  printf("验证消除WarpDivergence版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试消除 Bank Conflicts 版本 ==========
  printf("\n--- 测试消除 Bank Conflicts 版本 ---\n");
  timer.start_gpu();
  launchReduceEliminateBankConflicts(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("消除BankConflicts版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证消除 Bank Conflicts 版本
  printf("验证消除BankConflicts版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试消除并行共享内存版本 ==========
  printf("\n--- 测试消除并行共享内存版本 ---\n");
  timer.start_gpu();
  launchReduceEliminatePallelSharedMemory(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("消除并行共享内存版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证消除并行共享内存版本
  printf("验证消除并行共享内存版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试消除 Warp Reduce 版本 ==========
  printf("\n--- 测试消除 Warp Reduce 版本 ---\n");
  timer.start_gpu();
  launchReduceEliminateWrapReduce(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("消除WarpReduce版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证消除 Warp Reduce 版本
  printf("验证消除WarpReduce版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试消除 Warp Shuffle 版本 ==========
  printf("\n--- 测试消除 Warp Shuffle 版本 ---\n");
  timer.start_gpu();
  launchReduceEliminateWrapShuffle(d_input, d_output, N);
  timer.stop_gpu();
  timer.duration_gpu("消除WarpShuffle版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(&h_output, d_output, sizeof(float),
                        cudaMemcpyDeviceToHost));

  // 验证消除 Warp Shuffle 版本
  printf("验证消除WarpShuffle版本: ");
  if (verifyResult(h_output, h_output_ref)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // 清理
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  free(h_input);

  printf("=== 程序结束 ===\n");
  return 0;
}
