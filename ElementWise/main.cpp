#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "add.hpp"
#include "utils/device_info.hpp"
#include "utils/timer.hpp"

int main() {
  // 打印 GPU 设备信息
  printDeviceInfo();

  printf("=== CUDA ElementWise Add 算子示例 ===\n");
  printf("数组大小: %d 元素\n", N);

  // 分配主机内存
  size_t bytes = N * sizeof(float);
  float *h_a = (float *)malloc(bytes);
  float *h_b = (float *)malloc(bytes);
  float *h_c_naive = (float *)malloc(bytes);
  float *h_c_float4 = (float *)malloc(bytes);
  float *h_c_cpu = (float *)malloc(bytes);

  // 初始化数组
  for (int i = 0; i < N; i++) {
    h_a[i] = (float)(rand() % 100) / 100.0f;
    h_b[i] = (float)(rand() % 100) / 100.0f;
  }

  // 分配设备内存
  float *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // 拷贝数据到设备
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // 使用 Timer 计时
  Timer timer;

  // CPU 计算参考结果
  printf("正在计算CPU参考结果...\n");
  elementwiseAddCPU(h_a, h_b, h_c_cpu, N);

  // GPU Warmup - 避免首次运行的额外开销
  printf("GPU Warmup...\n");
  launchElementwiseAddNaive(d_a, d_b, d_c, N);
  launchElementwiseAddFloat4(d_a, d_b, d_c, N);
  launchElementwiseAddFloat4Aligned(d_a, d_b, d_c, N);
  launchElementwiseAddFloat4X2(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  // ========== 测试朴素版本 ==========
  printf("\n--- 测试朴素版本 ---\n");
  CUDA_CHECK(cudaMemset(d_c, 0, bytes));
  timer.start_gpu();
  launchElementwiseAddNaive(d_a, d_b, d_c, N);
  timer.stop_gpu();
  timer.duration_gpu("朴素版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_c_naive, d_c, bytes, cudaMemcpyDeviceToHost));

  // 验证朴素版本
  printf("验证朴素版本: ");
  if (verifyResult(h_c_naive, h_c_cpu, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
    return 1;
  }

  // ========== 测试 float4 向量化版本 ==========
  printf("\n--- 测试 float4 向量化版本 ---\n");
  CUDA_CHECK(cudaMemset(d_c, 0, bytes));
  timer.start_gpu();
  launchElementwiseAddFloat4(d_a, d_b, d_c, N);
  timer.stop_gpu();
  timer.duration_gpu("float4向量化版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_c_float4, d_c, bytes, cudaMemcpyDeviceToHost));

  // 验证 float4 向量化版本
  printf("验证float4向量化版本: ");
  if (verifyResult(h_c_float4, h_c_cpu, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
    return 1;
  }

  // ========== 测试 float4 向量化对齐版本 ==========
  printf("\n--- 测试 float4 向量化对齐版本（假设数据对齐） ---\n");
  CUDA_CHECK(cudaMemset(d_c, 0, bytes));
  timer.start_gpu();
  launchElementwiseAddFloat4Aligned(d_a, d_b, d_c, N);
  timer.stop_gpu();
  timer.duration_gpu("float4向量化对齐版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_c_float4, d_c, bytes, cudaMemcpyDeviceToHost));

  // 验证 float4 向量化对齐版本
  printf("验证float4向量化对齐版本: ");
  if (verifyResult(h_c_float4, h_c_cpu, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
    return 1;
  }

  // ========== 测试 float4 x2 版本（每个线程处理 8 个元素） ==========
  printf("\n--- 测试 float4 x2 版本（每个线程处理 8 个元素） ---\n");
  CUDA_CHECK(cudaMemset(d_c, 0, bytes));
  timer.start_gpu();
  launchElementwiseAddFloat4X2(d_a, d_b, d_c, N);
  timer.stop_gpu();
  timer.duration_gpu("float4 x2版本耗时");

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_c_float4, d_c, bytes, cudaMemcpyDeviceToHost));

  // 验证 float4 x2 版本
  printf("验证float4 x2版本: ");
  if (verifyResult(h_c_float4, h_c_cpu, N)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
    return 1;
  }

  // 清理
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c_naive);
  free(h_c_float4);
  free(h_c_cpu);

  printf("=== 程序结束 ===\n");
  return 0;
}
