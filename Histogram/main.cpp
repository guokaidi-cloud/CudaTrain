#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "histogram.hpp"
#include "utils/cuda_utils.hpp"
#include "utils/device_info.hpp"
#include "utils/timer.hpp"

int main() {
  printDeviceInfo();

  const int n = HISTOGRAM_N;
  const int numBins = HISTOGRAM_NUM_BINS;

  printf("=== CUDA Histogram 示例（朴素实现，参考 Matmul 风格）===\n");
  printf("输入长度: %d, Bins: %d\n", n, numBins);

  size_t inputBytes = n * sizeof(unsigned int);
  size_t histBytes = numBins * sizeof(unsigned int);

  unsigned int *h_input = (unsigned int *)malloc(inputBytes);
  unsigned int *h_hist = (unsigned int *)malloc(histBytes);
  unsigned int *h_hist_ref = (unsigned int *)malloc(histBytes);

  for (int i = 0; i < n; ++i) {
    h_input[i] = (unsigned int)(rand() % numBins);
  }

  unsigned int *d_input;
  unsigned int *d_hist;
  CUDA_CHECK(cudaMalloc(&d_input, inputBytes));
  CUDA_CHECK(cudaMalloc(&d_hist, histBytes));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_hist, 0, histBytes));

  Timer timer;

  timer.start_gpu();
  launchHistogramNaive(d_input, d_hist, n, numBins);
  timer.stop_gpu();
  timer.duration_gpu("朴素 Histogram 耗时");

  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, histBytes, cudaMemcpyDeviceToHost));

  histogramCPU(h_input, h_hist_ref, n, numBins);

  printf("验证朴素版本: ");
  if (verifyHistogram(h_hist, h_hist_ref, numBins)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试 launchHistogramDeviceCub ==========
  printf("\n--- 测试 CUB Histogram (launchHistogramCub) ---\n");
  CUDA_CHECK(cudaMemset(d_hist, 0, histBytes));
  timer.start_gpu();
  launchHistogramDeviceCub(d_input, d_hist, n, numBins);
  timer.stop_gpu();
  timer.duration_gpu("CUB Histogram 耗时");
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, histBytes, cudaMemcpyDeviceToHost));
  printf("验证 CUB 版本: ");
  if (verifyHistogram(h_hist, h_hist_ref, numBins)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  // ========== 测试 launchHistogramShared ==========
  printf("\n--- 测试 Shared Histogram (launchHistogramShared) ---\n");
  CUDA_CHECK(cudaMemset(d_hist, 0, histBytes));
  timer.start_gpu();
  launchHistogramShared(d_input, d_hist, n, numBins);
  timer.stop_gpu();
  timer.duration_gpu("Shared Histogram 耗时");
  CUDA_CHECK(cudaMemcpy(h_hist, d_hist, histBytes, cudaMemcpyDeviceToHost));
  printf("验证 Shared 版本: ");
  if (verifyHistogram(h_hist, h_hist_ref, numBins)) {
    printf("通过!\n");
  } else {
    printf("失败!\n");
  }

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_hist));
  free(h_input);
  free(h_hist);
  free(h_hist_ref);

  printf("=== 程序结束 ===\n");
  return 0;
}
