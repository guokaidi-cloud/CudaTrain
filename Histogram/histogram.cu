#include "histogram.hpp"
#include <cstdio>
#include <cstdlib>

#include <cub/cub.cuh>

__inline__ __device__ int warpReduceSum(int val) {
  unsigned mask = __activemask();
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    val += __shfl_down_sync(mask, val, offset);
  }
  return val;
}
// 朴素 Histogram kernel：每个线程处理一个输入元素，atomicAdd 到对应 bin
// 与 Matmul 的 matrixMulNaive 类似，不做 coalescing、shared memory 等优化
__global__ void histogramNaive(const unsigned int *__restrict__ input,
                               unsigned int *__restrict__ hist, int n,
                               int numBins) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n)
    return;

  unsigned int bin = input[idx];
  if (bin < (unsigned int)numBins) {
    atomicAdd(&hist[bin], 1u);
  }
}

// Shared：块内用 shared 做直方图 + atomicAdd
__global__ void histogramShared(const unsigned int *__restrict__ input,
                                unsigned int *__restrict__ hist, int n,
                                int numBins) {
  extern __shared__ unsigned int sharedHist[];

  for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    sharedHist[i] = 0;
  __syncthreads();

  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < n) {
    if (input[tid] < (unsigned int)numBins)
      atomicAdd(&sharedHist[input[tid]], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < numBins; i += blockDim.x)
    atomicAdd(&hist[i], sharedHist[i]);
}

void histogramCPU(const unsigned int *input, unsigned int *hist, int n,
                  int numBins) {
  for (int i = 0; i < numBins; ++i)
    hist[i] = 0;
  for (int i = 0; i < n; ++i) {
    unsigned int b = input[i];
    if (b < (unsigned int)numBins)
      hist[b]++;
  }
}

bool verifyHistogram(const unsigned int *hist_gpu, const unsigned int *hist_cpu,
                     int numBins) {
  for (int i = 0; i < numBins; ++i) {
    if (hist_gpu[i] != hist_cpu[i]) {
      printf("验证失败: bin[%d] gpu=%u cpu=%u\n", i, hist_gpu[i], hist_cpu[i]);
      return false;
    }
  }
  return true;
}

void launchHistogramNaive(const unsigned int *d_input, unsigned int *d_hist,
                          int n, int numBins) {
  const int blockSize = 256;
  const int gridSize = CEIL_DIV(n, blockSize);
  histogramNaive<<<gridSize, blockSize>>>(d_input, d_hist, n, numBins);
}

void launchHistogramDeviceCub(const unsigned int *d_input, unsigned int *d_hist,
                              int n, int numBins) {
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  //注意numBins+1需要加1，因为numBins传进去是10，我们也确实有10个区间
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_input, d_hist, numBins + 1, 0, numBins,
                                      n);

  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                      d_input, d_hist, numBins + 1, 0, numBins,
                                      n);

  cudaFree(d_temp_storage);
}

void launchHistogramShared(const unsigned int *d_input, unsigned int *d_hist,
                           int n, int numBins) {
  const int blockSize = 256;
  const int maxBlocks =
      4096; // 限制 block 数，使 stride < n 时每个线程处理多元素
  const int gridSize = min(CEIL_DIV(n, blockSize), maxBlocks);
  histogramShared<<<gridSize, blockSize, numBins * sizeof(unsigned int)>>>(
      d_input, d_hist, n, numBins);
}