#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <cuda_runtime.h>

// 默认配置（与 Matmul 风格一致）
#define HISTOGRAM_N (1024 * 1024) // 输入元素个数
#define HISTOGRAM_NUM_BINS 256    // bin 个数，输入值范围 [0, numBins)

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

// 朴素 Histogram kernel：每个线程处理一个元素，用 atomicAdd 写入对应 bin
__global__ void histogramNaive(const unsigned int *__restrict__ input,
                               unsigned int *__restrict__ hist, int n,
                               int numBins);

__global__ void histogramShared(const unsigned int *__restrict__ input,
                                unsigned int *__restrict__ hist, int n,
                                int numBins);

// CPU 参考实现（用于验证）
void histogramCPU(const unsigned int *input, unsigned int *hist, int n,
                  int numBins);

bool verifyHistogram(const unsigned int *hist_gpu, const unsigned int *hist_cpu,
                     int numBins);

void launchHistogramNaive(const unsigned int *d_input, unsigned int *d_hist,
                          int n, int numBins);

void launchHistogramDeviceCub(const unsigned int *d_input, unsigned int *d_hist,
                              int n, int numBins);

void launchHistogramShared(const unsigned int *d_input, unsigned int *d_hist,
                           int n, int numBins);
#endif // HISTOGRAM_HPP
