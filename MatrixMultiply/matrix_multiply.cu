#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// 矩阵大小
#define N 1024
#define BLOCK_SIZE 16

// CUDA 错误检查宏
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t error = call;                                                  \
    if (error != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);              \
      fprintf(stderr, "code: %d, reason: %s\n", error,                         \
              cudaGetErrorString(error));                                      \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// 朴素的矩阵乘法 kernel
__global__ void matrixMulNaive(float *A, float *B, float *C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n && col < n) {
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
      sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
  }
}

// 使用共享内存的优化矩阵乘法 kernel
__global__ void matrixMulShared(float *A, float *B, float *C, int n) {
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * BLOCK_SIZE + ty;
  int col = bx * BLOCK_SIZE + tx;

  float sum = 0.0f;

  // 分块计算
  for (int m = 0; m < (n + BLOCK_SIZE - 1) / BLOCK_SIZE; m++) {
    // 加载数据到共享内存
    if (row < n && (m * BLOCK_SIZE + tx) < n) {
      As[ty][tx] = A[row * n + m * BLOCK_SIZE + tx];
    } else {
      As[ty][tx] = 0.0f;
    }

    if (col < n && (m * BLOCK_SIZE + ty) < n) {
      Bs[ty][tx] = B[(m * BLOCK_SIZE + ty) * n + col];
    } else {
      Bs[ty][tx] = 0.0f;
    }

    __syncthreads();

    // 计算部分和
    for (int k = 0; k < BLOCK_SIZE; k++) {
      sum += As[ty][k] * Bs[k][tx];
    }

    __syncthreads();
  }

  if (row < n && col < n) {
    C[row * n + col] = sum;
  }
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

  // 创建CUDA事件用于计时
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  // 运行朴素版本
  CUDA_CHECK(cudaEventRecord(start));
  matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float naiveTime;
  CUDA_CHECK(cudaEventElapsedTime(&naiveTime, start, stop));
  printf("朴素版本耗时: %.3f ms\n", naiveTime);

  // 运行共享内存优化版本
  CUDA_CHECK(cudaEventRecord(start));
  matrixMulShared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float sharedTime;
  CUDA_CHECK(cudaEventElapsedTime(&sharedTime, start, stop));
  printf("共享内存版本耗时: %.3f ms\n", sharedTime);
  printf("加速比: %.2fx\n", naiveTime / sharedTime);

  // 拷贝结果回主机
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

  // 验证结果（对于小矩阵）
  if (N <= 512) {
    printf("正在验证结果...\n");
    matrixMulCPU(h_A, h_B, h_C_ref, N);
    if (verifyResult(h_C, h_C_ref, N)) {
      printf("验证通过!\n");
    }
  }

  // 清理
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
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
