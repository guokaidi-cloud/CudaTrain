#include "reduce.hpp"
#include "utils/cuda_utils.hpp"
#include <cmath>
#include <cstdio>

// ==================== Kernel 实现 ====================

// 朴素的 reduce kernel（求和）
// 方案1：使用原子操作（简单，但性能差，适合 naive 版本）
// 每个 block 在共享内存中做 reduce，然后使用原子操作累加到全局输出
__global__ void reduceNaive(float *input, float *output, int n) {
  // 共享内存，每个 block 存储一个部分和
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  // 计算全局索引
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 每个线程加载一个元素到共享内存
  sdata[tid] = (idx < n) ? input[idx] : 0.0f;

  // 同步，确保所有数据都加载完成
  __syncthreads();

  // 在共享内存中做 reduce（树形归约）
  // 标准的树形归约：从上往下，每次合并相邻的两半
  for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2*s) == 0) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 第一个线程使用原子操作累加到全局输出
  // 这样所有 block 的结果会自动合并，只需要一个 kernel 调用
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}


// 消除 warp divergence 的 reduce kernel
// 使用从上往下的归约方式，确保同一 warp 内的线程执行相同路径
__global__ void reduceEliminaetWarpDivergence(float *input, float *output, int n) {
  __shared__ float sdata[BLOCK_SIZE];
  int tid = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
  // 每个线程加载一个元素到共享内存
  sdata[tid] = (idx < n) ? input[idx] : 0.0f;
  __syncthreads();

  for(unsigned int s=1; s < blockDim.x; s *= 2) {
    if(threadIdx.x * 2 *s < blockDim.x) {
        int index = threadIdx.x * 2 * s;
        if(index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
    }
    __syncthreads();
  }

  // 第一个线程使用原子操作累加到全局输出
  if (tid == 0) {
    atomicAdd(output, sdata[0]);
  }
}


// 消除 bank conflicts 的 reduce kernel
// 使用从上往下的归约方式，确保同一 warp 内的线程执行相同路径
__global__ void reduceEliminateBankConflicts(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 每个线程加载一个元素到共享内存
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for(unsigned int s= blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
  
    // 第一个线程使用原子操作累加到全局输出
    if (tid == 0) {
      atomicAdd(output, sdata[0]);
    }
}


// 消除并行共享内存的 reduce kernel
// 使用并行共享内存，每个线程处理两个元素，提高内存带宽利用率
__global__ void reduceEliminatePallelSharedMemory(float *input, float *output, int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + tid;  // 每个 block 处理 2*blockDim.x 个元素
    
    // 每个线程处理两个元素，提高内存带宽利用率
    float val1 = (idx < n) ? input[idx] : 0.0f;
    float val2 = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = val1 + val2;
    __syncthreads();

    for(unsigned int s= blockDim.x / 2; s > 0; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
  
    // 第一个线程使用原子操作累加到全局输出
    if (tid == 0) {
      atomicAdd(output, sdata[0]);
    }
}


// 消除 wrap reduce 的 reduce kernel
// 使用 wrap reduce，确保同一 warp 内的线程执行相同路径
__global__ void reduceEliminateWrapReduce(float *input, float *output, int n) {
    volatile __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + tid;  // 每个 block 处理 2*blockDim.x 个元素
    
    // 每个线程处理两个元素，提高内存带宽利用率
    float val1 = (idx < n) ? input[idx] : 0.0f;
    float val2 = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sdata[tid] = val1 + val2;
    __syncthreads();

    // 第一阶段：从 blockDim.x/2 到 32，使用共享内存 reduce（需要同步）
    for(unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if(tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 第二阶段：合并 sdata[0-31] 和 sdata[32-63]
    // 这是跨 warp 操作（warp 0 读取 warp 1 写入的数据），需要同步
    if(tid < 32){
        sdata[tid] += sdata[tid + 32];
    }
    __syncthreads();

    // 第三阶段：在最后一个 warp 内部进行 reduce（不需要同步）
    // 同一 warp 内的线程是隐式同步的，volatile 确保内存可见性
    if(tid < 32){
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
  
    // 第一个线程使用原子操作累加到全局输出
    if (tid == 0) {
      atomicAdd(output, sdata[0]);
    }
}


// 消除 wrap shuffle 的 reduce kernel
// 使用 wrap shuffle，确保同一 warp 内的线程执行相同路径
__global__ void reduceEliminateWrapShuffleV1(float *input, float *output, int n) {
  const int WARP_SIZE = 32;
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  int lane_id = threadIdx.x % WARP_SIZE;
  int warp_id = threadIdx.x / WARP_SIZE;

  float sum = (idx < n) ? input[idx] : 0.0f;

  for(int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }

  if (lane_id == 0) {
    atomicAdd(output, sum);
  }
}

// 消除 wrap shuffle 的 reduce kernel
// 使用 wrap shuffle，确保同一 warp 内的线程执行相同路径
__global__ void reduceEliminateWrapShuffleV2(float *input, float *output, int n) {
    const int WARP_SIZE = 32;
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 2 + tid;  // 每个 block 处理 2*blockDim.x 个元素
    
    // 每个线程处理两个元素，提高内存带宽利用率
    float sum = 0.0f;
    float val1 = (idx < n) ? input[idx] : 0.0f;
    float val2 = (idx + blockDim.x < n) ? input[idx + blockDim.x] : 0.0f;
    sum = val1 + val2;

    sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
    sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);

    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    // 每个 warp 的第一个线程（lane_id == 0）存储该 warp 的结果
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    if (lane_id == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();
    
    // 第一个 warp 负责归约所有 warp 的结果
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
    }

    // 只有第一个 warp 的第一个线程有最终结果
    if(warp_id == 0 && lane_id == 0){
        atomicAdd(output, sum);
    }

}
// ==================== 辅助函数实现 ====================

// CPU reduce（用于验证）
float reduceCPU(float *input, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; i++) {
    sum += input[i];
  }
  return sum;
}

// 验证结果（使用相对误差）
bool verifyResult(float result_gpu, float result_cpu) {
  float absError = fabs(result_gpu - result_cpu);
  float relError = absError / (fabs(result_cpu) + 1e-6f);

  printf("GPU结果: %f, CPU结果: %f, 绝对误差: %f, 相对误差: %.6f%%\n",
         result_gpu, result_cpu, absError, relError * 100);

  // 相对误差超过 0.1% 算失败
  if (relError > 1e-3f) {
    printf("验证失败: 相对误差过大\n");
    return false;
  }
  return true;
}

// ==================== Kernel 启动器函数实现 ====================

// 朴素版本启动器（使用原子操作，只需要一个 kernel 调用）
// 优点：简单，只需要一次 kernel 调用
// 缺点：原子操作会有竞争，性能较差（但适合 naive 版本）
void launchReduceNaive(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceNaive<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminaetWarpDivergence(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminaetWarpDivergence<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminateBankConflicts(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminateBankConflicts<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminatePallelSharedMemory(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  // 每个 block 处理 2 * BLOCK_SIZE 个元素（每个线程处理2个元素）
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE * 2);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminatePallelSharedMemory<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminateWrapReduce(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  // 每个 block 处理 2 * BLOCK_SIZE 个元素（每个线程处理2个元素）
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE * 2);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminateWrapReduce<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminateWrapShuffleV1(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminateWrapShuffleV1<<<numBlocks, blockDim>>>(d_input, d_output, n);
}


void launchReduceEliminateWrapShuffleV2(float *d_input, float *d_output, int n) {
  dim3 blockDim(BLOCK_SIZE);
  // 每个 block 处理 2 * BLOCK_SIZE 个元素（每个线程处理2个元素）
  int numBlocks = CEIL_DIV(n, BLOCK_SIZE * 2);

  // 初始化输出为0
  CUDA_CHECK(cudaMemset(d_output, 0, sizeof(float)));

  // 一个 kernel 调用即可，所有 block 的结果通过原子操作自动合并
  reduceEliminateWrapShuffleV2<<<numBlocks, blockDim>>>(d_input, d_output, n);
}