#include "matmul.hpp"
#include <cmath>
#include <cstdio>

// ==================== Kernel 实现 ====================

// 朴素的矩阵乘法 kernel（非 coalesced 版本）
// 使用二维 block (32×32)，warp 内 threadIdx.x 连续变化，threadIdx.y 相同
// 同一 warp 内: x (行索引) 连续变化，y (列索引) 相同
// 写入 C[x * k + y] 时，x 变化导致地址间隔为 k，内存访问不连续
__global__ void matrixMulNaive(float *A, float *B, float *C, int k) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x; // 行索引
  const int y = blockIdx.y * blockDim.y + threadIdx.y; // 列索引

  if (x < k && y < k) {
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      sum += A[x * k + i] * B[i * k + y];
    }
    C[x * k + y] = sum;
  }
}

// Global Memory Coalescing 优化版本（coalesced 版本）
// 使用一维 block (1024 线程)，通过 threadIdx.x 计算 x 和 y
// 同一 warp 内: threadIdx.x=0-31，threadIdx.x/32=0，threadIdx.x%32=0-31
// 所以 x (行索引) 相同，y (列索引) 连续变化
// 写入 C[x * k + y] 时，y 连续导致地址连续，内存访问合并
__global__ void matrixMulGlobalMemoryCoalescing(float *A, float *B, float *C,
                                                int k) {
  const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE); // 行索引
  const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE); // 列索引

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
// 方阵版本：A(k×k) × B(k×k) = C(k×k)
__global__ void matrixMulSharedMemory(float *A, float *B, float *C, int k) {
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
  A += cRow * BLOCK_SIZE * k; // A 的第 cRow*BLOCK_SIZE 行
  B += cCol * BLOCK_SIZE;     // B 的第 cCol*BLOCK_SIZE 列
  C += cRow * BLOCK_SIZE * k + cCol * BLOCK_SIZE; // C 的对应位置

  float tmp = 0.0f;

  // 沿着 k 维度分块迭代
  for (int bkIdx = 0; bkIdx < k; bkIdx += BLOCK_SIZE) {
    // 将 A 和 B 的一个 block 加载到共享内存
    // threadCol 作为连续索引，实现 coalesced 访问
    As[threadRow][threadCol] = A[threadRow * k + threadCol];
    Bs[threadRow][threadCol] = B[threadRow * k + threadCol];

    // 等待所有线程加载完成
    __syncthreads();

    // 移动 A 和 B 指针到下一个 block
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * k;

    // 计算当前 block 的点积
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow][dotIdx] * Bs[dotIdx][threadCol];
    }

    // 等待所有线程计算完成，再加载下一个 block
    __syncthreads();
  }

  // 写回结果
  C[threadRow * k + threadCol] = tmp;
}

// 1D Block Tiling 优化版本
// 每个线程计算 TM 个输出元素，提高计算密度
// 方阵版本：A(k×k) × B(k×k) = C(k×k)
__global__ void matrixMul1DBlockTiling(float *A, float *B, float *C, int k) {
  // 共享内存缓存
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // 当前 block 在 C 中的位置
  const int cRow = blockIdx.y;
  const int cCol = blockIdx.x;

  // 每个线程负责计算 C 中的 TM 个元素（一列中的 TM 个）
  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  // 移动指针到当前 block 的起始位置
  A += cRow * BM * k;
  B += cCol * BN;
  C += cRow * BM * k + cCol * BN;

  // 计算加载 A 和 B 时的索引
  const int innerRowA = threadIdx.x / BK;
  const int innerColA = threadIdx.x % BK;
  const int innerRowB = threadIdx.x / BN;
  const int innerColB = threadIdx.x % BN;

  // 每个线程在寄存器中存储 TM 个结果
  float threadResults[TM] = {0.0f};

  // 外循环：沿 K 维度分块迭代
  for (int bkIdx = 0; bkIdx < k; bkIdx += BK) {
    // 将 A 和 B 的一个 block 加载到共享内存
    As[innerRowA * BK + innerColA] = A[innerRowA * k + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * k + innerColB];
    __syncthreads();

    // 移动指针到下一个 block
    A += BK;
    B += BK * k;

    // 计算每个线程的 TM 个结果
    for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // 缓存 Bs 中的元素到寄存器
      float Btmp = Bs[dotIdx * BN + threadCol];
      for (int resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * Btmp;
      }
    }
    __syncthreads();
  }

  // 写回 TM 个结果
  for (int resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * k + threadCol] = threadResults[resIdx];
  }
}

// 2D Block Tiling 优化版本
// 每个线程计算 TM×TN 个输出元素（2D 区域）
// 使用多次加载填充共享内存
__global__ void matrixMul2DBlockTiling(float *A, float *B, float *C, int k) {
  __shared__ float As[BM2 * BK2];
  __shared__ float Bs[BK2 * BN2];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM2 * BN2;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM2 * TN2);

  const uint threadCol = threadIdx.x % (BN2 / TN2);
  const uint threadRow = threadIdx.x / (BN2 / TN2);

  // advance pointers to the starting positions
  A += cRow * BM2 * k;
  B += cCol * BN2;
  C += cRow * BM2 * k + cCol * BN2;

  const uint innerRowA = threadIdx.x / BK2;
  const uint innerColA = threadIdx.x % BK2;
  const uint strideA = numThreadsBlocktile / BK2;

  const uint innerRowB = threadIdx.x / BN2;
  const uint innerColB = threadIdx.x % BN2;
  const uint strideB = numThreadsBlocktile / BN2;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM2 * TN2] = {0.0};
  // register caches for As and Bs
  float regM[TM2] = {0.0};
  float regN[TN2] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < k; bkIdx += BK2) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM2; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK2 + innerColA] =
          A[(innerRowA + loadOffset) * k + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK2; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN2 + innerColB] =
          B[(innerRowB + loadOffset) * k + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK2;     // move BK columns to right
    B += BK2 * k; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK2; ++dotIdx) {
      // load relevant As & Bs entries into registers
      for (uint i = 0; i < TM2; ++i) {
        regM[i] = As[(threadRow * TM2 + i) * BK2 + dotIdx];
      }
      for (uint i = 0; i < TN2; ++i) {
        regN[i] = Bs[dotIdx * BN2 + threadCol * TN2 + i];
      }
      // perform outer product on register cache, accumulate into threadResults
      for (uint resIdxM = 0; resIdxM < TM2; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN2; ++resIdxN) {
          threadResults[resIdxM * TN2 + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM2; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN2; ++resIdxN) {
      C[(threadRow * TM2 + resIdxM) * k + threadCol * TN2 + resIdxN] =
          threadResults[resIdxM * TN2 + resIdxN];
    }
  }
}

// Vectorized 2D Block Tiling 优化版本
// 使用 float4 向量化加载，提高内存带宽利用率
// A 在加载时转置，优化后续访问模式
__global__ void matrixMulVectorized(float *A, float *B, float *C, int k) {
  // 注意：As 转置存储，所以维度是 BKV × BMV
  __shared__ float As[BKV * BMV];
  __shared__ float Bs[BKV * BNV];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BMV * BNV;
  const uint numThreadsBlocktile = totalResultsBlocktile / (TMV * TNV);

  const uint threadCol = threadIdx.x % (BNV / TNV);
  const uint threadRow = threadIdx.x / (BNV / TNV);

  // advance pointers to the starting positions
  A += cRow * BMV * k;
  B += cCol * BNV;
  C += cRow * BMV * k + cCol * BNV;

  // 每个线程加载 4 个元素（使用 float4）
  // A: 每行 BKV 个元素，共 BMV 行，每个线程加载 4 个
  const uint innerRowA = threadIdx.x / (BKV / 4);
  const uint innerColA = threadIdx.x % (BKV / 4);
  const uint strideA = numThreadsBlocktile / (BKV / 4);

  // B: 每行 BNV 个元素，共 BKV 行，每个线程加载 4 个
  const uint innerRowB = threadIdx.x / (BNV / 4);
  const uint innerColB = threadIdx.x % (BNV / 4);
  const uint strideB = numThreadsBlocktile / (BNV / 4);

  // allocate thread-local cache for results in registerfile
  float threadResults[TMV * TNV] = {0.0};
  // register caches for As and Bs
  float regM[TMV] = {0.0};
  float regN[TNV] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < k; bkIdx += BKV) {
    // populate the SMEM caches using float4 vectorized loads
    for (uint loadOffset = 0; loadOffset < BMV; loadOffset += strideA) {
      // 使用 float4 加载 A 的 4 个连续元素
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + loadOffset) * k + innerColA * 4])[0];
      // 转置存储到 As: 原来 A[row][col] -> As[col][row]
      As[(innerColA * 4 + 0) * BMV + innerRowA + loadOffset] = tmp.x;
      As[(innerColA * 4 + 1) * BMV + innerRowA + loadOffset] = tmp.y;
      As[(innerColA * 4 + 2) * BMV + innerRowA + loadOffset] = tmp.z;
      As[(innerColA * 4 + 3) * BMV + innerRowA + loadOffset] = tmp.w;
    }

    for (uint loadOffset = 0; loadOffset < BKV; loadOffset += strideB) {
      // 使用 float4 加载 B 的 4 个连续元素
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + loadOffset) * BNV + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + loadOffset) * k + innerColB * 4])[0];
    }
    __syncthreads();

    // advance blocktile
    A += BKV;
    B += BKV * k;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BKV; ++dotIdx) {
      // 从转置后的 As 加载: As[dotIdx][threadRow * TMV + i]
      for (uint i = 0; i < TMV; ++i) {
        regM[i] = As[dotIdx * BMV + threadRow * TMV + i];
      }
      for (uint i = 0; i < TNV; ++i) {
        regN[i] = Bs[dotIdx * BNV + threadCol * TNV + i];
      }
      // perform outer product on register cache
      for (uint resIdxM = 0; resIdxM < TMV; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TNV; ++resIdxN) {
          threadResults[resIdxM * TNV + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TMV; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TNV; ++resIdxN) {
      C[(threadRow * TMV + resIdxM) * k + threadCol * TNV + resIdxN] =
          threadResults[resIdxM * TNV + resIdxN];
    }
  }
}

// Warp Tiling 优化版本
// 在 warp 级别进行 tiling，为 Tensor Core 优化做准备
__global__ void matrixMulWarpTiling(float *A, float *B, float *C, int k) {
  // 转置存储 A，正常存储 B
  __shared__ float As[BKW * BMW];
  __shared__ float Bs[BKW * BNW];

  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  // Block 内的 warp 布局
  const uint warpIdx = threadIdx.x / WARPSIZE;
  const uint warpCol = warpIdx % (BNW / WN); // warp 在 block 中的列位置
  const uint warpRow = warpIdx / (BNW / WN); // warp 在 block 中的行位置

  // 线程在 warp 内的位置
  const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
  const uint threadColInWarp = threadIdxInWarp % (WSUBN / TNW); // 0-3
  const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TNW); // 0-7

  // 加载参数
  const uint numThreadsBlocktile = BMW * BNW / (TMW * TNW * WMITER * WNITER);

  const uint innerRowA = threadIdx.x / (BKW / 4);
  const uint innerColA = threadIdx.x % (BKW / 4);
  const uint strideA = numThreadsBlocktile / (BKW / 4);

  const uint innerRowB = threadIdx.x / (BNW / 4);
  const uint innerColB = threadIdx.x % (BNW / 4);
  const uint strideB = numThreadsBlocktile / (BNW / 4);

  // 移动指针到当前 block 的起始位置
  A += cRow * BMW * k;
  B += cCol * BNW;
  C += cRow * BMW * k + cCol * BNW;

  // 每个线程的结果寄存器: WMITER * TMW * WNITER * TNW
  float threadResults[WMITER * TMW * WNITER * TNW] = {0.0};
  // 寄存器缓存
  float regM[WMITER * TMW] = {0.0};
  float regN[WNITER * TNW] = {0.0};

  // 外循环：遍历 K 维度的 block tiles
  for (uint bkIdx = 0; bkIdx < k; bkIdx += BKW) {
    // 使用 float4 向量化加载，同时转置 A
    for (uint loadOffset = 0; loadOffset < BMW; loadOffset += strideA) {
      float4 tmp = reinterpret_cast<float4 *>(
          &A[(innerRowA + loadOffset) * k + innerColA * 4])[0];
      // 转置存储: A[row][col] -> As[col][row]
      As[(innerColA * 4 + 0) * BMW + innerRowA + loadOffset] = tmp.x;
      As[(innerColA * 4 + 1) * BMW + innerRowA + loadOffset] = tmp.y;
      As[(innerColA * 4 + 2) * BMW + innerRowA + loadOffset] = tmp.z;
      As[(innerColA * 4 + 3) * BMW + innerRowA + loadOffset] = tmp.w;
    }

    for (uint loadOffset = 0; loadOffset < BKW; loadOffset += strideB) {
      reinterpret_cast<float4 *>(
          &Bs[(innerRowB + loadOffset) * BNW + innerColB * 4])[0] =
          reinterpret_cast<float4 *>(
              &B[(innerRowB + loadOffset) * k + innerColB * 4])[0];
    }
    __syncthreads();

    A += BKW;
    B += BKW * k;

    // 计算 warptile
    for (uint dotIdx = 0; dotIdx < BKW; ++dotIdx) {
      // 加载 warp 的 A 子块到寄存器
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint i = 0; i < TMW; ++i) {
          regM[wSubRowIdx * TMW + i] =
              As[(dotIdx * BMW) + warpRow * WM + wSubRowIdx * WSUBM +
                 threadRowInWarp * TMW + i];
        }
      }
      // 加载 warp 的 B 子块到寄存器
      for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint i = 0; i < TNW; ++i) {
          regN[wSubColIdx * TNW + i] =
              Bs[(dotIdx * BNW) + warpCol * WN + wSubColIdx * WSUBN +
                 threadColInWarp * TNW + i];
        }
      }

      // 执行 warptile 矩阵乘法（外积累加）
      for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
          for (uint resIdxM = 0; resIdxM < TMW; ++resIdxM) {
            for (uint resIdxN = 0; resIdxN < TNW; ++resIdxN) {
              threadResults[(wSubRowIdx * TMW + resIdxM) * (WNITER * TNW) +
                            (wSubColIdx * TNW) + resIdxN] +=
                  regM[wSubRowIdx * TMW + resIdxM] *
                  regN[wSubColIdx * TNW + resIdxN];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // 写回结果
  for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
      // 计算 C 中的基地址
      float *C_interim = C + (warpRow * WM) * k + (warpCol * WN) +
                         (wSubRowIdx * WSUBM) * k + (wSubColIdx * WSUBN) +
                         (threadRowInWarp * TMW) * k + (threadColInWarp * TNW);
      for (uint resIdxM = 0; resIdxM < TMW; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TNW; ++resIdxN) {
          C_interim[resIdxM * k + resIdxN] =
              threadResults[(wSubRowIdx * TMW + resIdxM) * (WNITER * TNW) +
                            (wSubColIdx * TNW) + resIdxN];
        }
      }
    }
  }
}

// ==================== 辅助函数实现 ====================

// CPU 矩阵乘法（用于验证）
void matrixMulCPU(float *A, float *B, float *C, int k) {
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < k; j++) {
      float sum = 0.0f;
      for (int l = 0; l < k; l++) {
        sum += A[i * k + l] * B[l * k + j];
      }
      C[i * k + j] = sum;
    }
  }
}

// 验证结果（使用相对误差，对大矩阵更合理）
bool verifyResult(float *C_gpu, float *C_cpu, int k) {
  float maxRelError = 0.0f;
  int errorCount = 0;
  for (int i = 0; i < k * k; i++) {
    float absError = fabs(C_gpu[i] - C_cpu[i]);
    float relError = absError / (fabs(C_cpu[i]) + 1e-6f);
    if (relError > maxRelError) {
      maxRelError = relError;
    }
    // 相对误差超过 0.1% 算失败
    if (relError > 1e-3f) {
      if (errorCount < 3) { // 只打印前3个错误
        printf("验证失败: C_gpu[%d] = %f, C_cpu[%d] = %f, 相对误差: %.6f%%\n",
               i, C_gpu[i], i, C_cpu[i], relError * 100);
      }
      errorCount++;
    }
  }
  if (errorCount > 0) {
    printf("共 %d 个元素误差过大，最大相对误差: %.6f%%\n", errorCount,
           maxRelError * 100);
    return false;
  }
  return true;
}

// ==================== Kernel 启动器函数实现 ====================

// 朴素版本启动器
void launchMatrixMulNaive(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(CEIL_DIV(k, BLOCK_SIZE), CEIL_DIV(k, BLOCK_SIZE));
  matrixMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// Global Memory Coalescing 版本启动器
void launchMatrixMulCoalescing(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BLOCK_SIZE * BLOCK_SIZE);
  dim3 gridDim(CEIL_DIV(k, BLOCK_SIZE), CEIL_DIV(k, BLOCK_SIZE));
  matrixMulGlobalMemoryCoalescing<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// Shared Memory 版本启动器
void launchMatrixMulSharedMemory(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  dim3 gridDim(CEIL_DIV(k, BLOCK_SIZE), CEIL_DIV(k, BLOCK_SIZE));
  matrixMulSharedMemory<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// 1D Block Tiling 版本启动器
void launchMatrixMul1DBlockTiling(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BM * BN / TM); // 512 threads
  dim3 gridDim(CEIL_DIV(k, BN), CEIL_DIV(k, BM));
  matrixMul1DBlockTiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// 2D Block Tiling 版本启动器
void launchMatrixMul2DBlockTiling(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BM2 * BN2 / (TM2 * TN2)); // 256 threads
  dim3 gridDim(CEIL_DIV(k, BN2), CEIL_DIV(k, BM2));
  matrixMul2DBlockTiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// Vectorized 版本启动器
void launchMatrixMulVectorized(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BMV * BNV / (TMV * TNV)); // 256 threads
  dim3 gridDim(CEIL_DIV(k, BNV), CEIL_DIV(k, BMV));
  matrixMulVectorized<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// Warp Tiling 版本启动器
void launchMatrixMulWarpTiling(float *d_A, float *d_B, float *d_C, int k) {
  dim3 blockDim(BMW * BNW / (TMW * TNW * WMITER * WNITER)); // 256 threads
  dim3 gridDim(CEIL_DIV(k, BNW), CEIL_DIV(k, BMW));
  matrixMulWarpTiling<<<gridDim, blockDim>>>(d_A, d_B, d_C, k);
}

// GPU Warmup - 运行所有 kernel 一次
void warmupAllKernels(float *d_A, float *d_B, float *d_C, int k) {
  launchMatrixMulNaive(d_A, d_B, d_C, k);
  launchMatrixMulCoalescing(d_A, d_B, d_C, k);
  launchMatrixMulSharedMemory(d_A, d_B, d_C, k);
  launchMatrixMul1DBlockTiling(d_A, d_B, d_C, k);
  launchMatrixMul2DBlockTiling(d_A, d_B, d_C, k);
  launchMatrixMulVectorized(d_A, d_B, d_C, k);
  launchMatrixMulWarpTiling(d_A, d_B, d_C, k);
  cudaDeviceSynchronize();
}
