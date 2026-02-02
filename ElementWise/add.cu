#include "add.hpp"

// ==================== Kernel 实现 ====================

// 朴素版本：每个线程处理一个元素
__global__ void elementwise_add_naive(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 优化版本1：使用 float4 向量化，每个线程处理 4 个元素
// 使用 __restrict__ 和向量化加载/存储
__global__ void elementwise_add_float4(float *__restrict__ a, 
                                       float *__restrict__ b, 
                                       float *__restrict__ c, 
                                       int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < n) {
        // 使用向量化内存访问，每个线程处理 4 个连续的元素
        // 这样可以减少内存事务数量，提高带宽利用率
        float4 a4 = *reinterpret_cast<float4*>(&a[idx]);
        float4 b4 = *reinterpret_cast<float4*>(&b[idx]);
        *reinterpret_cast<float4*>(&c[idx]) = make_float4(
            a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    } else {
        // 处理剩余的元素（不足 4 个）
        for (int i = 0; i < 4 && idx + i < n; i++) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// 优化版本2：假设数据已对齐的 float4 版本（避免边界检查）
__global__ void elementwise_add_float4_aligned(float *__restrict__ a, 
                                                float *__restrict__ b, 
                                                float *__restrict__ c, 
                                                int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 4;
    
    // 假设 n 是 4 的倍数，这样可以完全避免边界检查
    if (idx < n) {
        float4 a4 = *reinterpret_cast<float4*>(&a[idx]);
        float4 b4 = *reinterpret_cast<float4*>(&b[idx]);
        *reinterpret_cast<float4*>(&c[idx]) = make_float4(
            a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w);
    }
}

// 优化版本3：使用更少的线程但每个线程处理更多工作
// 这个版本每个线程处理 8 个元素（2 个 float4）
__global__ void elementwise_add_float4_x2(float *__restrict__ a, 
                                          float *__restrict__ b, 
                                          float *__restrict__ c, 
                                          int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = tid * 8;  // 每个线程处理 8 个元素
    
    if (idx + 7 < n) {
        // 处理第一个 float4
        float4 a4_1 = *reinterpret_cast<float4*>(&a[idx]);
        float4 b4_1 = *reinterpret_cast<float4*>(&b[idx]);
        *reinterpret_cast<float4*>(&c[idx]) = make_float4(
            a4_1.x + b4_1.x, a4_1.y + b4_1.y, a4_1.z + b4_1.z, a4_1.w + b4_1.w);
        
        // 处理第二个 float4
        float4 a4_2 = *reinterpret_cast<float4*>(&a[idx + 4]);
        float4 b4_2 = *reinterpret_cast<float4*>(&b[idx + 4]);
        *reinterpret_cast<float4*>(&c[idx + 4]) = make_float4(
            a4_2.x + b4_2.x, a4_2.y + b4_2.y, a4_2.z + b4_2.z, a4_2.w + b4_2.w);
    } else {
        // 边界处理
        for (int i = 0; i < 8 && idx + i < n; i++) {
            c[idx + i] = a[idx + i] + b[idx + i];
        }
    }
}

// ==================== Kernel 启动器函数实现 ====================

void launchElementwiseAddNaive(float *d_a, float *d_b, float *d_c, int n) {
    int blockSize = BLOCK_SIZE;
    int gridSize = CEIL_DIV(n, blockSize);
    elementwise_add_naive<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

void launchElementwiseAddFloat4(float *d_a, float *d_b, float *d_c, int n) {
    int blockSize = BLOCK_SIZE;
    int gridSize = CEIL_DIV(n, 4 * blockSize);  // 每个线程处理 4 个元素
    elementwise_add_float4<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

void launchElementwiseAddFloat4Aligned(float *d_a, float *d_b, float *d_c, int n) {
    // 确保 n 是 4 的倍数
    int aligned_n = (n / 4) * 4;
    int blockSize = BLOCK_SIZE;
    int gridSize = CEIL_DIV(aligned_n, 4 * blockSize);
    elementwise_add_float4_aligned<<<gridSize, blockSize>>>(d_a, d_b, d_c, aligned_n);
    
    // 处理剩余的元素（如果 n 不是 4 的倍数）
    if (aligned_n < n) {
        elementwise_add_naive<<<1, 256>>>(d_a + aligned_n, d_b + aligned_n, 
                                         d_c + aligned_n, n - aligned_n);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launchElementwiseAddFloat4X2(float *d_a, float *d_b, float *d_c, int n) {
    int blockSize = BLOCK_SIZE;
    int gridSize = CEIL_DIV(n, 8 * blockSize);  // 每个线程处理 8 个元素
    elementwise_add_float4_x2<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
}

// ==================== CPU 参考实现 ====================

void elementwiseAddCPU(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ==================== 验证函数 ====================

bool verifyResult(float *result_gpu, float *result_cpu, int n, float tolerance) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(result_gpu[i] - result_cpu[i]);
        if (diff > tolerance) {
            printf("验证失败: 索引 %d, GPU结果=%.6f, CPU结果=%.6f, 差异=%.6f\n",
                   i, result_gpu[i], result_cpu[i], diff);
            return false;
        }
    }
    return true;
}
