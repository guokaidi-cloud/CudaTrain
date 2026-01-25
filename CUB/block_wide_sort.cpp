#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

//
// Block-sorting CUDA kernel
//
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void BlockSortKernel(int *d_in, int *d_out) {
  // Specialize BlockLoad, BlockStore, and BlockRadixSort collective types
  using BlockLoadT = cub::BlockLoad<int, BLOCK_THREADS, ITEMS_PER_THREAD,
                                    cub::BLOCK_LOAD_TRANSPOSE>;
  using BlockStoreT = cub::BlockStore<int, BLOCK_THREADS, ITEMS_PER_THREAD,
                                      cub::BLOCK_STORE_TRANSPOSE>;
  using BlockRadixSortT =
      cub::BlockRadixSort<int, BLOCK_THREADS, ITEMS_PER_THREAD>;

  // Allocate type-safe, repurposable shared memory for collectives
  __shared__ union {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockRadixSortT::TempStorage sort;
  } temp_storage;

  // Obtain this block's segment of consecutive keys (blocked across threads)
  int thread_keys[ITEMS_PER_THREAD];
  int block_offset = blockIdx.x * (BLOCK_THREADS * ITEMS_PER_THREAD);
  BlockLoadT(temp_storage.load).Load(d_in + block_offset, thread_keys);

  __syncthreads(); // Barrier for smem reuse

  // Collectively sort the keys
  BlockRadixSortT(temp_storage.sort).Sort(thread_keys);

  __syncthreads(); // Barrier for smem reuse

  // Store the sorted segment
  BlockStoreT(temp_storage.store).Store(d_out + block_offset, thread_keys);
}

int main() {
  // 配置参数
  const int BLOCK_THREADS = 128;
  const int ITEMS_PER_THREAD = 4;
  const int ELEMENTS_PER_BLOCK = BLOCK_THREADS * ITEMS_PER_THREAD; // 512
  const int NUM_BLOCKS = 1;
  const int TOTAL_ELEMENTS = NUM_BLOCKS * ELEMENTS_PER_BLOCK; // 512

  std::cout << "==========================================" << std::endl;
  std::cout << "CUB Block-wide Sort 测试" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "Block Threads: " << BLOCK_THREADS << std::endl;
  std::cout << "Items Per Thread: " << ITEMS_PER_THREAD << std::endl;
  std::cout << "Elements Per Block: " << ELEMENTS_PER_BLOCK << std::endl;
  std::cout << "Total Elements: " << TOTAL_ELEMENTS << std::endl;
  std::cout << "==========================================" << std::endl;

  // 创建测试数据
  std::vector<int> h_in(TOTAL_ELEMENTS);

  // 使用随机数生成器创建测试数据
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(1, 1000);

  std::cout << "\n生成测试数据..." << std::endl;
  for (int i = 0; i < TOTAL_ELEMENTS; i++) {
    h_in[i] = dis(gen);
  }

  // 打印前20个和后20个输入数据
  std::cout << "\n输入数据 (前20个): ";
  for (int i = 0; i < std::min(20, TOTAL_ELEMENTS); i++) {
    std::cout << std::setw(4) << h_in[i] << " ";
  }
  std::cout << "\n输入数据 (后20个): ";
  for (int i = std::max(0, TOTAL_ELEMENTS - 20); i < TOTAL_ELEMENTS; i++) {
    std::cout << std::setw(4) << h_in[i] << " ";
  }
  std::cout << std::endl;

  // 分配 GPU 内存
  int *d_in, *d_out;
  cudaError_t err;

  err = cudaMalloc(&d_in, TOTAL_ELEMENTS * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "CUDA 内存分配失败 (d_in): " << cudaGetErrorString(err)
              << std::endl;
    return 1;
  }

  err = cudaMalloc(&d_out, TOTAL_ELEMENTS * sizeof(int));
  if (err != cudaSuccess) {
    std::cerr << "CUDA 内存分配失败 (d_out): " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_in);
    return 1;
  }

  // 将数据复制到 GPU
  err = cudaMemcpy(d_in, h_in.data(), TOTAL_ELEMENTS * sizeof(int),
                   cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    std::cerr << "数据复制到 GPU 失败: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }

  // 执行 kernel
  std::cout << "\n执行排序 kernel..." << std::endl;
  BlockSortKernel<BLOCK_THREADS, ITEMS_PER_THREAD>
      <<<NUM_BLOCKS, BLOCK_THREADS>>>(d_in, d_out);

  // 检查 kernel 执行错误
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cerr << "Kernel 执行失败: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }

  // 等待 kernel 完成
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "CUDA 同步失败: " << cudaGetErrorString(err) << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }

  // 将结果复制回 CPU
  std::vector<int> h_out(TOTAL_ELEMENTS);
  err = cudaMemcpy(h_out.data(), d_out, TOTAL_ELEMENTS * sizeof(int),
                   cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    std::cerr << "数据复制回 CPU 失败: " << cudaGetErrorString(err)
              << std::endl;
    cudaFree(d_in);
    cudaFree(d_out);
    return 1;
  }

  // 打印结果
  std::cout << "\n==========================================" << std::endl;
  std::cout << "排序结果" << std::endl;
  std::cout << "==========================================" << std::endl;

  std::cout << "\n输出数据 (前20个): ";
  for (int i = 0; i < std::min(20, TOTAL_ELEMENTS); i++) {
    std::cout << std::setw(4) << h_out[i] << " ";
  }
  std::cout << "\n输出数据 (后20个): ";
  for (int i = std::max(0, TOTAL_ELEMENTS - 20); i < TOTAL_ELEMENTS; i++) {
    std::cout << std::setw(4) << h_out[i] << " ";
  }
  std::cout << std::endl;

  // 验证排序结果
  bool is_sorted = true;
  for (int i = 1; i < TOTAL_ELEMENTS; i++) {
    if (h_out[i] < h_out[i - 1]) {
      is_sorted = false;
      std::cerr << "\n排序错误: 位置 " << i - 1 << " = " << h_out[i - 1]
                << ", 位置 " << i << " = " << h_out[i] << std::endl;
      break;
    }
  }

  if (is_sorted) {
    std::cout << "\n✓ 排序验证通过！数据已正确排序。" << std::endl;
  } else {
    std::cout << "\n✗ 排序验证失败！" << std::endl;
  }

  // 打印统计信息
  std::cout << "\n==========================================" << std::endl;
  std::cout << "统计信息" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << "最小值: " << h_out[0] << std::endl;
  std::cout << "最大值: " << h_out[TOTAL_ELEMENTS - 1] << std::endl;
  std::cout << "中位数: " << h_out[TOTAL_ELEMENTS / 2] << std::endl;

  // 清理
  cudaFree(d_in);
  cudaFree(d_out);

  std::cout << "\n程序执行完成！" << std::endl;
  return is_sorted ? 0 : 1;
}