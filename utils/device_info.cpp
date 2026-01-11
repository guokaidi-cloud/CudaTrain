#include "device_info.h"
#include "cuda_utils.h"
#include <cstdio>

void printDeviceInfo(int deviceId) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));

  printf("=== GPU 设备信息 ===\n");
  printf("设备名称: %s\n", prop.name);
  printf("  - SM 数量: %d\n", prop.multiProcessorCount);
  printf("  - 最大线程数/Block: %d\n", prop.maxThreadsPerBlock);
  printf("  - 最大共享内存/Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
  printf("  - 总全局内存: %.2f GB\n",
         prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
  printf("  - Warp 大小: %d\n", prop.warpSize);
  printf("  - 计算能力: %d.%d\n", prop.major, prop.minor);
  printf("  - 内存时钟频率: %.2f GHz\n", prop.memoryClockRate / 1e6);
  printf("  - 内存总线宽度: %d bits\n", prop.memoryBusWidth);
  printf("  - L2 缓存大小: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
  printf("====================\n\n");
}

cudaDeviceProp getDeviceProperties(int deviceId) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceId));
  return prop;
}
