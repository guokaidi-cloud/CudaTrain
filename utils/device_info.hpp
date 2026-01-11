#ifndef DEVICE_INFO_HPP
#define DEVICE_INFO_HPP

#include <cuda_runtime.h>

// 打印 GPU 设备信息
void printDeviceInfo(int deviceId = 0);

// 获取 GPU 设备属性
cudaDeviceProp getDeviceProperties(int deviceId = 0);

#endif // DEVICE_INFO_HPP

