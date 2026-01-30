# CUDA 学习资料大纲（从小白到入门）

本仓库整理 CUDA 从入门到进阶的官方文档、硬件架构、进阶库与开源项目，便于系统学习与查阅。

---

## 目录

- [1. 官方入门资料](#1-官方入门资料)
- [2. NVIDIA GPU 硬件体系架构](#2-nvidia-gpu-硬件体系架构)
- [3. 进阶材料](#3-进阶材料)
- [4. 编译与指令集](#4-编译与指令集)
- [5. NVIDIA GPU 分析工具](#5-nvidia-gpu-分析工具)
- [6. 开源学习项目](#6-开源学习项目)
- [7. 其它资料](#7-其它资料)

---

## 1. 官方入门资料

### 1.1 CUDA 简单介绍

- [An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)  
  入门博客，讲解 CUDA 核心概念与基本用法。

### 1.2 CUDA 编程指南

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)  
  NVIDIA 官方权威文档，涵盖 CUDA 特性、API 与最佳实践，必备参考。

### 1.3 CUDA 编程手册（PDF）

- [CUDA Programming Guide (PDF)](https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf)  
  官方编程手册，适合系统学习与离线查阅。

---

## 2. NVIDIA GPU 硬件体系架构

- [NVIDIA GPU 硬件架构概览](https://zhuanlan.zhihu.com/p/1910093597233099119)（知乎）

### 2.1 NVIDIA Ampere 架构

| 资料 | 说明 |
|------|------|
| [Ampere 架构白皮书 (PDF)](https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf) | 官方技术白皮书：第三代 Tensor Core、多实例 GPU 等。 |
| [NVIDIA Ampere Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) | 开发者博客，通俗讲解 Ampere 创新点与性能提升。 |

### 2.2 NVIDIA Hopper 架构

| 资料 | 说明 |
|------|------|
| [GTC22 Hopper 白皮书 (PDF)](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf) | 聚焦 HPC 与 AI 加速，含技术细节与性能数据。 |
| [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) | 官方博客：Transformer Engine、DPX 指令集等。 |

### 2.3 NVIDIA Blackwell 架构

| 资料 | 说明 |
|------|------|
| [Blackwell 架构总览](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) | 官方架构介绍页。 |
| [Blackwell 技术简报 (PDF)](https://cdn.prod.website-files.com/61dda201f29b7efc52c5fbaf/6602ea9d0ce8cb73fb6de87f_nvidia-blackwell-architecture-technical-brief.pdf) | 硬件设计、性能与新功能概览。 |
| [NVFP4 技术介绍](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/) | 16 位精度与 4 位速度/能效结合的 NVFP4。 |

### 2.4 端侧产品（车端 Drive 系列）

#### 2.4.1 Drive AGX Orin

- 常用 OrinX / OrinY，基于 **Ampere**，在 CPU/Cache、GPU 规模上有差异，含 **DLA Core**。
- [Orin 特性说明](https://developer.nvidia.com/drive/agx#section-orin-features)

#### 2.4.2 Drive AGX Thor

- 常用 **Thor** 产品，基于 **Blackwell**，无 DLA Core。
- [Thor 特性说明](https://developer.nvidia.com/drive/agx#section-thor-features)

---

## 3. 进阶材料

### 3.1 CUDA Core / Tensor Core — SM 运行单元

- 待补充：SM、CUDA Core、Tensor Core 与 warp 调度等。

### 3.2 TensorRT — 推理引擎

- 待补充：TensorRT 入门与部署流程。

#### 3.2.1 CUDA Graph

- 待补充：CUDA Graph 用法与性能优化。

### 3.3 cuBLAS — 线性代数库

- 待补充：cuBLAS API 与典型用法（如矩阵乘、批处理）。

### 3.4 CUTLASS — 线性代数运算模板库

- 待补充：CUTLASS 与自定义 kernel 开发。

### 3.5 CUB — 原语与模板库

- 待补充：BlockReduce、BlockScan、Warp 原语等。

### 3.6 其它

- 待补充：Thrust、cuDNN、cuFFT 等。

---

## 4. 编译与指令集

### 4.1 PTX — 中间虚拟指令集

- 待补充：PTX 语法与阅读方法。

### 4.2 SASS — 最终机器指令集

- 待补充：SASS 与性能/占用分析。

---

## 5. NVIDIA GPU 分析工具

### 5.1 NVIDIA Nsight Systems

- 待补充：系统级性能与时间线分析。

### 5.2 NVIDIA Nsight Compute (NCU)

- 待补充：Kernel 级指标与 Roofline 分析。

---

## 6. 开源学习项目

| 项目 | 类型 | 链接 |
|------|------|------|
| **CUDA Samples** | 官方示例 | [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples) |
| **LeetCUDA** | 习题与练习 | [xlite-dev/LeetCUDA](https://github.com/xlite-dev/LeetCUDA) |
| **Lidar AI Solution (BEVFusion)** | CV / 自动驾驶 | [NVIDIA-AI-IOT/Lidar_AI_Solution](https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution) |
| **TensorRT-LLM** | LLM 推理 | [NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) |

### 6.4 TensorRT-LLM（LLM 类）

- 支持：KV-Cache、PagedAttention、FlashAttention、MHA/MQA/GQA 等。
- [TensorRT-LLM 介绍](https://zhuanlan.zhihu.com/p/669576221)（知乎）

---

## 7. 其它资料

### 7.1 CUDA Kernel 优化

- 待补充：优秀优化文章与实战案例链接。

---

