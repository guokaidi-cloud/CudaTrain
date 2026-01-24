学习文档：[CUB官网链接](https://docs.nvidia.com/cuda/cub/index.html)

# 什么是CUB？

CUB 为 CUDA 编程模型的每一层都提供了最先进的、可重复使用的软件组件：
## Parallel primitives (并行基本元素)
### Thread primitives
1. Thread-level **reduction**, etc.
2. 针对每种底层的 CUDA 架构进行了专门的优化

### Warp-wide “collective” primitives
1. Cooperative **warp-wide prefix scan, reduction**, etc.
2. 针对每种底层的 CUDA 架构进行了专门的优化


### Block-wide “collective” primitives
1. Cooperative **I/O, sort, scan, reduction, histogram**, etc.
2. 可兼容任意的线程块大小和类型

### Device-wide primitives
1. Parallel **sort, prefix scan, reduction, histogram**, etc.
2. 与 CUDA 动态并行性兼容

## Utilities
1. Fancy iterators（精妙的迭代器）
2. Thread and thread block I/O（线程和线程块的 I/O 操作）
3. PTX intrinsics（PTX 内存操作指令）
4. Device, kernel, and storage management（设备、内核和存储管理）

## CUB’s collective primitives
集合式软件原语对于构建高性能且易于维护的 CUDA 核心代码至关重要。这些集合功能使得复杂的并行代码能够被重复使用，而非重新编写，也能够被重新编译，而非手动移植。

作为一种 SIMT 编程模型，CUDA 既提供了标量软件接口，也提供了集合式软件接口。传统的软件接口是标量的：单个线程调用一个库函数来执行某些操作（这可能包括启动并行子任务）。另一方面，集合式接口由一组并行线程同时进入以执行某些协作操作。
CUB 的基本组件不受任何特定并行度或数据类型的限制。这种灵活性使得它们：
1. 能够适应并满足外部核心计算的需求
2. 可极其简便地调整为不同的粒度（每个块中的线程数、每个线程处理的项数等）

Thus CUB is CUDA Unbound!