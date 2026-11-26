#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/functional>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "../utils/cuda_utils.hpp"

constexpr int kThreads = 128;
using BlockReduce = cub::BlockReduce<float, kThreads>;

// scores 允许有限数和表示 mask 的 -inf。约定整行都被 mask 时输出全 0。
__device__ float softmax_weight(float score, float row_max) {
    return score == -CUDART_INF_F ? 0.0f : expf(score - row_max);
}

// 一个 block 负责一行。共享内存仅用于规约和广播，与列数无关。
__global__ void softmax_row_simple_kernel(const float* input, float* output, size_t cols) {
    const size_t offset = size_t(blockIdx.x) * cols;
    __shared__ BlockReduce::TempStorage scratch;
    __shared__ float row_max;
    __shared__ float row_sum;

    // 一个线程可能负责多列，所以需要累加
    float local_max = -CUDART_INF_F;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, input[offset + col]);
    }
    const float maximum = BlockReduce(scratch).Reduce(local_max, cuda::maximum<>{});
    if (threadIdx.x == 0) row_max = maximum;
    __syncthreads();

    // 一个线程可能负责多列，所以需要累加
    float local_sum = 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float weight = softmax_weight(input[offset + col], row_max);
        output[offset + col] = weight;
        local_sum += weight;
    }
    const float sum = BlockReduce(scratch).Sum(local_sum);
    if (threadIdx.x == 0) row_sum = sum;
    __syncthreads();

    if (row_sum > 0.0f) {
        for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
            output[offset + col] /= row_sum;
        }
    }
}

// 在线计算每行的 (m, l)。每个线程先处理自己的列，再用同一公式合并线程的结果。
struct RowStats {
    float maximum;
    float sum;
};

struct MergeStats {
    __device__ RowStats operator()(RowStats a, RowStats b) const {
        // 同时覆盖两个分块都为空的情况：(-inf, 0)。
        if (a.maximum == b.maximum) return {a.maximum, a.sum + b.sum};
        if (a.maximum > b.maximum) {
            return {a.maximum, a.sum + b.sum * expf(b.maximum - a.maximum)};
        }
        return {b.maximum, b.sum + a.sum * expf(a.maximum - b.maximum)};
    }
};

__global__ void softmax_row_online_kernel(const float* input, float* output, size_t cols) {
    using StatsReduce = cub::BlockReduce<RowStats, kThreads>;
    __shared__ StatsReduce::TempStorage scratch;
    __shared__ RowStats row_stats;
    const size_t offset = size_t(blockIdx.x) * cols;
    RowStats local{-CUDART_INF_F, 0.0f};
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = input[offset + col];
        if (x != -CUDART_INF_F) local = MergeStats{}(local, {x, 1.0f});
    }
    const RowStats total = StatsReduce(scratch).Reduce(local, MergeStats{});
    if (threadIdx.x == 0) row_stats = total;
    __syncthreads();

    const float inv_sum = row_stats.sum > 0.0f ? 1.0f / row_stats.sum : 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        output[offset + col] = softmax_weight(input[offset + col], row_stats.maximum) * inv_sum;
    }
}

void softmax_row_simple(const float* input, float* output, int rows, int cols) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("negative softmax dimension");
    if (rows == 0 || cols == 0) return;
    softmax_row_simple_kernel<<<rows, kThreads>>>(input, output, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void check_result(const char* quantity, double actual, double expected) {
    const double tolerance = 2e-5 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: GPU=%.9g CPU=%.12g\n", quantity, actual, expected);
        throw std::runtime_error("GPU/CPU mismatch");
    }
}

// 用 double 在 CPU 上逐行计算参考概率。全 mask 行的概率和约定为 0。
void check_probabilities(const std::vector<float>& scores, int cols,
                         const thrust::host_vector<float>& prob,
                         const thrust::host_vector<float>& online_prob) {
    const float masked = -std::numeric_limits<float>::infinity();
    std::vector<double> weights(cols);
    for (size_t offset = 0; offset < scores.size(); offset += cols) {
        const float* row = scores.data() + offset;
        const double maximum = *std::max_element(row, row + cols);
        double sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            weights[col] = row[col] == masked ? 0.0 : std::exp(row[col] - maximum);
            sum += weights[col];
        }
        double prob_sum = 0.0, online_sum = 0.0;
        for (int col = 0; col < cols; ++col) {
            const double expected = sum == 0.0 ? 0.0 : weights[col] / sum;
            check_result("softmax", prob[offset + col], expected);
            check_result("online softmax", online_prob[offset + col], expected);
            prob_sum += prob[offset + col];
            online_sum += online_prob[offset + col];
        }
        const double expected_sum = sum == 0.0 ? 0.0 : 1.0;
        check_result("softmax row sum", prob_sum, expected_sum);
        check_result("online softmax row sum", online_sum, expected_sum);
    }
}

void test_case(const char* name, int rows, int cols, const std::vector<float>& scores,
               bool print_probabilities = false) {
    thrust::device_vector<float> gpu_scores(scores);
    thrust::device_vector<float> probabilities(scores.size()), online_prob(scores.size());
    softmax_row_simple(gpu_scores.data().get(), probabilities.data().get(), rows, cols);
    softmax_row_online_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), online_prob.data().get(), cols);
    CUDA_CHECK(cudaGetLastError());
    const thrust::host_vector<float> prob = probabilities;
    check_probabilities(scores, cols, prob, online_prob);
    std::printf("PASS %s\n", name);
    if (print_probabilities) {
        for (int row = 0; row < rows; ++row) {
            for (int c = 0; c < cols; ++c) std::printf(" %.7f", prob[row * cols + c]);
            std::printf("\n");
        }
    }
}

// 每个样本用一对 CUDA event 包围一批完整运算，避免逐个 kernel 同步。
// 交替测量两种实现，取各自 7 批单次平均耗时的中位数。
template <typename Baseline, typename Online>
void benchmark_pair(Baseline baseline, Online online) {
    constexpr int warmup = 10, batches = 7, repeats = 20;
    for (int i = 0; i < warmup; ++i) {
        baseline();
        online();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    auto measure_us = [&](auto launch) {
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < repeats; ++i) launch();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return 1000.0f * ms / repeats;
    };
    std::array<float, batches> reference_us, online_us;
    for (int i = 0; i < batches; ++i) {
        if (i % 2 == 0) reference_us[i] = measure_us(baseline);
        online_us[i] = measure_us(online);
        if (i % 2 != 0) reference_us[i] = measure_us(baseline);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(reference_us.begin(), reference_us.end());
    std::sort(online_us.begin(), online_us.end());
    const float base = reference_us[batches / 2], candidate = online_us[batches / 2];
    std::printf("  softmax  baseline %9.2f us   online %9.2f us   time change %+6.1f%%\n",
                base, candidate, 100.0f * (candidate / base - 1.0f));
}

void benchmark_shape(int rows, int cols) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sample(-3.0f, 3.0f);
    std::vector<float> scores(size_t(rows) * cols);
    for (float& s : scores) s = sample(rng);
    thrust::device_vector<float> gpu_scores(scores);
    thrust::device_vector<float> prob(scores.size()), online_prob(scores.size());

    auto simple = [&] {
        softmax_row_simple_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), prob.data().get(), cols);
    };
    auto online = [&] {
        softmax_row_online_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), online_prob.data().get(), cols);
    };
    simple();
    online();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    check_probabilities(scores, cols, prob, online_prob);

    std::printf("rows=%d cols=%d\n", rows, cols);
    benchmark_pair(simple, online);
}

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--bench")) {
        std::fprintf(stderr, "Usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    test_case("original scores", 2, 4, {1, 2, 3, 0, 0.2f, 0.5f, 0.1f, 1.5f}, true);
    test_case("equal scores", 1, 4, {0, 0, 0, 0});
    test_case("large positive and negative scores", 1, 4, {2e38f, -2e38f, 2e38f, 0});

    // 行宽超过 1024，覆盖尾列最大值、部分 mask 和整行 mask。
    constexpr int cols = 1025;
    const float masked = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(3 * cols, masked);
    for (int c = 0; c < cols; ++c) {
        scores[c] = (c % 29) - 1000.0f;
        if (c >= 300 && c < 600) scores[cols + c] = 1000.0f + c % 7;
    }
    scores[cols - 1] = 1000.0f;
    test_case("1025 columns and masked rows", 3, cols, scores);
    if (argc == 2) {
        std::puts("FP32 CUDA events: 10 warmups, median of 7 batches x 20 evaluations. "
                  "Scores already on GPU.");
        for (int width : {128, 1024, 4096}) benchmark_shape(256, width);
    }
}
