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

// FlashAttention 的在线更新步骤。一个 block 把最多 kThreads 个新 key 合入一行。
// m 是历史最大分数，l = sum(exp(s - m))，o 是已经归一化的 value 加权和。
__global__ void softmax_row_incremental_kernel(
    float* output, float* maxima, float* sums, const float* scores, const float* values,
    size_t score_stride, int tile_cols, size_t value_dim) {
    const size_t row = blockIdx.x;
    const int lane = threadIdx.x;
    __shared__ BlockReduce::TempStorage scratch;
    __shared__ float weights[kThreads];
    __shared__ float merged_max, old_mass, merged_sum;

    const float score = lane < tile_cols ? scores[row * score_stride + lane] : -CUDART_INF_F;
    const float tile_max = BlockReduce(scratch).Reduce(score, cuda::maximum<>{});
    if (lane == 0) {
        const float old_max = maxima[row];
        merged_max = fmaxf(old_max, tile_max);
        old_mass = sums[row] * softmax_weight(old_max, merged_max);
        maxima[row] = merged_max;
    }
    __syncthreads();

    // 新旧权重都以 merged_max 为基准，分母可以直接相加。
    const float weight = softmax_weight(score, merged_max);
    const float tile_sum = BlockReduce(scratch).Sum(weight);
    if (lane == 0) {
        merged_sum = old_mass + tile_sum;
        sums[row] = merged_sum;
    }
    __syncthreads();

    // 先归一化再乘 V，避免很大的有限 V 在中间分子求和时溢出。
    const float inv_sum = merged_sum > 0.0f ? 1.0f / merged_sum : 0.0f;
    const float old_fraction = old_mass * inv_sum;
    weights[lane] = weight * inv_sum;
    __syncthreads();

    for (size_t d = lane; d < value_dim; d += blockDim.x) {
        float updated = old_fraction * output[row * value_dim + d];
        for (int col = 0; col < tile_cols; ++col) {
            updated = fmaf(weights[col], values[col * value_dim + d], updated);
        }
        output[row * value_dim + d] = updated;
    }
}

void softmax_row_simple(const float* input, float* output, int rows, int cols) {
    if (rows < 0 || cols < 0) throw std::invalid_argument("negative softmax dimension");
    if (rows == 0 || cols == 0) return;
    softmax_row_simple_kernel<<<rows, kThreads>>>(input, output, cols);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

static void launch_attention_tiles(float* output, float* maxima, float* sums,
                                   const float* scores, const float* values,
                                   int rows, int cols, int value_dim, int score_stride) {
    for (size_t start = 0; start < size_t(cols); start += kThreads) {
        const int tile_cols = std::min(size_t(kThreads), size_t(cols) - start);
        softmax_row_incremental_kernel<<<rows, kThreads>>>(
            output, maxima, sums, scores + start, values + start * value_dim,
            score_stride, tile_cols, value_dim);
        CUDA_CHECK(cudaGetLastError());
    }
}

// 初值：output[rows, value_dim] = 0，maxima[rows] = -inf，sums[rows] = 0。
// 每次传入新分块的 scores 和 values 首地址。values 为 [cols, value_dim]，
// 各行 query 共用。score_stride 是相邻两行 scores 的间距（以 float 计）。
void softmax_row_incremental(float* output, float* maxima, float* sums,
                             const float* scores, const float* values,
                             int rows, int cols, int value_dim, int score_stride) {
    if (rows < 0 || cols < 0 || value_dim <= 0 || score_stride < cols) {
        throw std::invalid_argument("invalid attention dimensions or score stride");
    }
    if (rows == 0 || cols == 0) return;
    launch_attention_tiles(output, maxima, sums, scores, values, rows, cols, value_dim, score_stride);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CPU 参考直接用 double 计算整个前缀的 softmax 和 P @ V，独立于 GPU 的递推公式。
void check_result(const char* quantity, float actual, double expected) {
    if (actual == expected) return;  // 包含整行 mask 时 m = -inf
    const double tolerance = 2e-5 * std::max(1.0, std::abs(expected));
    if (!std::isfinite(actual) || !std::isfinite(expected) ||
        std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: GPU=%.9g CPU=%.12g\n", quantity, actual, expected);
        throw std::runtime_error("GPU/CPU mismatch");
    }
}

void test_case(const char* name, int rows, int cols, int dim, int chunk,
               const std::vector<float>& scores, const std::vector<float>& values,
               bool print_probabilities = false) {
    const float masked = -std::numeric_limits<float>::infinity();
    thrust::device_vector<float> gpu_scores(scores), gpu_values(values);
    thrust::device_vector<float> probabilities(scores.size()), output(size_t(rows) * dim, 0.0f);
    thrust::device_vector<float> maxima(rows, masked), sums(rows, 0.0f);

    softmax_row_simple(gpu_scores.data().get(), probabilities.data().get(), rows, cols);
    const thrust::host_vector<float> prob = probabilities;
    softmax_row_online_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), probabilities.data().get(), cols);
    CUDA_CHECK(cudaGetLastError());
    const thrust::host_vector<float> online_prob = probabilities;

    for (int start = 0; start < cols; start += chunk) {
        const int count = std::min(chunk, cols - start);
        const int end = start + count;
        softmax_row_incremental(output.data().get(), maxima.data().get(), sums.data().get(),
                                gpu_scores.data().get() + start,
                                gpu_values.data().get() + size_t(start) * dim, rows, count, dim, cols);
        const thrust::host_vector<float> result = output, row_max = maxima, row_sum = sums;
        for (int row = 0; row < rows; ++row) {
            const float* s = scores.data() + size_t(row) * cols;
            const double maximum = *std::max_element(s, s + end);
            std::vector<double> weights(end);
            double sum = 0.0;
            for (int c = 0; c < end; ++c) {
                weights[c] = s[c] == masked ? 0.0 : std::exp(s[c] - maximum);
                sum += weights[c];
            }
            check_result("m", row_max[row], maximum);
            check_result("l", row_sum[row], sum);
            for (int d = 0; d < dim; ++d) {
                double weighted_sum = 0.0;
                for (int c = 0; c < end; ++c) {
                    weighted_sum += weights[c] * values[size_t(c) * dim + d];
                }
                check_result("attention", result[size_t(row) * dim + d],
                             sum == 0.0 ? 0.0 : weighted_sum / sum);
            }
            if (end == cols) {
                for (int c = 0; c < cols; ++c) {
                    const double expected = sum == 0.0 ? 0.0 : weights[c] / sum;
                    check_result("softmax", prob[size_t(row) * cols + c], expected);
                    check_result("online softmax", online_prob[size_t(row) * cols + c], expected);
                }
            }
        }
    }
    std::printf("PASS %s\n", name);
    if (print_probabilities) {
        for (int row = 0; row < rows; ++row) {
            for (int c = 0; c < cols; ++c) std::printf(" %.7f", prob[row * cols + c]);
            std::printf("\n");
        }
    }
}

// 基准用相同的 FP32 输入、一个 block 一行，比较两种 P @ V 计算方法。
__global__ void apply_values_kernel(const float* probabilities, const float* values,
                                    float* output, size_t cols, size_t dim) {
    const size_t row = blockIdx.x;
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) {
        float result = 0.0f;
        for (size_t c = 0; c < cols; ++c) {
            result = fmaf(probabilities[row * cols + c], values[c * dim + d], result);
        }
        output[row * dim + d] = result;
    }
}

__global__ void reset_attention_kernel(float* output, float* maxima, float* sums, size_t dim) {
    const size_t row = blockIdx.x;
    if (threadIdx.x == 0) {
        maxima[row] = -CUDART_INF_F;
        sums[row] = 0.0f;
    }
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x) output[row * dim + d] = 0.0f;
}

// 每个样本用一对 CUDA event 包围一批完整运算，避免逐个 kernel 同步。
// 交替测量两种实现，取各自 7 批单次平均耗时的中位数。
template <typename Baseline, typename Online>
void benchmark_pair(const char* operation, Baseline baseline, Online online) {
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
    std::printf("  %-16s  baseline %9.2f us   online %9.2f us   time change %+6.1f%%\n",
                operation, base, candidate, 100.0f * (candidate / base - 1.0f));
}

void benchmark_shape(int rows, int cols, int dim) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> sample(-3.0f, 3.0f);
    std::vector<float> scores(size_t(rows) * cols), values(size_t(cols) * dim);
    for (float& s : scores) s = sample(rng);
    for (float& v : values) v = sample(rng);
    thrust::device_vector<float> gpu_scores(scores), gpu_values(values);
    thrust::device_vector<float> prob(scores.size()), online_prob(scores.size());
    thrust::device_vector<float> direct_out(size_t(rows) * dim), online_out(size_t(rows) * dim);
    thrust::device_vector<float> maxima(rows), sums(rows);

    auto simple = [&] {
        softmax_row_simple_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), prob.data().get(), cols);
    };
    auto online = [&] {
        softmax_row_online_kernel<<<rows, kThreads>>>(gpu_scores.data().get(), online_prob.data().get(), cols);
    };
    auto materialized = [&] {
        simple();
        apply_values_kernel<<<rows, kThreads>>>(prob.data().get(), gpu_values.data().get(),
                                               direct_out.data().get(), cols, dim);
    };
    auto streaming = [&] {
        reset_attention_kernel<<<rows, kThreads>>>(online_out.data().get(), maxima.data().get(),
                                                  sums.data().get(), dim);
        launch_attention_tiles(online_out.data().get(), maxima.data().get(), sums.data().get(),
                               gpu_scores.data().get(), gpu_values.data().get(), rows, cols, dim, cols);
    };

    // 确认两个对照组逐元素一致；分配、输入传输、校验均在计时区间之外。
    materialized();
    online();
    streaming();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    auto compare = [](const char* label, const thrust::host_vector<float>& expected,
                                         const thrust::host_vector<float>& actual) {
        for (size_t i = 0; i < actual.size(); ++i) check_result(label, actual[i], expected[i]);
    };
    compare("benchmark softmax", prob, online_prob);
    compare("benchmark attention", direct_out, online_out);

    std::printf("rows=%d cols=%d value_dim=%d\n", rows, cols, dim);
    benchmark_pair("softmax", simple, online);
    benchmark_pair("softmax(S) @ V", materialized, streaming);
}

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::string_view(argv[1]) != "--bench")) {
        std::fprintf(stderr, "Usage: %s [--bench]\n", argv[0]);
        return 1;
    }
    // V = I 时，attention 输出等于整行 softmax，可直接比较两种写法。
    test_case("original scores, V=I", 2, 4, 4, 2,
              {1, 2, 3, 0, 0.2f, 0.5f, 0.1f, 1.5f},
              {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, true);
    test_case("equal scores, two updates", 1, 4, 2, 2,
              {0, 0, 0, 0}, {1, 0, 0, 1, 1, 0, 0, 1});
    test_case("large finite values", 1, 4, 1, 2,
              {0, 0, 0, 0}, {2e38f, 2e38f, 2e38f, 2e38f});

    // 行宽超过原共享数组上限，块尾不齐，value 维度超过一个线程块。
    constexpr int cols = 1025, dim = 131;
    const float masked = -std::numeric_limits<float>::infinity();
    std::vector<float> scores(3 * cols, masked), values(cols * dim);
    for (int c = 0; c < cols; ++c) {
        scores[c] = (c % 29) - 1000.0f;
        if (c >= 300 && c < 600) scores[cols + c] = 1000.0f + c % 7;
        for (int d = 0; d < dim; ++d) {
            values[c * dim + d] = ((c * 13 + d * 7) % 61 - 30) / 11.0f;
        }
    }
    scores[cols - 1] = 1000.0f;
    test_case("1025 columns, partial tiles and masked rows", 3, cols, dim, 300, scores, values);
    if (argc == 2) {
        std::puts("FP32 CUDA events: 10 warmups, median of 7 batches x 20 evaluations. "
                  "Attention includes online state reset. Scores and V already on GPU.");
        for (int width : {128, 1024, 4096}) benchmark_shape(256, width, 64);
    }
}
