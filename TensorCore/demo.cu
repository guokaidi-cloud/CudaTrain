// code reference: https://leimao.github.io/blog/NVIDIA-Tensor-Core-Programming/
// reference: https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/

#include <cassert>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           int const line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char* const file, int const line)
{
    cudaError_t const err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

template <class T>
float measure_performance(std::function<T(cudaStream_t)> bound_function,
                          cudaStream_t stream, int num_repeats = 100,
                          int num_warmups = 100)
{
    cudaEvent_t start, stop;
    float time;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    for (int i = 0; i < num_warmups; ++i)
        bound_function(stream);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
    CHECK_CUDA_ERROR(cudaEventRecord(start, stream));

    for (int i = 0; i < num_repeats; ++i)
        bound_function(stream);

    CHECK_CUDA_ERROR(cudaEventRecord(stop, stream));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));
    CHECK_LAST_CUDA_ERROR();
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    return time / num_repeats;
}

// 列主序存储；与 cpu_gemm 一致。转置时 WMMA fragment 的 layout 与指针见 launch 侧。
template <typename T1, typename T2, int WMMA_M, int WMMA_N, int WMMA_K,
          typename LayoutA, typename LayoutB>
__global__ void wmma_gemm_kernel(
    const T1* A, const T1* B, T2* C,
    int m, int n, int k,
    int lda, int ldb, int ldc,
    bool is_A_transpose, bool is_B_transpose,
    float alpha, float beta)
{
    uint32_t const warpM{(blockIdx.x * blockDim.x + threadIdx.x) / 32};
    uint32_t const warpN{blockIdx.y * blockDim.y + threadIdx.y};

    uint32_t const row{warpM * static_cast<uint32_t>(WMMA_M)};
    uint32_t const col{warpN * static_cast<uint32_t>(WMMA_N)};

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T1, LayoutA> a_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T1, LayoutB> b_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> acc_frag;
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, T2> c_frag;

    nvcuda::wmma::fill_fragment(acc_frag, static_cast<T2>(0));

    for (uint32_t ki = 0; ki < static_cast<uint32_t>(k); ki += WMMA_K)
    {
        uint32_t const matrix_mma_a_row_idx{is_A_transpose ? ki : row};
        uint32_t const matrix_mma_a_col_idx{is_A_transpose ? row : ki};
        uint32_t const matrix_mma_b_row_idx{is_B_transpose ? col : ki};
        uint32_t const matrix_mma_b_col_idx{is_B_transpose ? ki : col};

        if (matrix_mma_a_row_idx < static_cast<uint32_t>(is_A_transpose ? k : m) &&
            matrix_mma_a_col_idx < static_cast<uint32_t>(is_A_transpose ? m : k) &&
            matrix_mma_b_row_idx < static_cast<uint32_t>(is_B_transpose ? n : k) &&
            matrix_mma_b_col_idx < static_cast<uint32_t>(is_B_transpose ? k : n))
        {
            T1 const* const a_ptr{A + matrix_mma_a_row_idx +
                                  matrix_mma_a_col_idx * static_cast<uint32_t>(lda)};
            T1 const* const b_ptr{B + matrix_mma_b_row_idx +
                                  matrix_mma_b_col_idx * static_cast<uint32_t>(ldb)};
            nvcuda::wmma::load_matrix_sync(a_frag, a_ptr, lda);
            nvcuda::wmma::load_matrix_sync(b_frag, b_ptr, ldb);
            nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    if (row < static_cast<uint32_t>(m) && col < static_cast<uint32_t>(n))
    {
        T2* const c_ptr{C + row + col * static_cast<uint32_t>(ldc)};
        nvcuda::wmma::load_matrix_sync(c_frag, c_ptr, ldc, nvcuda::wmma::mem_col_major);
        for (int i = 0; i < c_frag.num_elements; i++)
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        nvcuda::wmma::store_matrix_sync(c_ptr, c_frag, ldc, nvcuda::wmma::mem_col_major);
    }
}

void launch_wmma_half(
    const __half* A, const __half* B, float* C,
    int m, int n, int k,
    bool ta, bool tb,
    cudaStream_t stream)
{
    int const lda = ta ? k : m;
    int const ldb = tb ? n : k;
    int const ldc = m;

    constexpr int kWarpX = 4;
    constexpr int kWarpY = 4;
    dim3 const block(kWarpX * 32, kWarpY);
    dim3 const grid((m + 16 * kWarpX - 1) / (16 * kWarpX),
                    (n + 16 * kWarpY - 1) / (16 * kWarpY));

    if (!ta && !tb)
        wmma_gemm_kernel<__half, float, 16, 16, 16, nvcuda::wmma::col_major, nvcuda::wmma::col_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else if (ta && !tb)
        wmma_gemm_kernel<__half, float, 16, 16, 16, nvcuda::wmma::row_major, nvcuda::wmma::col_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else if (!ta && tb)
        wmma_gemm_kernel<__half, float, 16, 16, 16, nvcuda::wmma::col_major, nvcuda::wmma::row_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else
        wmma_gemm_kernel<__half, float, 16, 16, 16, nvcuda::wmma::row_major, nvcuda::wmma::row_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);

    CHECK_LAST_CUDA_ERROR();
}

void launch_wmma_int8(
    const signed char* A, const signed char* B, int32_t* C,
    int m, int n, int k,
    bool ta, bool tb,
    cudaStream_t stream)
{
    int const lda = ta ? k : m;
    int const ldb = tb ? n : k;
    int const ldc = m;

    constexpr int kWarpX = 4;
    constexpr int kWarpY = 4;
    dim3 const block(kWarpX * 32, kWarpY);
    dim3 const grid((m + 16 * kWarpX - 1) / (16 * kWarpX),
                    (n + 16 * kWarpY - 1) / (16 * kWarpY));

    if (!ta && !tb)
        wmma_gemm_kernel<signed char, int32_t, 16, 16, 16, nvcuda::wmma::col_major, nvcuda::wmma::col_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else if (ta && !tb)
        wmma_gemm_kernel<signed char, int32_t, 16, 16, 16, nvcuda::wmma::row_major, nvcuda::wmma::col_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else if (!ta && tb)
        wmma_gemm_kernel<signed char, int32_t, 16, 16, 16, nvcuda::wmma::col_major, nvcuda::wmma::row_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);
    else
        wmma_gemm_kernel<signed char, int32_t, 16, 16, 16, nvcuda::wmma::row_major, nvcuda::wmma::row_major><<<grid, block, 0, stream>>>(A, B, C, m, n, k, lda, ldb, ldc, ta, tb, 1.f, 0.f);

    CHECK_LAST_CUDA_ERROR();
}

template <typename T1, typename T2>
void launch_wmma_mm(const T1* A, const T1* B, T2* C,
                    int m, int n, int k, bool ta, bool tb, cudaStream_t stream)
{
    if constexpr (std::is_same_v<T1, __half> && std::is_same_v<T2, float>)
        launch_wmma_half(A,B,C,m,n,k,ta,tb,stream);
    else if constexpr (std::is_same_v<T1, signed char> && std::is_same_v<T2, int32_t>)
        launch_wmma_int8(A,B,C,m,n,k,ta,tb,stream);
}

template <typename T1, typename T2>
void cpu_gemm(const T1* A, const T1* B, T2* C,
              int m, int n, int k, bool ta, bool tb)
{
    for (int ni = 0; ni < n; ni++)
    for (int mi = 0; mi < m; mi++)
    {
        T2 sum = 0;
        for (int ki = 0; ki < k; ki++)
        {
            T1 av = ta ? A[mi*k + ki] : A[ki*m + mi];
            T1 bv = tb ? B[ki*n + ni] : B[ni*k + ki];
            sum += av * bv;
        }
        C[ni*m + mi] = sum;
    }
}

int main()
{
    constexpr int m = 1024, n = 1024, k = 1024;
    constexpr int kGpuWarmup = 2;
    constexpr int kGpuIters = 10;

    cudaStream_t stream;
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream));

    cudaEvent_t ev_start, ev_stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&ev_start));
    CHECK_CUDA_ERROR(cudaEventCreate(&ev_stop));

    // ---------------- HMMA Test ----------------
    std::vector<float> A_host(m*k), B_host(k*n), C_cpu(m*n), C_gpu(m*n);
    std::vector<__half> A_half(m*k), B_half(k*n);

    std::default_random_engine e(0);
    std::uniform_real_distribution<float> dist(-1,1);

    for (auto& x : A_host) x = dist(e);
    for (auto& x : B_host) x = dist(e);
    for (int i=0; i<m*k; i++) A_half[i] = __float2half(A_host[i]);
    for (int i=0; i<k*n; i++) B_half[i] = __float2half(B_host[i]);

    __half *dA, *dB; float *dC;
    CHECK_CUDA_ERROR(cudaMalloc(&dA, m*k*sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dB, k*n*sizeof(__half)));
    CHECK_CUDA_ERROR(cudaMalloc(&dC, m*n*sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(dA, A_half.data(), m*k*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dB, B_half.data(), k*n*sizeof(__half), cudaMemcpyHostToDevice));

    for (bool ta : {false, true})
    for (bool tb : {false, true})
    {
        auto const t_cpu0 = std::chrono::high_resolution_clock::now();
        cpu_gemm(A_host.data(), B_host.data(), C_cpu.data(), m, n, k, ta, tb);
        auto const t_cpu1 = std::chrono::high_resolution_clock::now();
        double const cpu_ms = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();

        for (int w = 0; w < kGpuWarmup; ++w)
        {
            launch_wmma_mm(dA, dB, dC, m, n, k, ta, tb, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }

        CHECK_CUDA_ERROR(cudaEventRecord(ev_start, stream));
        for (int it = 0; it < kGpuIters; ++it)
            launch_wmma_mm(dA, dB, dC, m, n, k, ta, tb, stream);
        CHECK_CUDA_ERROR(cudaEventRecord(ev_stop, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        float gpu_ms_total = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_ms_total, ev_start, ev_stop));
        float const gpu_ms = gpu_ms_total / static_cast<float>(kGpuIters);

        CHECK_CUDA_ERROR(cudaMemcpy(C_gpu.data(), dC, m * n * sizeof(float), cudaMemcpyDeviceToHost));

        float err = 0;
        for (int i = 0; i < m * n; i++)
            err += fabsf(C_gpu[i] - C_cpu[i]);
        std::cout << std::fixed << std::setprecision(4) << "FP16 | ta=" << ta << " tb=" << tb
                  << " | err=" << err / m / n << " | cpu_ms=" << cpu_ms
                  << " | gpu_avg_ms=" << gpu_ms << " (" << kGpuIters << " iters)\n";
    }

    // ---------------- IMMA Test ----------------
    std::vector<signed char> Ai(m*k), Bi(k*n);
    std::vector<int32_t> Ci_cpu(m*n), Ci_gpu(m*n);
    std::uniform_int_distribution<int> idist(-2,2);
    for (auto& x : Ai) x = idist(e);
    for (auto& x : Bi) x = idist(e);

    signed char *dAi, *dBi; int32_t *dCi;
    CHECK_CUDA_ERROR(cudaMalloc(&dAi, m*k*sizeof(signed char)));
    CHECK_CUDA_ERROR(cudaMalloc(&dBi, k*n*sizeof(signed char)));
    CHECK_CUDA_ERROR(cudaMalloc(&dCi, m*n*sizeof(int32_t)));
    CHECK_CUDA_ERROR(cudaMemcpy(dAi, Ai.data(), m*k*sizeof(signed char), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(dBi, Bi.data(), k*n*sizeof(signed char), cudaMemcpyHostToDevice));

    for (bool ta : {false, true})
    for (bool tb : {false, true})
    {
        auto const t_cpu0 = std::chrono::high_resolution_clock::now();
        cpu_gemm(Ai.data(), Bi.data(), Ci_cpu.data(), m, n, k, ta, tb);
        auto const t_cpu1 = std::chrono::high_resolution_clock::now();
        double const cpu_ms = std::chrono::duration<double, std::milli>(t_cpu1 - t_cpu0).count();

        for (int w = 0; w < kGpuWarmup; ++w)
        {
            launch_wmma_mm(dAi, dBi, dCi, m, n, k, ta, tb, stream);
            CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        }

        CHECK_CUDA_ERROR(cudaEventRecord(ev_start, stream));
        for (int it = 0; it < kGpuIters; ++it)
            launch_wmma_mm(dAi, dBi, dCi, m, n, k, ta, tb, stream);
        CHECK_CUDA_ERROR(cudaEventRecord(ev_stop, stream));
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream));
        float gpu_ms_total = 0.f;
        CHECK_CUDA_ERROR(cudaEventElapsedTime(&gpu_ms_total, ev_start, ev_stop));
        float const gpu_ms = gpu_ms_total / static_cast<float>(kGpuIters);

        CHECK_CUDA_ERROR(cudaMemcpy(Ci_gpu.data(), dCi, m * n * sizeof(int32_t), cudaMemcpyDeviceToHost));

        bool ok = true;
        for (int i = 0; i < m * n; i++)
            if (Ci_gpu[i] != Ci_cpu[i])
            {
                ok = false;
                break;
            }
        std::cout << std::fixed << std::setprecision(4) << "INT8 | ta=" << ta << " tb=" << tb
                  << " | ok=" << ok << " | cpu_ms=" << cpu_ms << " | gpu_avg_ms=" << gpu_ms << " ("
                  << kGpuIters << " iters)\n";
    }

    CHECK_CUDA_ERROR(cudaEventDestroy(ev_start));
    CHECK_CUDA_ERROR(cudaEventDestroy(ev_stop));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dAi);
    cudaFree(dBi);
    cudaFree(dCi);
    cudaStreamDestroy(stream);
    return 0;
}