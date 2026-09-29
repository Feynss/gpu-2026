#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace
{

    constexpr int kBlockX = 32;
    constexpr int kBlockY = 8;
    constexpr int kVec = 4;

    __global__ void GemmKernel(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int n)
    {
        const int col = (blockIdx.x * blockDim.x + threadIdx.x) * kVec;

        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= n || col >= n)
            return;

        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

#pragma unroll 4
        for (int k = 0; k < n; ++k)
        {
            const float av = a[row * n + k];

            const float4 bv = *reinterpret_cast<const float4 *>(b + k * n + col);

            acc.x += av * bv.x;
            acc.y += av * bv.y;
            acc.z += av * bv.z;
            acc.w += av * bv.w;
        }

        *reinterpret_cast<float4 *>(c + row * n + col) = acc;
    }

    __global__ void GemmScalar(const float *__restrict__ a, const float *__restrict__ b, float *__restrict__ c, int n)
    {
        const int col = blockIdx.x * blockDim.x + threadIdx.x;

        const int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row >= n || col >= n)
            return;

        float sum = 0.0f;

#pragma unroll 4
        for (int k = 0; k < n; ++k)
            sum += a[row * n + k] * b[k * n + col];

        c[row * n + col] = sum;
    }

} // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float> &a, const std::vector<float> &b, int n)
{
    if (n <= 0)
        return {};

    const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);

    static float *d_a = nullptr;
    static float *d_b = nullptr;
    static float *d_c = nullptr;
    static int capacity = 0;

    if (n > capacity)
    {
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);

        cudaMalloc(&d_a, bytes);
        cudaMalloc(&d_b, bytes);
        cudaMalloc(&d_c, bytes);

        capacity = n;
    }

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);

    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    constexpr dim3 block(kBlockX, kBlockY);

    if (n >= kVec)
    {
        const dim3 grid((n / kVec + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);

        GemmKernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    else
    {
        const dim3 grid((n + kBlockX - 1) / kBlockX, (n + kBlockY - 1) / kBlockY);

        GemmScalar<<<grid, block>>>(d_a, d_b, d_c, n);
    }

    std::vector<float> c(static_cast<std::size_t>(n) * n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    return c;
}