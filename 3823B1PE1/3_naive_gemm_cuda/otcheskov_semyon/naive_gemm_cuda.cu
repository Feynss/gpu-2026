#include <vector>
#include <cstddef>
#include <cassert>

#include <cuda_runtime.h>
#include "naive_gemm_cuda.h"

namespace {

constexpr int kVec = 4;
constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

__device__ __forceinline__ float4 load_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__global__ void naive_gemm_vec4_kernel(const float* __restrict__ a,
                                       const float* __restrict__ b,
                                       float* __restrict__ c,
                                       int n) {

    const int col_quad = blockIdx.x * blockDim.x + threadIdx.x;

    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    const int col = col_quad * kVec;

    if (row >= n || col >= n) {
        return;
    }

    const std::size_t row_offset = static_cast<std::size_t>(row) * n;
    const float* a_row = a + row_offset;

    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < n; k += kVec) {
        const float4 a4 = load_float4(a_row + k);

        const std::size_t b_col_offset = static_cast<std::size_t>(k) * n + col;
        const std::size_t b_row_stride = static_cast<std::size_t>(n);

        const float4 b0 = load_float4(b + b_col_offset + 0 * b_row_stride);
        const float4 b1 = load_float4(b + b_col_offset + 1 * b_row_stride);
        const float4 b2 = load_float4(b + b_col_offset + 2 * b_row_stride);
        const float4 b3 = load_float4(b + b_col_offset + 3 * b_row_stride);

        // C[row][col..col+3] += A[row][k..k+3] * B[k..k+3][col..col+3]
        acc.x += a4.x * b0.x + a4.y * b1.x + a4.z * b2.x + a4.w * b3.x;
        acc.y += a4.x * b0.y + a4.y * b1.y + a4.z * b2.y + a4.w * b3.y;
        acc.z += a4.x * b0.z + a4.y * b1.z + a4.z * b2.z + a4.w * b3.z;
        acc.w += a4.x * b0.w + a4.y * b1.w + a4.z * b2.w + a4.w * b3.w;
    }

    *reinterpret_cast<float4*>(c + row_offset + col) = acc;
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    assert(n % kVec == 0 && "naive_gemm_vec4_kernel requires n % 4 == 0");

    const std::size_t elem_count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elem_count * sizeof(float);

    std::vector<float> c(elem_count);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block(kBlockX, kBlockY);
    const int col_quads = n / kVec;
    const dim3 grid((col_quads + kBlockX - 1) / kBlockX,
                    (n          + kBlockY - 1) / kBlockY);

    naive_gemm_vec4_kernel<<<grid, block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}