#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <cassert>

#include <cuda_runtime.h>
#include "naive_gemm_cuda.h"


#define CHECK_ERROR(X)                                              \
    do {                                                            \
        cudaError_t err_ = (X);                                     \
        if (err_ != cudaSuccess) {                                  \
            fprintf(stderr, "CUDA error at %s:%d: '%s' -> %s\n",    \
                    __FILE__, __LINE__, #X,                         \
                    cudaGetErrorString(err_));                      \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while (0)

namespace {

constexpr int kBlockX = 16;
constexpr int kBlockY = 16;

constexpr int kTile = 4;  
constexpr int kBlockDimX = kBlockX * kTile; // 64 columns per block
constexpr int kBlockDimY = kBlockY * kTile; // 64 rows per block

__device__ __forceinline__ float4 load_float4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

__global__ void naive_gemm_reg4x4_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;

    const int row_start = by * kBlockDimY + ty * kTile;
    const int col_start = bx * kBlockDimX + tx * kTile;

    if (row_start >= n || col_start >= n) return;

    float4 acc0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc1 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc2 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 acc3 = make_float4(0.f, 0.f, 0.f, 0.f);

    const float* a_row0 = a + static_cast<std::size_t>(row_start + 0) * n;
    const float* a_row1 = a + static_cast<std::size_t>(row_start + 1) * n;
    const float* a_row2 = a + static_cast<std::size_t>(row_start + 2) * n;
    const float* a_row3 = a + static_cast<std::size_t>(row_start + 3) * n;

    for (int k = 0; k < n; k += 4) {
        float4 a0 = load_float4(a_row0 + k);
        float4 a1 = load_float4(a_row1 + k);
        float4 a2 = load_float4(a_row2 + k);
        float4 a3 = load_float4(a_row3 + k);

        const float* b_row0 = b + static_cast<std::size_t>(k + 0) * n + col_start;
        const float* b_row1 = b + static_cast<std::size_t>(k + 1) * n + col_start;
        const float* b_row2 = b + static_cast<std::size_t>(k + 2) * n + col_start;
        const float* b_row3 = b + static_cast<std::size_t>(k + 3) * n + col_start;

        float4 b0 = load_float4(b_row0);
        float4 b1 = load_float4(b_row1);
        float4 b2 = load_float4(b_row2);
        float4 b3 = load_float4(b_row3);

        // Row 0
        acc0.x += a0.x * b0.x + a0.y * b1.x + a0.z * b2.x + a0.w * b3.x;
        acc0.y += a0.x * b0.y + a0.y * b1.y + a0.z * b2.y + a0.w * b3.y;
        acc0.z += a0.x * b0.z + a0.y * b1.z + a0.z * b2.z + a0.w * b3.z;
        acc0.w += a0.x * b0.w + a0.y * b1.w + a0.z * b2.w + a0.w * b3.w;

        // Row 1
        acc1.x += a1.x * b0.x + a1.y * b1.x + a1.z * b2.x + a1.w * b3.x;
        acc1.y += a1.x * b0.y + a1.y * b1.y + a1.z * b2.y + a1.w * b3.y;
        acc1.z += a1.x * b0.z + a1.y * b1.z + a1.z * b2.z + a1.w * b3.z;
        acc1.w += a1.x * b0.w + a1.y * b1.w + a1.z * b2.w + a1.w * b3.w;

        // Row 2
        acc2.x += a2.x * b0.x + a2.y * b1.x + a2.z * b2.x + a2.w * b3.x;
        acc2.y += a2.x * b0.y + a2.y * b1.y + a2.z * b2.y + a2.w * b3.y;
        acc2.z += a2.x * b0.z + a2.y * b1.z + a2.z * b2.z + a2.w * b3.z;
        acc2.w += a2.x * b0.w + a2.y * b1.w + a2.z * b2.w + a2.w * b3.w;

        // Row 3
        acc3.x += a3.x * b0.x + a3.y * b1.x + a3.z * b2.x + a3.w * b3.x;
        acc3.y += a3.x * b0.y + a3.y * b1.y + a3.z * b2.y + a3.w * b3.y;
        acc3.z += a3.x * b0.z + a3.y * b1.z + a3.z * b2.z + a3.w * b3.z;
        acc3.w += a3.x * b0.w + a3.y * b1.w + a3.z * b2.w + a3.w * b3.w;
    }

    float* c_row0 = c + static_cast<std::size_t>(row_start + 0) * n + col_start;
    float* c_row1 = c + static_cast<std::size_t>(row_start + 1) * n + col_start;
    float* c_row2 = c + static_cast<std::size_t>(row_start + 2) * n + col_start;
    float* c_row3 = c + static_cast<std::size_t>(row_start + 3) * n + col_start;

    *reinterpret_cast<float4*>(c_row0) = acc0;
    *reinterpret_cast<float4*>(c_row1) = acc1;
    *reinterpret_cast<float4*>(c_row2) = acc2;
    *reinterpret_cast<float4*>(c_row3) = acc3;
}

// for n < 4
__global__ void naive_gemm_scalar_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         float* __restrict__ c,
                                         int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.f;
        for (int k = 0; k < n; ++k) {
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

}  // namespace

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n <= 0) {
        return {};
    }

    assert((n % kTile == 0 || n < 4) && "n must be a multiple of 4 or less than 4");

    const std::size_t elem_count = static_cast<std::size_t>(n) * n;
    const std::size_t bytes = elem_count * sizeof(float);

    std::vector<float> c(elem_count);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    CHECK_ERROR(cudaMalloc(&d_a, bytes));
    CHECK_ERROR(cudaMalloc(&d_b, bytes));
    CHECK_ERROR(cudaMalloc(&d_c, bytes));

    CHECK_ERROR(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice));

    if (n >= 4) {
        const dim3 block(kBlockX, kBlockY);
        const dim3 grid((n + kBlockDimX - 1) / kBlockDimX,
                        (n + kBlockDimY - 1) / kBlockDimY);
        naive_gemm_reg4x4_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    } else {
        const int block_size = 16;
        const dim3 block(block_size, block_size);
        const dim3 grid((n + block_size - 1) / block_size,
                        (n + block_size - 1) / block_size);
        naive_gemm_scalar_kernel<<<grid, block>>>(d_a, d_b, d_c, n);
    }
    
    CHECK_ERROR(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_a));
    CHECK_ERROR(cudaFree(d_b));
    CHECK_ERROR(cudaFree(d_c));

    return c;
}