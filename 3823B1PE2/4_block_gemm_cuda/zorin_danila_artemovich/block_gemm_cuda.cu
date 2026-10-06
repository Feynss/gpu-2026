#include "block_gemm_cuda.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace {
constexpr int kTile = 16;

__global__ void BlockGemmKernel(const float* a, const float* b, float* c, int n) {
    __shared__ float tile_a[kTile][kTile];
    __shared__ float tile_b[kTile][kTile];
    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    float sum = 0.0f;

    for (int tile = 0; tile < n; tile += kTile) {
        tile_a[threadIdx.y][threadIdx.x] = (row < n && tile + threadIdx.x < n)
            ? a[row * n + tile + threadIdx.x] : 0.0f;
        tile_b[threadIdx.y][threadIdx.x] = (tile + threadIdx.y < n && col < n)
            ? b[(tile + threadIdx.y) * n + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n) {
        c[row * n + col] = sum;
    }
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}
} // namespace

std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n < 0 || a.size() != static_cast<std::size_t>(n) * n ||
        b.size() != static_cast<std::size_t>(n) * n) {
        throw std::invalid_argument("matrices must have exactly n*n elements");
    }
    std::vector<float> c(a.size(), 0.0f);
    if (n == 0) return c;

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    const std::size_t bytes = a.size() * sizeof(float);
    CheckCuda(cudaMalloc(&da, bytes), "cudaMalloc(A)");
    try {
        CheckCuda(cudaMalloc(&db, bytes), "cudaMalloc(B)");
        CheckCuda(cudaMalloc(&dc, bytes), "cudaMalloc(C)");
        CheckCuda(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice), "copy A");
        CheckCuda(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice), "copy B");
        const dim3 block(kTile, kTile);
        const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
        BlockGemmKernel<<<grid, block>>>(da, db, dc, n);
        CheckCuda(cudaGetLastError(), "BlockGemmKernel launch");
        CheckCuda(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost), "copy C");
    } catch (...) {
        cudaFree(dc); cudaFree(db); cudaFree(da);
        throw;
    }
    CheckCuda(cudaFree(dc), "cudaFree(C)");
    CheckCuda(cudaFree(db), "cudaFree(B)");
    CheckCuda(cudaFree(da), "cudaFree(A)");
    return c;
}
