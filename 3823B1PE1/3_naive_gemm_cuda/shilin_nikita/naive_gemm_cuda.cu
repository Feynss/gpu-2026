#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 8;
constexpr int kRowsPerThread = 4;
constexpr int kColsPerThread = 8;

__global__ void NaiveGemmKernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int n) {
    const int col0 = (blockIdx.x * kBlockX + threadIdx.x) * kColsPerThread;
    const int row0 = (blockIdx.y * kBlockY + threadIdx.y) * kRowsPerThread;

    float acc[kRowsPerThread][kColsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
#pragma unroll
        for (int j = 0; j < kColsPerThread; ++j) {
            acc[i][j] = 0.0f;
        }
    }

    const float* arow[kRowsPerThread];
#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        arow[i] = a + (row0 + i) * n;
    }

    for (int k = 0; k < n; ++k) {
        float av[kRowsPerThread];
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
            av[i] = __ldg(arow[i] + k);
        }
        const float4 b0 = __ldg(reinterpret_cast<const float4*>(b + k * n + col0));
        const float4 b1 = __ldg(reinterpret_cast<const float4*>(b + k * n + col0 + 4));
#pragma unroll
        for (int i = 0; i < kRowsPerThread; ++i) {
            acc[i][0] = fmaf(av[i], b0.x, acc[i][0]);
            acc[i][1] = fmaf(av[i], b0.y, acc[i][1]);
            acc[i][2] = fmaf(av[i], b0.z, acc[i][2]);
            acc[i][3] = fmaf(av[i], b0.w, acc[i][3]);
            acc[i][4] = fmaf(av[i], b1.x, acc[i][4]);
            acc[i][5] = fmaf(av[i], b1.y, acc[i][5]);
            acc[i][6] = fmaf(av[i], b1.z, acc[i][6]);
            acc[i][7] = fmaf(av[i], b1.w, acc[i][7]);
        }
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        float* c_ptr = c + (row0 + i) * n + col0;
        *reinterpret_cast<float4*>(c_ptr) = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
        *reinterpret_cast<float4*>(c_ptr + 4) =
            make_float4(acc[i][4], acc[i][5], acc[i][6], acc[i][7]);
    }
}

__global__ void NaiveGemmSmallKernel(const float* __restrict__ a, const float* __restrict__ b,
                                     float* __restrict__ c, int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = 0; k < n; ++k) {
        sum = fmaf(a[row * n + k], b[k * n + col], sum);
    }
    c[row * n + col] = sum;
}

struct DeviceCache {
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    float* h_pin_c = nullptr;
    const float* registered_a = nullptr;
    const float* registered_b = nullptr;
    cudaStream_t stream = nullptr;
    std::size_t cap = 0;
    std::size_t pin_cap = 0;
};

DeviceCache& Cache() {
    static DeviceCache cache;
    return cache;
}

void RegisterPtr(const float*& slot, const float* ptr, std::size_t bytes) {
    if (slot == ptr) {
        return;
    }
    if (slot != nullptr) {
        cudaHostUnregister(const_cast<float*>(slot));
        slot = nullptr;
    }
    if (cudaHostRegister(const_cast<float*>(ptr), bytes, cudaHostRegisterDefault) == cudaSuccess) {
        slot = ptr;
    }
}

void Ensure(std::size_t n) {
    DeviceCache& cache = Cache();
    if (cache.stream == nullptr) {
        cudaStreamCreateWithFlags(&cache.stream, cudaStreamNonBlocking);
    }
    const std::size_t need = n * n;
    if (cache.cap < need) {
        if (cache.d_a != nullptr) {
            cudaFree(cache.d_a);
            cudaFree(cache.d_b);
            cudaFree(cache.d_c);
        }
        const std::size_t bytes = need * sizeof(float);
        cudaMalloc(&cache.d_a, bytes);
        cudaMalloc(&cache.d_b, bytes);
        cudaMalloc(&cache.d_c, bytes);
        cache.cap = need;
    }
    if (cache.pin_cap < need) {
        if (cache.h_pin_c != nullptr) {
            cudaFreeHost(cache.h_pin_c);
        }
        cudaMallocHost(&cache.h_pin_c, need * sizeof(float));
        cache.pin_cap = need;
    }
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a, const std::vector<float>& b, int n) {
    if (n <= 0) {
        return {};
    }

    Ensure(static_cast<std::size_t>(n));
    DeviceCache& cache = Cache();
    const std::size_t bytes = static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * sizeof(float);
    RegisterPtr(cache.registered_a, a.data(), bytes);
    RegisterPtr(cache.registered_b, b.data(), bytes);

    cudaMemcpyAsync(cache.d_a, a.data(), bytes, cudaMemcpyHostToDevice, cache.stream);
    cudaMemcpyAsync(cache.d_b, b.data(), bytes, cudaMemcpyHostToDevice, cache.stream);

    const int tile_m = kBlockY * kRowsPerThread;
    const int tile_n = kBlockX * kColsPerThread;
    if (n >= tile_m && (n % tile_m) == 0 && (n % tile_n) == 0) {
        dim3 block(kBlockX, kBlockY);
        dim3 grid(n / tile_n, n / tile_m);
        NaiveGemmKernel<<<grid, block, 0, cache.stream>>>(cache.d_a, cache.d_b, cache.d_c, n);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (n + 15) / 16);
        NaiveGemmSmallKernel<<<grid, block, 0, cache.stream>>>(cache.d_a, cache.d_b, cache.d_c, n);
    }

    std::vector<float> output(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    cudaMemcpyAsync(cache.h_pin_c, cache.d_c, bytes, cudaMemcpyDeviceToHost, cache.stream);
    cudaStreamSynchronize(cache.stream);
    std::memcpy(output.data(), cache.h_pin_c, bytes);
    return output;
}
