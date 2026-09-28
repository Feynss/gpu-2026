#include "softmax_cuda.h"

#include <cuda_runtime.h>

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ in, float* __restrict__ out,
                               int row_size) {
    const int row = blockIdx.x;
    const float* row_in = in + static_cast<size_t>(row) * row_size;
    float* row_out = out + static_cast<size_t>(row) * row_size;

    const int tid = threadIdx.x;
    const int n4 = row_size >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(row_in);
    float4* out4 = reinterpret_cast<float4*>(row_out);

    float local_max = -1e30f;
    for (int i = tid; i < n4; i += kThreads) {
        const float4 v = in4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }

    local_max = warp_reduce_max(local_max);
    __shared__ float warp_max[kThreads / 32];
    __shared__ float warp_sum[kThreads / 32];
    if ((tid & 31) == 0) {
        warp_max[tid >> 5] = local_max;
    }
    __syncthreads();
    if (tid < 32) {
        local_max = (tid < kThreads / 32) ? warp_max[tid] : -1e30f;
        local_max = warp_reduce_max(local_max);
        if (tid == 0) {
            warp_max[0] = local_max;
        }
    }
    __syncthreads();
    const float row_max = warp_max[0];

    float local_sum = 0.f;
    for (int i = tid; i < n4; i += kThreads) {
        const float4 v = in4[i];
        float4 e;
        e.x = __expf(v.x - row_max);
        e.y = __expf(v.y - row_max);
        e.z = __expf(v.z - row_max);
        e.w = __expf(v.w - row_max);
        out4[i] = e;
        local_sum += e.x + e.y + e.z + e.w;
    }

    local_sum = warp_reduce_sum(local_sum);
    if ((tid & 31) == 0) {
        warp_sum[tid >> 5] = local_sum;
    }
    __syncthreads();
    if (tid < 32) {
        local_sum = (tid < kThreads / 32) ? warp_sum[tid] : 0.f;
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) {
            warp_sum[0] = local_sum;
        }
    }
    __syncthreads();
    const float inv_sum = 1.0f / warp_sum[0];

    for (int i = tid; i < n4; i += kThreads) {
        float4 e = out4[i];
        e.x *= inv_sum;
        e.y *= inv_sum;
        e.z *= inv_sum;
        e.w *= inv_sum;
        out4[i] = e;
    }
}

struct DeviceWorkspace {
    float* in = nullptr;
    float* out = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;

    void ensure(size_t n) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }
        if (n <= cap) {
            return;
        }
        cudaFree(in);
        cudaFree(out);
        cudaMalloc(&in, n * sizeof(float));
        cudaMalloc(&out, n * sizeof(float));
        cap = n;
    }
};

DeviceWorkspace& workspace() {
    static DeviceWorkspace mem;
    return mem;
}

}  // namespace

std::vector<float> SoftmaxCUDA(const std::vector<float>& input, int row_count) {
    if (row_count <= 0 || input.empty()) {
        return {};
    }

    const int row_size = static_cast<int>(input.size() / static_cast<size_t>(row_count));
    auto& d = workspace();
    d.ensure(input.size());

    const size_t bytes = input.size() * sizeof(float);
    cudaMemcpyAsync(d.in, input.data(), bytes, cudaMemcpyHostToDevice, d.stream);
    softmax_kernel<<<row_count, kThreads, 0, d.stream>>>(d.in, d.out, row_size);

    std::vector<float> output(input.size());
    cudaMemcpyAsync(output.data(), d.out, bytes, cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);
    return output;
}
