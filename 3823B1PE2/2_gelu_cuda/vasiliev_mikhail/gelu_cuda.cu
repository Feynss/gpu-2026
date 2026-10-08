#include "gelu_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

namespace {

constexpr float kA = -1.5957691216f;
constexpr float kB = kA * 0.044715f;

constexpr size_t kChunk = size_t(1) << 22;
constexpr int kStreams = 4;
constexpr int kBlock = 256;

__global__ void GeluKernel(const float* __restrict__ in,
                           float* __restrict__ out, size_t n) {
    size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        const float x = in[i];
        out[i] = x / (1.0f + __expf(x * (kA + kB * x * x)));
    }
}

struct Context {
    float* d_in = nullptr;
    float* d_out = nullptr;
    float* h_in = nullptr;
    float* h_out = nullptr;
    size_t capacity = 0;
    cudaStream_t streams[kStreams];
    cudaEvent_t* events = nullptr;
    size_t events_cap = 0;
    bool streams_ready = false;

    void Init() {
        if (streams_ready) return;
        for (int s = 0; s < kStreams; ++s)
            cudaStreamCreateWithFlags(&streams[s], cudaStreamNonBlocking);
        streams_ready = true;
    }

    void Reserve(size_t n) {
        Init();
        if (n > capacity) {
            cudaDeviceSynchronize();
            if (d_in) cudaFree(d_in);
            if (d_out) cudaFree(d_out);
            if (h_in) cudaFreeHost(h_in);
            if (h_out) cudaFreeHost(h_out);
            const size_t bytes = n * sizeof(float);
            cudaMalloc(&d_in, bytes);
            cudaMalloc(&d_out, bytes);
            cudaMallocHost(&h_in, bytes);
            cudaMallocHost(&h_out, bytes);
            capacity = n;
        }
        const size_t chunks = (n + kChunk - 1) / kChunk;
        if (chunks > events_cap) {
            if (events) {
                for (size_t c = 0; c < events_cap; ++c)
                    cudaEventDestroy(events[c]);
                delete[] events;
            }
            events = new cudaEvent_t[chunks];
            for (size_t c = 0; c < chunks; ++c)
                cudaEventCreateWithFlags(&events[c], cudaEventDisableTiming);
            events_cap = chunks;
        }
    }
};

Context g_ctx;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t n = input.size();
    if (n == 0) return {};

    Context& ctx = g_ctx;
    ctx.Reserve(n);

    const size_t chunks = (n + kChunk - 1) / kChunk;

    for (size_t c = 0; c < chunks; ++c) {
        const size_t off = c * kChunk;
        const size_t len = std::min(kChunk, n - off);
        const size_t bytes = len * sizeof(float);
        cudaStream_t st = ctx.streams[c % kStreams];

        std::memcpy(ctx.h_in + off, input.data() + off, bytes);
        cudaMemcpyAsync(ctx.d_in + off, ctx.h_in + off, bytes, cudaMemcpyHostToDevice, st);
        const int grid = static_cast<int>((len + kBlock - 1) / kBlock);
        GeluKernel<<<grid, kBlock, 0, st>>>(ctx.d_in + off, ctx.d_out + off,
                                            len);
        cudaMemcpyAsync(ctx.h_out + off, ctx.d_out + off, bytes, cudaMemcpyDeviceToHost, st);
        cudaEventRecord(ctx.events[c], st);
    }

    std::vector<float> output(n);

    for (size_t c = 0; c < chunks; ++c) {
        const size_t off = c * kChunk;
        const size_t len = std::min(kChunk, n - off);
        cudaEventSynchronize(ctx.events[c]);
        std::memcpy(output.data() + off, ctx.h_out + off, len * sizeof(float));
    }

    return output;
}
