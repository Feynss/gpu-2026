#include "gelu_cuda.h"

#include <algorithm>
#include <cuda_runtime.h>

namespace {

constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2 / pi)
constexpr float kBeta = 0.044715f;
constexpr int kBlockSize = 256;

__global__ void GeluKernel(float* data, size_t size) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = data[i];
        float inner = kAlpha * (x + kBeta * x * x * x);
        // tanh(y) = 1 - 2 / (exp(2y) + 1) -- avoids a direct tanh() call
        float t = 1.0f - 2.0f / (expf(2.0f * inner) + 1.0f);
        data[i] = 0.5f * x * (1.0f + t);
    }
}

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const size_t size = input.size();
    if (size == 0) {
        return {};
    }
    const size_t bytes = size * sizeof(float);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Pinned (page-locked) host memory is required for a memcpy to actually
    // run asynchronously relative to the host; with regular pageable
    // memory the driver silently serializes the copy.
    float* h_pinned = nullptr;
    cudaMallocHost(&h_pinned, bytes);
    std::copy(input.begin(), input.end(), h_pinned);

    // Device memory is allocated once and reused in place for both the
    // input and the output of the kernel.
    float* d_data = nullptr;
    cudaMalloc(&d_data, bytes);

    cudaMemcpyAsync(d_data, h_pinned, bytes, cudaMemcpyHostToDevice, stream);

    const int grid = static_cast<int>((size + kBlockSize - 1) / kBlockSize);
    GeluKernel<<<grid, kBlockSize, 0, stream>>>(d_data, size);

    // The host-side result vector is allocated here, while the H2D copy
    // and the kernel above are already queued and running on the GPU --
    // this is the host-allocation / device-computation overlap.
    std::vector<float> output(size);

    cudaMemcpyAsync(h_pinned, d_data, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    std::copy(h_pinned, h_pinned + size, output.begin());

    cudaFree(d_data);
    cudaFreeHost(h_pinned);
    cudaStreamDestroy(stream);

    return output;
}
