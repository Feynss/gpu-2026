#include "gelu_cuda.h"

#include <cstdio>
#include <cstdlib>
#include <thread>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                        \
            std::fprintf(stderr, "CUDA error \"%s\" at %s:%d\n",          \
                         cudaGetErrorString(err_), __FILE__, __LINE__);   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

namespace {

    __global__ void GeluKernel(float* data, int n) {
        const float c1 = 1.5957691216057308f;   // 2 * sqrt(2 / pi)
        const float c2 = 0.0713548162726009f;   // c1 * 0.044715

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) {
            const float x = data[i];
            const float u = x * (c1 + c2 * x * x);
            data[i] = x / (1.0f + __expf(-u));
        }
    }
    struct DeviceBuffer {
        float* ptr = nullptr;
        size_t capacity = 0;

        float* Get(size_t n) {
            if (n > capacity) {
                if (ptr != nullptr) {
                    CUDA_CHECK(cudaFree(ptr));
                }
                CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(float)));
                capacity = n;
            }
            return ptr;
        }

        ~DeviceBuffer() {
            if (ptr != nullptr) {
                cudaFree(ptr);  
            }
        }
    };

    DeviceBuffer g_device_buffer;

}  // namespace

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* d_data = g_device_buffer.Get(n);
    std::vector<float> output;
    std::thread alloc_thread([&output, n] { output.resize(n); });

    CUDA_CHECK(cudaMemcpy(d_data, input.data(), bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    GeluKernel << <num_blocks, block_size >> > (d_data, n);
    CUDA_CHECK(cudaGetLastError());

    alloc_thread.join();

    CUDA_CHECK(cudaMemcpy(output.data(), d_data, bytes, cudaMemcpyDeviceToHost));

    return output;
}