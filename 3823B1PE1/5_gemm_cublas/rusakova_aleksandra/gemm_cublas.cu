#include "gemm_cublas.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace {

struct DeviceWorkspace {
    float* a = nullptr;
    float* b = nullptr;
    float* c = nullptr;
    size_t cap = 0;
    cudaStream_t stream = nullptr;
    cublasHandle_t handle = nullptr;

    void ensure(size_t n) {
        if (stream == nullptr) {
            cudaStreamCreate(&stream);
        }
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetStream(handle, stream);
        }
        const size_t need = n * n;
        if (need <= cap) {
            return;
        }
        cudaFree(a);
        cudaFree(b);
        cudaFree(c);
        cudaMalloc(&a, need * sizeof(float));
        cudaMalloc(&b, need * sizeof(float));
        cudaMalloc(&c, need * sizeof(float));
        cap = need;
    }
};

DeviceWorkspace& workspace() {
    static DeviceWorkspace mem;
    return mem;
}

}  // namespace

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    if (n <= 0) {
        return {};
    }

    auto& d = workspace();
    d.ensure(static_cast<size_t>(n));

    const size_t bytes = static_cast<size_t>(n) * static_cast<size_t>(n) * sizeof(float);
    cudaMemcpyAsync(d.a, a.data(), bytes, cudaMemcpyHostToDevice, d.stream);
    cudaMemcpyAsync(d.b, b.data(), bytes, cudaMemcpyHostToDevice, d.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;
    // Row-major C = A * B is equivalent to column-major C = B * A
    // with the same pointers (no explicit transpose).
    cublasSgemm(d.handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d.b, n, d.a, n, &beta,
                d.c, n);

    std::vector<float> output(static_cast<size_t>(n) * static_cast<size_t>(n));
    cudaMemcpyAsync(output.data(), d.c, bytes, cudaMemcpyDeviceToHost, d.stream);
    cudaStreamSynchronize(d.stream);
    return output;
}
