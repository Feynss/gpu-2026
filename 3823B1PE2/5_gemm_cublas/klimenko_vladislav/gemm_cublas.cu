#include "gemm_cublas.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstddef>
#include <cstring>
#include <vector>

namespace
{

    struct Workspace
    {
        float *h_a = nullptr;
        float *h_b = nullptr;
        float *h_c = nullptr;
        float *d_a = nullptr;
        float *d_b = nullptr;
        float *d_c = nullptr;
        size_t cap = 0;
        cudaStream_t stream = nullptr;
        cublasHandle_t handle = nullptr;

        void ensure(size_t bytes)
        {
            if (stream == nullptr)
            {
                cudaStreamCreate(&stream);
            }
            if (handle == nullptr)
            {
                cublasCreate(&handle);
                cublasSetStream(handle, stream);
            }
            if (bytes <= cap)
                return;

            if (h_a)
            {
                cudaFreeHost(h_a);
                cudaFreeHost(h_b);
                cudaFreeHost(h_c);
            }
            if (d_a)
            {
                cudaFree(d_a);
                cudaFree(d_b);
                cudaFree(d_c);
            }

            cudaMallocHost(&h_a, bytes);
            cudaMallocHost(&h_b, bytes);
            cudaMallocHost(&h_c, bytes);
            cudaMalloc(&d_a, bytes);
            cudaMalloc(&d_b, bytes);
            cudaMalloc(&d_c, bytes);

            cap = bytes;
        }
    };

    Workspace &workspace()
    {
        static Workspace ws;
        return ws;
    }

} // namespace

std::vector<float> GemmCUBLAS(const std::vector<float> &a,
                              const std::vector<float> &b,
                              int n)
{
    if (n <= 0)
        return {};

    const size_t elements = static_cast<size_t>(n) * n;
    const size_t bytes = elements * sizeof(float);

    auto &ws = workspace();
    ws.ensure(bytes);

    std::memcpy(ws.h_a, a.data(), bytes);
    std::memcpy(ws.h_b, b.data(), bytes);

    cudaMemcpyAsync(ws.d_a, ws.h_a, bytes, cudaMemcpyHostToDevice, ws.stream);
    cudaMemcpyAsync(ws.d_b, ws.h_b, bytes, cudaMemcpyHostToDevice, ws.stream);

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm(ws.handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                ws.d_b, n,
                ws.d_a, n,
                &beta,
                ws.d_c, n);

    cudaMemcpyAsync(ws.h_c, ws.d_c, bytes, cudaMemcpyDeviceToHost, ws.stream);
    cudaStreamSynchronize(ws.stream);

    return std::vector<float>(ws.h_c, ws.h_c + elements);
}