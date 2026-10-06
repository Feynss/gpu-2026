#include "naive_gemm_cuda.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace {

constexpr int rows_per_thread = 8;
constexpr int columns_per_thread = 4;

void CheckCUDA(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct DeviceMatrices {
    float* left_matrix = nullptr;
    float* right_matrix = nullptr;
    float* result_matrix = nullptr;

    ~DeviceMatrices() {
        cudaFree(left_matrix);
        cudaFree(right_matrix);
        cudaFree(result_matrix);
    }
};

__global__ void SmallGemm(const float* left_matrix,
                          const float* right_matrix,
                          float* result_matrix, int matrix_size) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= matrix_size || column >= matrix_size) {
        return;
    }

    float sum = 0.0f;
    for (int inner_index = 0; inner_index < matrix_size; ++inner_index) {
        float left_value = left_matrix[row * matrix_size + inner_index];
        float right_value = right_matrix[inner_index * matrix_size + column];
        sum = fmaf(left_value, right_value, sum);
    }
    result_matrix[row * matrix_size + column] = sum;
}

__global__ void NaiveGemm(const float* __restrict__ left_matrix,
                          const float* __restrict__ right_matrix,
                          float* __restrict__ result_matrix, int matrix_size) {
    int first_column = (blockIdx.x * blockDim.x + threadIdx.x) * columns_per_thread;
    int first_row = (blockIdx.y * blockDim.y + threadIdx.y) * rows_per_thread;
    float sums[rows_per_thread][columns_per_thread] = {};

    for (int inner_index = 0; inner_index < matrix_size; ++inner_index) {
        float4 right_values = *reinterpret_cast<const float4*>(
            right_matrix + inner_index * matrix_size + first_column);
        #pragma unroll
        for (int row_offset = 0; row_offset < rows_per_thread; ++row_offset) {
            float left_value = left_matrix[(first_row + row_offset) * matrix_size + inner_index];
            sums[row_offset][0] = fmaf(left_value, right_values.x, sums[row_offset][0]);
            sums[row_offset][1] = fmaf(left_value, right_values.y, sums[row_offset][1]);
            sums[row_offset][2] = fmaf(left_value, right_values.z, sums[row_offset][2]);
            sums[row_offset][3] = fmaf(left_value, right_values.w, sums[row_offset][3]);
        }
    }

    #pragma unroll
    for (int row_offset = 0; row_offset < rows_per_thread; ++row_offset) {
        *reinterpret_cast<float4*>(
            result_matrix + (first_row + row_offset) * matrix_size + first_column) =
            make_float4(sums[row_offset][0], sums[row_offset][1],
                        sums[row_offset][2], sums[row_offset][3]);
    }
}

}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& left_matrix,
                               const std::vector<float>& right_matrix,
                               int matrix_size) {
    size_t element_count = static_cast<size_t>(matrix_size) * matrix_size;
    size_t byte_count = element_count * sizeof(float);
    DeviceMatrices device_matrices;
    CheckCUDA(cudaMalloc(&device_matrices.left_matrix, byte_count));
    CheckCUDA(cudaMalloc(&device_matrices.right_matrix, byte_count));
    CheckCUDA(cudaMalloc(&device_matrices.result_matrix, byte_count));
    CheckCUDA(cudaMemcpy(device_matrices.left_matrix, left_matrix.data(),
                         byte_count, cudaMemcpyHostToDevice));
    CheckCUDA(cudaMemcpy(device_matrices.right_matrix, right_matrix.data(),
                         byte_count, cudaMemcpyHostToDevice));

    dim3 block_size(32, 4);
    if (matrix_size < 128) {
        dim3 grid_size((matrix_size + 31) / 32, (matrix_size + 3) / 4);
        SmallGemm<<<grid_size, block_size>>>(device_matrices.left_matrix,
                                            device_matrices.right_matrix,
                                            device_matrices.result_matrix, matrix_size);
    } else {
        dim3 grid_size(matrix_size / (block_size.x * columns_per_thread),
                       matrix_size / (block_size.y * rows_per_thread));
        NaiveGemm<<<grid_size, block_size>>>(device_matrices.left_matrix,
                                            device_matrices.right_matrix,
                                            device_matrices.result_matrix, matrix_size);
    }
    CheckCUDA(cudaGetLastError());

    std::vector<float> result_matrix(element_count);
    CheckCUDA(cudaMemcpy(result_matrix.data(), device_matrices.result_matrix,
                         byte_count, cudaMemcpyDeviceToHost));
    return result_matrix;
}
