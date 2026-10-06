#ifndef NAIVE_GEMM_CUDA_H
#define NAIVE_GEMM_CUDA_H

#include <vector>

std::vector<float> NaiveGemmCUDA(const std::vector<float>& left_matrix,
                               const std::vector<float>& right_matrix,
                               int matrix_size);

#endif
