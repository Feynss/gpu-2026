#include "gelu_omp.h"

#include <omp.h>
#include <cmath>
#include <vector>


std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size(), 0.0f);
    constexpr float CF = 0.044715f;
    constexpr float sqrt2oP = 0.7978845608028654f;
#pragma omp parallel for simd
    for (int i = 0; i < static_cast<int>(input.size()); i++) {
        float x = input[i];
        float x3 = x * x * x;
        float z = sqrt2oP * (x + CF * x3);
        float e2z = std::exp(2.0f * z);
        result[i] = 0.5f * x * (2.0f - (2.0f / (e2z + 1.0f)));
    }
    return result;
}