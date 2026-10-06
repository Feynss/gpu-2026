#include "gelu_omp.h"

#include <cmath>
#include <omp.h>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    // sqrt(2 / pi)
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;

    const size_t size = input.size();
    std::vector<float> output(size);

    const float* in = input.data();
    float* out = output.data();

    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < size; ++i) {
        float x = in[i];
        float inner = kAlpha * (x + kBeta * x * x * x);
        float t = 1.0f - 2.0f / (std::exp(2.0f * inner) + 1.0f);
        out[i] = 0.5f * x * (1.0f + t);
    }

    return output;
}
