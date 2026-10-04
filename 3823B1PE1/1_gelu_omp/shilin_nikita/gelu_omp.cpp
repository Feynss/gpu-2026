#include "gelu_omp.h"

#include <cmath>
#include <cstddef>

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const std::size_t n = input.size();
    if (n == 0) {
        return {};
    }

    std::vector<float> output(n);
    const float* __restrict__ in = input.data();
    float* __restrict__ out = output.data();

    constexpr float kA = 2.302208198144325f;
    constexpr float kB = 0.1029432395800235f;
    const std::ptrdiff_t nn = static_cast<std::ptrdiff_t>(n);

#pragma omp parallel for simd schedule(static) aligned(in, out : 16)
    for (std::ptrdiff_t i = 0; i < nn; ++i) {
        const float x = in[i];
        const float t = x * (kA + kB * x * x);
        out[i] = x / (1.0f + exp2f(-t));
    }

    return output;
}
