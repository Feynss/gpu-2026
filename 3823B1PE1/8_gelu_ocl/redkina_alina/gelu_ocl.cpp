#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>

namespace {

constexpr const char* kSrc = R"CLC(
inline float Gelu(float x) {
    float z = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    float t = 1.0f - 2.0f / (exp(2.0f * z) + 1.0f);
    return 0.5f * x * (1.0f + t);
}

__kernel void gelu4(__global const float4* in, __global float4* out, int n4) {
    int i = get_global_id(0);
    if (i >= n4) return;
    float4 x = in[i];
    x.x = Gelu(x.x);
    x.y = Gelu(x.y);
    x.z = Gelu(x.z);
    x.w = Gelu(x.w);
    out[i] = x;
}

__kernel void gelu(__global const float* in, __global float* out, int n) {
    int i = get_global_id(0);
    if (i < n) out[i] = Gelu(in[i]);
}
)CLC";

struct State {
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel k4 = nullptr;
    cl_kernel k1 = nullptr;
    cl_mem in = nullptr;
    cl_mem out = nullptr;
    int platform = -1;
    size_t cap = 0;

    void Clear() {
        if (k4) clReleaseKernel(k4);
        if (k1) clReleaseKernel(k1);
        if (program) clReleaseProgram(program);
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        if (queue) clReleaseCommandQueue(queue);
        if (ctx) clReleaseContext(ctx);
        *this = State{};
    }

    bool Init(int plat) {
        if (ctx && platform == plat) {
            return true;
        }
        Clear();

        cl_uint count = 0;
        if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS || static_cast<cl_uint>(plat) >= count) {
            return false;
        }
        std::vector<cl_platform_id> ids(count);
        clGetPlatformIDs(count, ids.data(), nullptr);

        cl_device_id device = nullptr;
        if (clGetDeviceIDs(ids[plat], CL_DEVICE_TYPE_GPU, 1, &device, nullptr) != CL_SUCCESS) {
            return false;
        }

        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueueWithProperties(ctx, device, nullptr, nullptr);
        const char* src = kSrc;
        program = clCreateProgramWithSource(ctx, 1, &src, nullptr, nullptr);
        if (!ctx || !queue || !program || clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr) != CL_SUCCESS) {
            Clear();
            return false;
        }
        k4 = clCreateKernel(program, "gelu4", nullptr);
        k1 = clCreateKernel(program, "gelu", nullptr);
        platform = plat;
        return k4 && k1;
    }

    bool Ensure(size_t bytes) {
        if (in && bytes <= cap) {
            return true;
        }
        if (in) clReleaseMemObject(in);
        if (out) clReleaseMemObject(out);
        in = clCreateBuffer(ctx, CL_MEM_READ_ONLY, bytes, nullptr, nullptr);
        out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, bytes, nullptr, nullptr);
        cap = in && out ? bytes : 0;
        return in && out;
    }
};

State& Runtime() {
    static State state;
    return state;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const size_t n = input.size();
    if (n == 0) {
        return {};
    }

    State& rt = Runtime();
    const size_t bytes = n * sizeof(float);
    if (!rt.Init(platform) || !rt.Ensure(bytes)) {
        return {};
    }

    clEnqueueWriteBuffer(rt.queue, rt.in, CL_FALSE, 0, bytes, input.data(), 0, nullptr, nullptr);

    const bool vec = n % 4 == 0;
    cl_kernel kernel = vec ? rt.k4 : rt.k1;
    int count = static_cast<int>(vec ? n / 4 : n);
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &rt.in);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &rt.out);
    clSetKernelArg(kernel, 2, sizeof(int), &count);

    const size_t local = 256;
    const size_t global = (static_cast<size_t>(count) + local - 1) / local * local;
    clEnqueueNDRangeKernel(rt.queue, kernel, 1, nullptr, &global, &local, 0, nullptr, nullptr);

    std::vector<float> output(n);
    clEnqueueReadBuffer(rt.queue, rt.out, CL_TRUE, 0, bytes, output.data(), 0, nullptr, nullptr);
    return output;
}
