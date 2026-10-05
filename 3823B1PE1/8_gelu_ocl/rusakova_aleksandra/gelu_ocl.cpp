#include "gelu_ocl.h"

#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

namespace {

const char kKernelSource[] = R"CLC(
__kernel void gelu_vec4(__global const float* in, __global float* out, int n4) {
    const int i = get_global_id(0);
    if (i >= n4) {
        return;
    }
    const float4 x = vload4(i, in);
    const float4 x2 = x * x;
    const float4 arg = -1.5957691216057308f * x * (1.0f + 0.044715f * x2);
    vstore4(x / (1.0f + native_exp(arg)), i, out);
}

__kernel void gelu_tail(__global const float* in, __global float* out, int start, int n) {
    const int i = get_global_id(0) + start;
    if (i >= n) {
        return;
    }
    const float x = in[i];
    const float x2 = x * x;
    out[i] = x / (1.0f + native_exp(-1.5957691216057308f * x * (1.0f + 0.044715f * x2)));
}
)CLC";

struct OclState {
    int platform_index = -1;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    cl_program program = nullptr;
    cl_kernel kernel_vec4 = nullptr;
    cl_kernel kernel_tail = nullptr;
    cl_mem d_in = nullptr;
    cl_mem d_out = nullptr;
    size_t cap = 0;

    void release_buffers() {
        if (d_in) {
            clReleaseMemObject(d_in);
            d_in = nullptr;
        }
        if (d_out) {
            clReleaseMemObject(d_out);
            d_out = nullptr;
        }
        cap = 0;
    }

    void release_all() {
        release_buffers();
        if (kernel_vec4) {
            clReleaseKernel(kernel_vec4);
            kernel_vec4 = nullptr;
        }
        if (kernel_tail) {
            clReleaseKernel(kernel_tail);
            kernel_tail = nullptr;
        }
        if (program) {
            clReleaseProgram(program);
            program = nullptr;
        }
        if (queue) {
            clReleaseCommandQueue(queue);
            queue = nullptr;
        }
        if (context) {
            clReleaseContext(context);
            context = nullptr;
        }
        platform_index = -1;
    }

    void init(int platform) {
        if (platform_index == platform && context != nullptr) {
            return;
        }
        release_all();

        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        cl_platform_id platforms[16];
        clGetPlatformIDs(num_platforms, platforms, nullptr);

        cl_device_id device = nullptr;
        clGetDeviceIDs(platforms[platform], CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);

        const char* src = kKernelSource;
        program = clCreateProgramWithSource(context, 1, &src, nullptr, nullptr);
        clBuildProgram(program, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
        kernel_vec4 = clCreateKernel(program, "gelu_vec4", nullptr);
        kernel_tail = clCreateKernel(program, "gelu_tail", nullptr);
        platform_index = platform;
    }

    void ensure(size_t n) {
        if (n <= cap) {
            return;
        }
        release_buffers();
        d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, n * sizeof(float), nullptr, nullptr);
        d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr, nullptr);
        cap = n;
    }
};

OclState& state() {
    static OclState s;
    return s;
}

}  // namespace

std::vector<float> GeluOCL(const std::vector<float>& input, int platform) {
    const int n = static_cast<int>(input.size());
    if (n == 0) {
        return {};
    }

    auto& s = state();
    s.init(platform);
    s.ensure(static_cast<size_t>(n));

    clEnqueueWriteBuffer(s.queue, s.d_in, CL_FALSE, 0, static_cast<size_t>(n) * sizeof(float),
                         input.data(), 0, nullptr, nullptr);

    const int n4 = n >> 2;
    if (n4 > 0) {
        clSetKernelArg(s.kernel_vec4, 0, sizeof(cl_mem), &s.d_in);
        clSetKernelArg(s.kernel_vec4, 1, sizeof(cl_mem), &s.d_out);
        clSetKernelArg(s.kernel_vec4, 2, sizeof(int), &n4);
        const size_t local = 256;
        const size_t global = (static_cast<size_t>(n4) + local - 1) / local * local;
        clEnqueueNDRangeKernel(s.queue, s.kernel_vec4, 1, nullptr, &global, &local, 0, nullptr,
                               nullptr);
    }

    const int rem = n & 3;
    if (rem > 0) {
        const int start = n4 << 2;
        clSetKernelArg(s.kernel_tail, 0, sizeof(cl_mem), &s.d_in);
        clSetKernelArg(s.kernel_tail, 1, sizeof(cl_mem), &s.d_out);
        clSetKernelArg(s.kernel_tail, 2, sizeof(int), &start);
        clSetKernelArg(s.kernel_tail, 3, sizeof(int), &n);
        const size_t global = static_cast<size_t>(rem);
        clEnqueueNDRangeKernel(s.queue, s.kernel_tail, 1, nullptr, &global, nullptr, 0, nullptr,
                               nullptr);
    }

    std::vector<float> output(static_cast<size_t>(n));
    clEnqueueReadBuffer(s.queue, s.d_out, CL_TRUE, 0, static_cast<size_t>(n) * sizeof(float),
                        output.data(), 0, nullptr, nullptr);
    return output;
}
