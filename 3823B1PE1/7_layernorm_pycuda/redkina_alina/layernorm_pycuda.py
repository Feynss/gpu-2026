import numpy as np
import pycuda.autoinit  # noqa: F401
from pycuda import driver as cuda
from pycuda.compiler import SourceModule

_SRC = r"""
__device__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float buf[32];
    v = warp_sum(v);
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) buf[wid] = v;
    __syncthreads();
    const int nwarps = blockDim.x >> 5;
    float w = (lane < nwarps) ? buf[lane] : 0.0f;
    if (wid == 0) w = warp_sum(w);
    if (threadIdx.x == 0) buf[0] = w;
    __syncthreads();
    return buf[0];
}

__global__ void layernorm_cached(const float* __restrict__ in, const float* __restrict__ gamma,
                                 const float* __restrict__ beta, float* __restrict__ out,
                                 int n, float eps) {
    extern __shared__ float row[];
    const float* src = in + static_cast<size_t>(blockIdx.x) * n;
    float* dst = out + static_cast<size_t>(blockIdx.x) * n;

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        row[i] = src[i];
    __syncthreads();

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += row[i];
    const float mean = block_sum(sum) / n;

    float var = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = row[i] - mean;
        var += d * d;
    }
    const float inv = rsqrtf(block_sum(var) / n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = gamma[i] * (row[i] - mean) * inv + beta[i];
}

__global__ void layernorm_plain(const float* __restrict__ in, const float* __restrict__ gamma,
                                const float* __restrict__ beta, float* __restrict__ out,
                                int n, float eps) {
    const float* src = in + static_cast<size_t>(blockIdx.x) * n;
    float* dst = out + static_cast<size_t>(blockIdx.x) * n;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        sum += src[i];
    const float mean = block_sum(sum) / n;

    float var = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = src[i] - mean;
        var += d * d;
    }
    const float inv = rsqrtf(block_sum(var) / n + eps);

    for (int i = threadIdx.x; i < n; i += blockDim.x)
        dst[i] = gamma[i] * (src[i] - mean) * inv + beta[i];
}
"""

_THREADS = 256
_MAX_COLS = 16384
_state = {}


def _prepare():
    if "mod" in _state:
        return
    mod = SourceModule(_SRC, options=["-O3"])
    cached = mod.get_function("layernorm_cached")
    cached.set_attribute(cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, _MAX_COLS * 4)
    _state["mod"] = mod
    _state["cached"] = cached
    _state["plain"] = mod.get_function("layernorm_plain")


def _alloc(key, nbytes):
    buf, cap = _state.get(key, (None, 0))
    if buf is None or cap < nbytes:
        buf = cuda.mem_alloc(nbytes)
        _state[key] = (buf, nbytes)
    return buf


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    data = np.ascontiguousarray(input, dtype=np.float32)
    scale = np.ascontiguousarray(gamma, dtype=np.float32).ravel()
    shift = np.ascontiguousarray(beta, dtype=np.float32).ravel()
    flat = np.ascontiguousarray(data.ravel())
    n = int(row_size)
    if n <= 0 or flat.size == 0:
        return np.empty(data.shape, dtype=np.float32)

    _prepare()
    rows = flat.size // n
    d_in = _alloc("d_in", flat.nbytes)
    d_out = _alloc("d_out", flat.nbytes)
    d_gamma = _alloc("d_gamma", scale.nbytes)
    d_beta = _alloc("d_beta", shift.nbytes)

    cuda.memcpy_htod(d_in, flat)
    cuda.memcpy_htod(d_gamma, scale)
    cuda.memcpy_htod(d_beta, shift)

    fn = _state["cached"] if n <= _MAX_COLS else _state["plain"]
    shared = n * 4 if n <= _MAX_COLS else 0
    fn(d_in, d_gamma, d_beta, d_out, np.int32(n), np.float32(eps),
       block=(_THREADS, 1, 1), grid=(rows, 1), shared=shared)

    out = np.empty(flat.size, dtype=np.float32)
    cuda.memcpy_dtoh(out, d_out)
    return out.reshape(data.shape)
