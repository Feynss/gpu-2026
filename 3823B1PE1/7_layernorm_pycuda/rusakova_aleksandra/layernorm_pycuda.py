import numpy as np

KERNEL = r"""
extern "C" __global__
void layernorm_kernel(const float* __restrict__ in,
                      const float* __restrict__ gamma,
                      const float* __restrict__ beta,
                      float* __restrict__ out,
                      int row_size,
                      float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;
    const int n4 = row_size >> 2;

    const float4* in4 = reinterpret_cast<const float4*>(in + (size_t)row * row_size);
    const float4* g4 = reinterpret_cast<const float4*>(gamma);
    const float4* b4 = reinterpret_cast<const float4*>(beta);
    float4* out4 = reinterpret_cast<float4*>(out + (size_t)row * row_size);

    float local_sum = 0.f;
    float local_sumsq = 0.f;
    for (int i = tid; i < n4; i += nthreads) {
        const float4 v = in4[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sumsq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    __shared__ float ssum[256];
    __shared__ float ssumsq[256];
    ssum[tid] = local_sum;
    ssumsq[tid] = local_sumsq;
    __syncthreads();

    for (int s = nthreads >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssumsq[tid] += ssumsq[tid + s];
        }
        __syncthreads();
    }

    const float inv_n = 1.0f / (float)row_size;
    const float mean = ssum[0] * inv_n;
    float var = ssumsq[0] * inv_n - mean * mean;
    var = fmaxf(var, 0.f);
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < n4; i += nthreads) {
        const float4 v = in4[i];
        const float4 g = g4[i];
        const float4 bt = b4[i];
        float4 y;
        y.x = g.x * (v.x - mean) * inv_std + bt.x;
        y.y = g.y * (v.y - mean) * inv_std + bt.y;
        y.z = g.z * (v.z - mean) * inv_std + bt.z;
        y.w = g.w * (v.w - mean) * inv_std + bt.w;
        out4[i] = y;
    }
}
"""

_module = None
_func = None
_d_in = None
_d_out = None
_d_gamma = None
_d_beta = None
_cap = 0
_gcap = 0


def _get_func():
    global _module, _func
    if _func is None:
        import pycuda.autoinit  # noqa: F401
        from pycuda.compiler import SourceModule

        _module = SourceModule(KERNEL, options=["-O3"])
        _func = _module.get_function("layernorm_kernel")
    return _func


def _ensure_buffers(n, row_size):
    global _d_in, _d_out, _d_gamma, _d_beta, _cap, _gcap
    import pycuda.driver as cuda

    if n > _cap:
        _d_in = cuda.mem_alloc(n * 4)
        _d_out = cuda.mem_alloc(n * 4)
        _cap = n
    if row_size > _gcap:
        _d_gamma = cuda.mem_alloc(row_size * 4)
        _d_beta = cuda.mem_alloc(row_size * 4)
        _gcap = row_size
    return _d_in, _d_out, _d_gamma, _d_beta


def layernorm_pycuda(input, gamma, beta, row_size, eps=1e-5):
    """
    Apply Layer Normalization to each row of the input matrix.

    Parameters
    ----------
    input : list or numpy.ndarray of float
        Flattened matrix in row‑major order. Its length must be divisible by row_size.
    gamma : list or numpy.ndarray of float
        Scale parameter, length = row_size.
    beta : list or numpy.ndarray of float
        Shift parameter, length = row_size.
    row_size : int
        Number of features per row (i.e., number of columns).
    eps : float, optional
        Small constant for numerical stability.

    Returns
    -------
    numpy.ndarray
        Flattened matrix of the same shape as input, containing the row‑wise
        normalized results.
    """
    import pycuda.driver as cuda

    x = np.asarray(input, dtype=np.float32)
    g = np.ascontiguousarray(np.asarray(gamma, dtype=np.float32))
    b = np.ascontiguousarray(np.asarray(beta, dtype=np.float32))
    row_size = int(row_size)

    if x.size == 0:
        return np.empty(0, dtype=np.float32)

    x = np.ascontiguousarray(x)
    row_count = x.size // row_size
    out = np.empty_like(x)

    func = _get_func()
    d_in, d_out, d_gamma, d_beta = _ensure_buffers(x.size, row_size)
    cuda.memcpy_htod(d_in, x)
    cuda.memcpy_htod(d_gamma, g)
    cuda.memcpy_htod(d_beta, b)

    func(
        d_in,
        d_gamma,
        d_beta,
        d_out,
        np.int32(row_size),
        np.float32(eps),
        block=(256, 1, 1),
        grid=(int(row_count), 1),
    )
    cuda.memcpy_dtoh(out, d_out)
    return out
