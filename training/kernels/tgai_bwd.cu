/* TGAI grouped MoE backward, fp32, sm_70. Device-only (NVRTC).
 *
 * Expert GEMMs are not in this file. The eight experts share M/N/K and are
 * already packed on the leading dimension, so the host issues one
 * cublasSgemmStridedBatched per logical product (gemm_tn_batched /
 * gemm_nt_batched / sgemm_batched) instead of one cublasSgemm per expert.
 * This file is the elementwise half of that path:
 *   silu_mul_bwd  - one launch over every expert row
 *   rmsnorm_bwd   - one launch over the grouped rows
 * Launch both on the cuBLAS stream. The second NT product uses beta=1 so
 * dh = dg @ Wg^T + du @ Wu^T needs no extra add kernel.
 */

__device__ __forceinline__ float blk_sum(float v, float* smem)
{
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float out = smem[0];
    __syncthreads();
    return out;
}

/* y = x * rsqrt(mean(x^2)+eps) * w.  One block per token. */
extern "C" __global__ void rmsnorm_bwd(const float* dy, const float* x, const float* w,
                                       float* dx, float* dw, int seq, int d, float eps)
{
    __shared__ float smem[256];
    const int t = (int)blockIdx.x;
    if (t >= seq)
        return;

    const int tid = (int)threadIdx.x;
    const int stride = (int)blockDim.x;
    const float* xrow = x + t * d;
    const float* dyrow = dy + t * d;
    float* dxrow = dx + t * d;

    float sumsq = 0.f;
    float dot = 0.f;
    for (int i = tid; i < d; i += stride) {
        const float xi = xrow[i];
        sumsq += xi * xi;
        dot += dyrow[i] * xi * w[i];
    }
    sumsq = blk_sum(sumsq, smem);
    dot = blk_sum(dot, smem);

    const float inv_d = 1.f / (float)d;
    const float rstd = rsqrtf(sumsq * inv_d + eps);
    const float scale = rstd * rstd * rstd * inv_d;

    for (int i = tid; i < d; i += stride) {
        const float xi = xrow[i];
        const float dyi = dyrow[i];
        dxrow[i] = rstd * dyi * w[i] - scale * xi * dot;
        atomicAdd(dw + i, dyi * xi * rstd);
    }
}

/* y = silu(a) * b, silu(v) = v * sigmoid(v). */
extern "C" __global__ void silu_mul_bwd(const float* g, const float* a, const float* b,
                                        float* da, float* db, int n)
{
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n)
        return;
    const float av = a[i];
    const float sig = 1.f / (1.f + expf(-av));
    const float silu = av * sig;
    const float dsilu = sig * (1.f + av * (1.f - sig));
    const float gv = g[i];
    da[i] = gv * b[i] * dsilu;
    db[i] = gv * silu;
}

/* Cross-block sum of g*e, kept in fp64 so the final fp32 store matches the oracle. */
__device__ double g_moe_acc = 0.0;
__device__ int g_moe_cnt = 0;

/* out += weight * e.  de = weight * g; dweight = sum(g*e). */
extern "C" __global__ void moe_add_bwd(const float* g, const float* e, float* de,
                                       float* dweight, float weight, int seq, int d, int blk)
{
    __shared__ double sh[256];
    const int t = (int)blockIdx.x;
    const int i = (int)threadIdx.x + (int)blockIdx.y * blk;
    const int tid = (int)threadIdx.x;

    double p = 0.0;
    if (t < seq && i < d) {
        const int idx = t * d + i;
        const float gv = g[idx];
        de[idx] = weight * gv;
        p = (double)gv * (double)e[idx];
    }

    sh[tid] = p;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&g_moe_acc, sh[0]);
        __threadfence();
        const int nblocks = (int)gridDim.x * (int)gridDim.y * (int)gridDim.z;
        const int old = atomicAdd(&g_moe_cnt, 1);
        if (old == nblocks - 1) {
            dweight[0] = (float)g_moe_acc;
            g_moe_acc = 0.0;
            g_moe_cnt = 0;
        }
    }
}

/* out[t] = emb[ids[t]].  Repeated ids require atomicAdd. */
extern "C" __global__ void emb_gather_bwd(const float* g, const int* ids, float* demb,
                                         int seq, int d)
{
    const int t = (int)blockIdx.x;
    if (t >= seq)
        return;
    const int id = ids[t];
    const float* grow = g + t * d;
    float* erow = demb + id * d;
    for (int j = (int)threadIdx.x; j < d; j += (int)blockDim.x)
        atomicAdd(erow + j, grow[j]);
}

/* Inverse of the in-place pair rotation (transpose of [[c,-s],[s,c]]). */
__device__ __forceinline__ void rope_pair(float* v, int e, float c, float s)
{
    const float ev = v[e];
    const float od = v[e + 1];
    v[e] = ev * c + od * s;
    v[e + 1] = -ev * s + od * c;
}

extern "C" __global__ void rope_bwd(float* dq, float* dk, int seq, int n_heads, int d_head,
                                    int cache_pos, const float* cos_tab,
                                    const float* sin_tab, int max_seq)
{
    const int t = (int)blockIdx.x;
    const int h = (int)blockIdx.y;
    if (t >= seq || h >= n_heads)
        return;

    int pos = cache_pos + t;
    if (pos >= max_seq)
        pos = max_seq - 1;
    if (pos < 0)
        pos = 0;

    const int half = d_head >> 1;
    const int base = (t * n_heads + h) * d_head;
    const int tab = pos * half;
    for (int i = (int)threadIdx.x; i < half; i += (int)blockDim.x) {
        const float c = cos_tab[tab + i];
        const float s = sin_tab[tab + i];
        const int e = base + (i << 1);
        rope_pair(dq, e, c, s);
        rope_pair(dk, e, c, s);
    }
}

/* x += y  =>  dx = g. */
extern "C" __global__ void residual_bwd(const float* g, float* dx, int n)
{
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n)
        dx[i] = g[i];
}

/* Softmax-over-selected-k jacobian. ids/w/dW rows are `stride` apart.
 * dlogits and dbias are pre-zeroed. Ties are already resolved in ids. */
extern "C" __global__ void moe_route_bwd(const float* dW, const int* ids, const float* w,
                                        float* dlogits, float* dbias,
                                        int n_experts, int top_k, int stride)
{
    if (threadIdx.x != 0 || top_k <= 0)
        return;

    const int t = (int)blockIdx.x;
    const int row = t * stride;
    const int lb = t * n_experts;

    float dot = 0.f;
    for (int k = 0; k < top_k; ++k)
        dot += w[row + k] * dW[row + k];

    for (int k = 0; k < top_k; ++k) {
        const int e = ids[row + k];
        const float ds = w[row + k] * (dW[row + k] - dot);
        if ((unsigned)e < (unsigned)n_experts) {
            dlogits[lb + e] += ds;
            atomicAdd(dbias + e, ds);
        }
    }
}
